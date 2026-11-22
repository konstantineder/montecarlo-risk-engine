import { Injectable } from "@nestjs/common";
import { spawn } from "child_process";
import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from "fs";
import { createInterface } from "readline";
import { join, resolve } from "path";

export interface RunCard {
  id: string;
  created: string;
  strategy: string;
  paths: number;
  seed: number;
  horizon_days: number;
  initial_dte: number;
  median: number;
  mean: number;
  p05: number;
  p95: number;
  prob_loss: number;
  deployed: number;
  lots: number;
}

@Injectable()
export class RunsService {
  private readonly root: string;
  private readonly repo: string;
  private readonly jobs = new Map<string, { status: string; run?: string; error?: string; log: string }>();

  constructor() {
    this.root = resolve(process.env.RUNS_DIR || join(__dirname, "..", "..", "..", "runs"));
    this.repo = resolve(this.root, "..");
  }

  list(): RunCard[] {
    if (!existsSync(this.root)) return [];
    const cards: RunCard[] = [];
    for (const name of readdirSync(this.root)) {
      const summary = join(this.root, name, "summary.json");
      if (!existsSync(summary)) continue;
      const raw = JSON.parse(readFileSync(summary, "utf8"));
      const mc = raw.monte_carlo || {};
      const cfg = raw.config || {};
      cards.push({
        id: raw.id || name,
        created: raw.created || statSync(summary).mtime.toISOString(),
        strategy: cfg.strategy || mc.strategy || "wheel",
        paths: mc.paths || cfg.paths || 0,
        seed: cfg.seed || 0,
        horizon_days: cfg.horizon_days || 0,
        initial_dte: cfg.initial_dte || cfg.put_dte || 0,
        median: mc.median || 0,
        mean: mc.mean || 0,
        p05: mc.p05 || 0,
        p95: mc.p95 || 0,
        prob_loss: mc.prob_loss || 0,
        deployed: raw.book?.deployed || 0,
        lots: cfg.lots || 0,
      });
    }
    return cards.sort((a, b) => (a.created < b.created ? 1 : -1));
  }

  get(id: string) {
    const file = join(this.root, id, "summary.json");
    if (!existsSync(file)) return null;
    const raw = JSON.parse(readFileSync(file, "utf8"));
    if (raw.median_path) {
      raw.median_path = {
        path: raw.median_path.path,
        pnl: raw.median_path.pnl,
        ending_spot: raw.median_path.ending_spot,
        ending_shares: raw.median_path.ending_shares,
      };
    }
    return raw;
  }

  paths(id: string) {
    const file = join(this.root, id, "paths.jsonl");
    if (!existsSync(file)) return null;
    return this.readLines(file, (row) => ({
      path: row.path,
      pnl: row.pnl,
      ending_spot: row.ending_spot,
      min_spot: row.min_spot,
      max_dd: row.max_dd,
      ending_shares: row.ending_shares,
      assignments: row.assignments,
      new_calls: row.new_calls,
      roll_cash: row.roll_cash,
      floor_after: row.floor_after,
      trades: Array.isArray(row.days)
        ? row.days.reduce((n: number, day: { trades?: unknown[] }) => n + (day.trades?.length || 0), 0)
        : 0,
    }));
  }

  path(id: string, index: number) {
    const file = join(this.root, id, "paths.jsonl");
    if (!existsSync(file) || Number.isNaN(index)) return null;
    return this.findLine(file, (row) => row.path === index);
  }

  series(id: string) {
    const file = join(this.root, id, "paths.jsonl");
    if (!existsSync(file)) return null;
    return this.aggregate(file);
  }

  simulate(body: Record<string, string>) {
    const id = Date.now().toString(36);
    const num = (key: string, fallback: number) => {
      const value = Number(body[key]);
      return Number.isFinite(value) ? value : fallback;
    };
    const text = (key: string, fallback: string) => {
      const value = String(body[key] || "").trim();
      return /^[0-9A-Za-z_.,:+\- ]+$/.test(value) ? value : fallback;
    };
    const strategy = body.strategy === "hold" ? "hold" : "wheel";
    const args = [
      "-m", "tqqq_book",
      "--lots", String(num("lots", 100)),
      "--horizon", String(num("horizon", 100)),
      "--paths", String(num("paths", 100)),
      "--seed", String(num("seed", 13)),
      "--spot", String(num("spot", 59.22)),
      "--iv", String(num("iv", 0.71)),
      "--strategy", strategy,
      "--puts", text("puts", "10:60:30,40:90:45,50:180:90"),
      "--otm", text("otm", "0.03,0.05,0.08,0.10"),
      "--contracts", text("contracts", "12,12,12,10"),
    ];
    const venv = join(this.repo, ".venv", "bin", "python");
    const python = existsSync(venv) ? venv : "python3";
    this.jobs.set(id, { status: "running", log: "" });
    const child = spawn(python, args, {
      cwd: this.repo,
      env: { ...process.env, PYTHONPATH: "src" },
    });
    child.stdout.on("data", (chunk) => {
      const job = this.jobs.get(id);
      if (job) job.log = (job.log + chunk.toString()).slice(-4000);
    });
    child.stderr.on("data", (chunk) => {
      const job = this.jobs.get(id);
      if (job) job.log = (job.log + chunk.toString()).slice(-4000);
    });
    child.on("close", (code) => {
      const job = this.jobs.get(id);
      if (!job) return;
      if (code === 0) {
        const newest = this.list()[0];
        job.status = "done";
        job.run = newest?.id;
      } else {
        job.status = "error";
        job.error = job.log || `exit ${code}`;
      }
    });
    return { job: id };
  }

  job(id: string) {
    return this.jobs.get(id) || null;
  }

  private aggregate(file: string) {
    return new Promise((resolveRows, reject) => {
      const byDay = new Map<number, { shares: number[]; puts: number[]; shortCalls: number[]; shortPuts: number[] }>();
      const stream = createReadStream(file);
      const lines = createInterface({ input: stream });
      lines.on("line", (line) => {
        if (!line.trim()) return;
        const row = JSON.parse(line);
        for (const day of row.days || []) {
          let shares = 0, puts = 0, shortCalls = 0, shortPuts = 0;
          for (const leg of day.position || []) {
            const qty = Number(leg.qty) || 0;
            if (leg.name === "share") shares = qty;
            else if (leg.name === "put" && leg.side === "long") puts += qty;
            else if (leg.name === "call" && leg.side === "short") shortCalls += qty;
            else if (leg.name === "put" && leg.side === "short") shortPuts += qty;
          }
          const bucket = byDay.get(day.day) || { shares: [], puts: [], shortCalls: [], shortPuts: [] };
          bucket.shares.push(shares / 100);
          bucket.puts.push(puts);
          bucket.shortCalls.push(shortCalls);
          bucket.shortPuts.push(shortPuts);
          byDay.set(day.day, bucket);
        }
      });
      lines.on("close", () => {
        const median = (values: number[]) => {
          const sorted = values.slice().sort((a, b) => a - b);
          return sorted[Math.floor(sorted.length / 2)] || 0;
        };
        resolveRows([...byDay.keys()].sort((a, b) => a - b).map((day) => {
          const bucket = byDay.get(day)!;
          return {
            day,
            shares: median(bucket.shares),
            puts: median(bucket.puts),
            shortCalls: median(bucket.shortCalls),
            shortPuts: median(bucket.shortPuts),
          };
        }));
      });
      lines.on("error", reject);
    });
  }

  private readLines(file: string, map: (row: any) => any): Promise<any[]> {
    return new Promise((resolveRows, reject) => {
      const rows: any[] = [];
      const stream = createReadStream(file);
      const lines = createInterface({ input: stream });
      lines.on("line", (line) => {
        if (!line.trim()) return;
        rows.push(map(JSON.parse(line)));
      });
      lines.on("close", () => resolveRows(rows));
      lines.on("error", reject);
    });
  }

  private findLine(file: string, match: (row: any) => boolean): Promise<any | null> {
    return new Promise((resolveRow, reject) => {
      const stream = createReadStream(file);
      const lines = createInterface({ input: stream });
      let found: any = null;
      lines.on("line", (line) => {
        if (found || !line.trim()) return;
        const row = JSON.parse(line);
        if (match(row)) {
          found = row;
          stream.destroy();
        }
      });
      lines.on("close", () => resolveRow(found));
      stream.on("close", () => resolveRow(found));
      lines.on("error", reject);
    });
  }
}
