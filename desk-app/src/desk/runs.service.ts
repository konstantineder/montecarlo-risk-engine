import { BadRequestException, Injectable } from "@nestjs/common";
import { spawn, spawnSync } from "child_process";
import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from "fs";
import { createInterface } from "readline";
import { join, resolve } from "path";
import { DEFAULTS, MAX_SIMULATIONS, planSimulations, SCALARS } from "./simulation-plan";
import { SimulationQueue } from "./simulation-queue";

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
  mode: string;
  configuration_count: number;
}

@Injectable()
export class RunsService {
  private readonly root: string;
  private readonly repo: string;
  private readonly queue: SimulationQueue;

  constructor() {
    this.repo = resolve(process.env.REPO_ROOT || join(__dirname, "..", "..", ".."));
    this.root = resolve(process.env.RUNS_DIR || join(this.repo, "runs"));
    this.queue = new SimulationQueue(this.repo, this.root);
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
        mode: cfg.mode || "separate",
        configuration_count: cfg.configuration_count || 1,
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
      configuration: row.configuration,
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

  streamJson(id: string, res: any, kind: "full" | "mtm" | "book" = "full") {
    const summary = this.get(id);
    const file = join(this.root, id, "paths.jsonl");
    const name = kind === "full" ? `${id}.json` : `${id}.${kind}.json`;
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Content-Disposition", `attachment; filename="${name}"`);
    res.write(`{"kind":"${kind}","summary":${JSON.stringify(summary)},"paths":[`);
    let first = true;
    const stream = createReadStream(file);
    const lines = createInterface({ input: stream });
    lines.on("line", (line) => {
      if (!line.trim()) return;
      const text = kind === "full" ? line : JSON.stringify(this.slimPath(JSON.parse(line), kind));
      res.write(first ? text : `,${text}`);
      first = false;
    });
    lines.on("close", () => {
      res.end("]}");
    });
    lines.on("error", () => res.end());
  }

  private slimPath(row: any, kind: "mtm" | "book") {
    return {
      path: row.path,
      configuration: row.configuration,
      config: row.config,
      pnl: row.pnl,
      ending_spot: row.ending_spot,
      min_spot: row.min_spot,
      max_dd: row.max_dd,
      ending_shares: row.ending_shares,
      assignments: row.assignments,
      new_calls: row.new_calls,
      roll_cash: row.roll_cash,
      days: (row.days || []).map((day: any) => {
        const slim: any = {
          day: day.day,
          price: day.price,
          pnl: day.pnl,
          mtm: day.mtm,
          shares: day.shares,
          trades: day.trades || [],
        };
        if (kind === "book") slim.position = day.position || [];
        return slim;
      }),
    };
  }

  streamCsv(id: string, res: any) {
    const python = existsSync(join(this.repo, ".venv", "bin", "python"))
      ? join(this.repo, ".venv", "bin", "python")
      : "python3";
    const script = join(this.repo, "desk-app", "export_run.py");
    res.setHeader("Content-Type", "application/zip");
    res.setHeader("Content-Disposition", `attachment; filename="${id}.zip"`);
    const child = spawn(python, [script, join(this.root, id)]);
    child.stdout.pipe(res);
    child.stderr.on("data", () => undefined);
    child.on("error", () => res.end());
  }

  configuration() {
    return { defaults: DEFAULTS, scalars: SCALARS, maxSimulations: MAX_SIMULATIONS };
  }

  preview(body: Record<string, unknown>) {
    try {
      const { configurations, ...preview } = planSimulations(body);
      return { ...preview, sample: configurations.slice(0, 3) };
    } catch (error) {
      throw new BadRequestException(error instanceof Error ? error.message : String(error));
    }
  }

  simulate(body: Record<string, unknown>) {
    try {
      return this.queue.enqueue(body);
    } catch (error) {
      throw new BadRequestException(error instanceof Error ? error.message : String(error));
    }
  }

  jobs() { return this.queue.list(); }
  job(id: string) { return this.queue.get(id); }

  catalogue() {
    const python = existsSync(join(this.repo, ".venv", "bin", "python"))
      ? join(this.repo, ".venv", "bin", "python")
      : "python3";
    const code = "from tqqq_book.desk import BookConfig; from tqqq_book.strategy import catalogue; import json; print(json.dumps(catalogue(BookConfig())))";
    const out = spawnSync(python, ["-c", code], { cwd: this.repo, env: { ...process.env, PYTHONPATH: "src" } });
    if (out.status !== 0) return [];
    return JSON.parse(out.stdout.toString() || "[]");
  }

  reread(id: string, body: Record<string, string>) {
    const run = this.get(id);
    if (!run) return null;
    const overrides = Object.fromEntries(Object.entries(body).filter(([, value]) => value !== "" && value != null));
    if (run.config?.mode === "combined") {
      try {
        return this.queue.enqueueCombined((run.configurations || []).map((group: any) => ({ ...this.configurationInput(group.config), ...overrides })));
      } catch (error) {
        throw new BadRequestException(error instanceof Error ? error.message : String(error));
      }
    }
    return this.simulate({ ...this.configurationInput(run.config || {}), ...overrides });
  }

  private configurationInput(cfg: any): Record<string, string> {
    const sleeves = (cfg.put_sleeves || []).map((row: { weight: number; dte: number; roll_at: number }) =>
      `${Number((row.weight * 100).toPrecision(12))}:${row.dte}:${row.roll_at}`
    ).join(",");
    const tenors = cfg.tenors || [];
    const names: Record<string, string> = {
      horizon: "horizon_days", vol: "tqqq_vol", drift: "qqq_mu", initialDte: "put_dte",
      reserve: "reserve_shares", minSpotFrac: "stagger_min_spot_frac",
    };
    const scalarValues = Object.fromEntries(SCALARS.map((p) => {
      const name = names[p.key] || p.key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
      return [p.key, String(cfg[name] ?? p.value)];
    }));
    return {
      ...scalarValues,
      strategy: cfg.strategy || "wheel",
      puts: sleeves || DEFAULTS.puts,
      otm: tenors.map((row: { otm: number }) => row.otm).join(",") || DEFAULTS.otm,
      contracts: tenors.map((row: { contracts: number; contracts_at_100?: number }) => row.contracts_at_100 ?? Math.round(row.contracts * 100 / (cfg.lots || 100))).join(",") || DEFAULTS.contracts,
      putRatioGrid: (cfg.put_ratio_grid || []).map((row: number[]) => `${row[0] * 100}:${row[1]}`).join(",") || DEFAULTS.putRatioGrid,
      rules: Array.isArray(cfg.rules) ? cfg.rules.join(",") : "",
    };
  }

  private aggregate(file: string) {
    return new Promise((resolveRows, reject) => {
      const byDay = new Map<number, { shares: number[]; puts: number[]; shortCalls: number[]; shortPuts: number[]; floor: number[]; avg: number[] }>();
      const stream = createReadStream(file);
      const lines = createInterface({ input: stream });
      lines.on("line", (line) => {
        if (!line.trim()) return;
        const row = JSON.parse(line);
        for (const day of row.days || []) {
          let shares = 0, puts = 0, shortCalls = 0, shortPuts = 0;
          let floor = 0, weighted = 0;
          for (const leg of day.position || []) {
            const qty = Number(leg.qty) || 0;
            if (leg.name === "share") shares = qty;
            else if (leg.name === "put" && leg.side === "long") {
              puts += qty;
              weighted += qty * (Number(leg.strike) || 0);
              floor = floor === 0 ? Number(leg.strike) : Math.min(floor, Number(leg.strike));
            }
            else if (leg.name === "call" && leg.side === "short") shortCalls += qty;
            else if (leg.name === "put" && leg.side === "short") shortPuts += qty;
          }
          const bucket = byDay.get(day.day) || { shares: [], puts: [], shortCalls: [], shortPuts: [], floor: [], avg: [] };
          bucket.shares.push(shares / 100);
          bucket.puts.push(puts);
          bucket.shortCalls.push(shortCalls);
          bucket.shortPuts.push(shortPuts);
          bucket.floor.push(floor);
          bucket.avg.push(puts ? weighted / puts : 0);
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
            floor: median(bucket.floor),
            avg: median(bucket.avg),
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
