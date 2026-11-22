import { Injectable } from "@nestjs/common";
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

  constructor() {
    this.root = resolve(process.env.RUNS_DIR || join(__dirname, "..", "..", "..", "runs"));
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
