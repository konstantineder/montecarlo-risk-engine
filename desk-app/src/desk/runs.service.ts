import { BadRequestException, Injectable } from "@nestjs/common";
import { spawn, spawnSync } from "child_process";
import { createReadStream, existsSync, openSync, readFileSync, readdirSync, statSync, writeFileSync, readSync, closeSync } from "fs";
import { createInterface } from "readline";
import { join, resolve } from "path";
import { DEFAULTS, MAX_SIMULATIONS, planSimulations, SCALARS, configurationInput } from "./simulation-plan";
import { SimulationQueue } from "./simulation-queue";
import { OptimizationService } from "./optimization-service";

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

interface RunMeta {
  index: any[];
  series: any[];
  offsets: Record<string, number>;
}

@Injectable()
export class RunsService {
  private readonly root: string;
  private readonly repo: string;
  private readonly queue: SimulationQueue;
  readonly optimizer: OptimizationService;
  private readonly memory = new Map<string, { stamp: number; value: any }>();
  private listStamp = 0;
  private listCards: RunCard[] | null = null;

  constructor() {
    this.repo = resolve(process.env.REPO_ROOT || join(__dirname, "..", "..", ".."));
    this.root = resolve(process.env.RUNS_DIR || join(this.repo, "runs"));
    this.queue = new SimulationQueue(this.repo, this.root);
    this.optimizer = new OptimizationService(this.root, this.queue);
  }

  list(limit = 20, offset = 0) {
    const cards = this.allCards();
    const size = Math.max(1, Math.min(200, limit));
    const start = Math.max(0, offset);
    return { items: cards.slice(start, start + size), total: cards.length, offset: start, limit: size };
  }

  private allCards(): RunCard[] {
    if (!existsSync(this.root)) return [];
    const stamp = this.folderStamp();
    if (this.listCards && this.listStamp === stamp) return this.listCards;
    const cards: RunCard[] = [];
    for (const name of readdirSync(this.root)) {
      const card = this.tile(name);
      if (card) cards.push(card);
    }
    cards.sort((a, b) => (a.created < b.created ? 1 : -1));
    this.listCards = cards;
    this.listStamp = stamp;
    return cards;
  }

  private folderStamp() {
    return readdirSync(this.root).reduce((sum, name) => {
      const file = join(this.root, name, "summary.json");
      return existsSync(file) ? sum + statSync(file).mtimeMs : sum;
    }, 0);
  }

  private tile(id: string): RunCard | null {
    const summary = join(this.root, id, "summary.json");
    if (!existsSync(summary)) return null;
    const stamp = statSync(summary).mtimeMs;
    const cached = this.memory.get(`tile:${id}`);
    if (cached && cached.stamp === stamp) return cached.value;
    const tileFile = join(this.root, id, "tile.json");
    if (existsSync(tileFile) && statSync(tileFile).mtimeMs >= stamp) {
      const card = JSON.parse(readFileSync(tileFile, "utf8"));
      this.memory.set(`tile:${id}`, { stamp, value: card });
      return card;
    }
    const raw = JSON.parse(readFileSync(summary, "utf8"));
    const mc = raw.monte_carlo || {};
    const cfg = raw.config || {};
    const card: RunCard = {
      id: raw.id || id,
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
    };
    try { writeFileSync(tileFile, JSON.stringify(card)); } catch { /* a read-only run still lists */ }
    this.memory.set(`tile:${id}`, { stamp, value: card });
    return card;
  }

  get(id: string) {
    const file = join(this.root, id, "summary.json");
    if (!existsSync(file)) return null;
    const stamp = statSync(file).mtimeMs;
    const cached = this.memory.get(`summary:${id}`);
    if (cached && cached.stamp === stamp) return cached.value;
    const raw = JSON.parse(readFileSync(file, "utf8"));
    if (raw.median_path) {
      raw.median_path = {
        path: raw.median_path.path,
        pnl: raw.median_path.pnl,
        ending_spot: raw.median_path.ending_spot,
        ending_shares: raw.median_path.ending_shares,
      };
    }
    this.memory.set(`summary:${id}`, { stamp, value: raw });
    return raw;
  }

  async paths(id: string) {
    const meta = await this.ensureMeta(id);
    return meta ? meta.index : null;
  }

  path(id: string, index: number) {
    const file = join(this.root, id, "paths.jsonl");
    if (!existsSync(file) || Number.isNaN(index)) return null;
    const stamp = statSync(file).mtimeMs;
    const key = `path:${id}:${index}`;
    const cached = this.memory.get(key);
    if (cached && cached.stamp === stamp) return Promise.resolve(cached.value);
    return this.ensureMeta(id).then((meta) => {
      const offset = meta?.offsets?.[String(index)];
      const row = offset == null ? null : this.readAt(file, offset);
      if (row) this.memory.set(key, { stamp, value: row });
      return row;
    });
  }

  async series(id: string) {
    const meta = await this.ensureMeta(id);
    return meta ? meta.series : null;
  }

  private async ensureMeta(id: string): Promise<RunMeta | null> {
    const file = join(this.root, id, "paths.jsonl");
    if (!existsSync(file)) return null;
    const stamp = statSync(file).mtimeMs;
    const cached = this.memory.get(`meta:${id}`);
    if (cached && cached.stamp === stamp) return cached.value;
    const indexFile = join(this.root, id, "index.json");
    const seriesFile = join(this.root, id, "series.json");
    const offsetFile = join(this.root, id, "offsets.json");
    if (existsSync(indexFile) && existsSync(seriesFile) && existsSync(offsetFile) && statSync(indexFile).mtimeMs >= stamp) {
      const meta = {
        index: JSON.parse(readFileSync(indexFile, "utf8")),
        series: JSON.parse(readFileSync(seriesFile, "utf8")),
        offsets: JSON.parse(readFileSync(offsetFile, "utf8")),
      };
      this.memory.set(`meta:${id}`, { stamp, value: meta });
      return meta;
    }
    const meta = await this.scan(file);
    try {
      writeFileSync(indexFile, JSON.stringify(meta.index));
      writeFileSync(seriesFile, JSON.stringify(meta.series));
      writeFileSync(offsetFile, JSON.stringify(meta.offsets));
    } catch { /* cache in memory even if the disk write fails */ }
    this.memory.set(`meta:${id}`, { stamp, value: meta });
    return meta;
  }

  private scan(file: string): Promise<RunMeta> {
    return new Promise((resolveRows, reject) => {
      const index: any[] = [];
      const offsets: Record<string, number> = {};
      const byDay = new Map<number, { shares: number[]; puts: number[]; shortCalls: number[]; shortPuts: number[]; floor: number[]; avg: number[] }>();
      const stream = createReadStream(file);
      const lines = createInterface({ input: stream });
      let offset = 0;
      lines.on("line", (line) => {
        const next = offset + Buffer.byteLength(line) + 1;
        if (line.trim()) {
          const row = JSON.parse(line);
          offsets[String(row.path)] = offset;
          let trades = 0;
        for (const day of row.days || []) {
          trades += day.trades?.length || 0;
          let shares = 0, puts = 0, shortCalls = 0, shortPuts = 0, floor = 0, weighted = 0;
          for (const leg of day.position || []) {
            const qty = Number(leg.qty) || 0;
            if (leg.name === "share") shares = qty;
            else if (leg.name === "put" && leg.side === "long") {
              puts += qty;
              weighted += qty * (Number(leg.strike) || 0);
              floor = floor === 0 ? Number(leg.strike) : Math.min(floor, Number(leg.strike));
            } else if (leg.name === "call" && leg.side === "short") shortCalls += qty;
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
        index.push({
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
          trades,
        });
        }
        offset = next;
      });
      lines.on("close", () => {
        const median = (values: number[]) => {
          const sorted = values.slice().sort((a, b) => a - b);
          return sorted[Math.floor(sorted.length / 2)] || 0;
        };
        resolveRows({
          index,
          offsets,
          series: [...byDay.keys()].sort((a, b) => a - b).map((day) => {
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
          }),
        });
      });
      lines.on("error", reject);
    });
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

  onApplicationShutdown() { this.queue.shutdown(); }
  pause(id: string) { try { return this.queue.pause(id); } catch (error) { throw new BadRequestException(error instanceof Error ? error.message : String(error)); } }
  resume(id: string) { try { return this.queue.resume(id); } catch (error) { throw new BadRequestException(error instanceof Error ? error.message : String(error)); } }
  jobs() { return this.queue.list(); }
  job(id: string) { return this.queue.get(id); }

  combineBatch(batch: string) {
    try { return this.queue.combineBatch(batch); }
    catch (error) { throw new BadRequestException(error instanceof Error ? error.message : String(error)); }
  }

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
        return this.queue.enqueueCombined((run.configurations || []).map((group: any) => ({ ...configurationInput(group.config), ...overrides })));
      } catch (error) {
        throw new BadRequestException(error instanceof Error ? error.message : String(error));
      }
    }
    return this.simulate({ ...configurationInput(run.config || {}), ...overrides });
  }

  private readAt(file: string, offset: number) {
    const handle = openSync(file, "r");
    try {
      const parts: Buffer[] = [];
      let at = offset;
      for (;;) {
        const chunk = Buffer.alloc(256 * 1024);
        const read = readSync(handle, chunk, 0, chunk.length, at);
        if (read <= 0) break;
        const slice = chunk.subarray(0, read);
        const end = slice.indexOf(10);
        parts.push(end === -1 ? slice : slice.subarray(0, end));
        if (end !== -1) break;
        at += read;
      }
      return JSON.parse(Buffer.concat(parts).toString("utf8"));
    } finally {
      closeSync(handle);
    }
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
