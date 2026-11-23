import { BadRequestException, NotFoundException } from "@nestjs/common";
import { randomUUID } from "crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "fs";
import { join } from "path";
import { createInterface } from "readline";
import { Configuration, configurationInput, parametersFor, planSimulations, SCALARS } from "./simulation-plan";
import { SimulationQueue } from "./simulation-queue";

export const RULES = ["harvest", "crash_roll", "keep_tenor", "roll_up", "roll_down", "put_ratio", "write_calls", "replace_assigned"];
export const STRATEGY_KEYS = ["otm", "contracts", "puts", "putRatioGrid", "rules", "strategy", "callPolicy", ...SCALARS.filter(p => ["Rules", "Dynamic"].includes(p.group)).map(p => p.key)];
const valid = (id: string) => /^[\w-]+$/.test(id);

export class OptimizationService {
  private contexts = new Map<string, any>();
  constructor(private root: string, private queue: SimulationQueue) {}

  private summary(id: string) {
    if (!valid(id) || !existsSync(join(this.root, id, "summary.json"))) throw new NotFoundException("No saved source simulation");
    return JSON.parse(readFileSync(join(this.root, id, "summary.json"), "utf8"));
  }

  private async path(id: string, index: number) {
    if (!Number.isSafeInteger(index) || index < 0) throw new BadRequestException("Choose a valid path index");
    const stream = createReadStream(join(this.root, id, "paths.jsonl"));
    const lines = createInterface({ input: stream });
    let n = 0;
    try { for await (const line of lines) { if (!line.trim()) continue; if (n++ === index) return JSON.parse(line); } }
    finally { lines.close(); stream.destroy(); }
    throw new NotFoundException("No saved path");
  }

  async context(id: string, index: number) {
    const cacheKey = `${id}/${index}`;
    if (this.contexts.has(cacheKey)) return this.contexts.get(cacheKey) as {sourceRun:string;path:number;base:Configuration;originalPnl:number;stockHold:number;rules:string[];strategyKeys:string[];parameters:ReturnType<typeof parametersFor>};
    const summary = this.summary(id), path = await this.path(id, index);
    const snapshot = path.config || summary.configurations?.find((g: any) => g.id === path.configuration)?.config || summary.config;
    const base = configurationInput(snapshot);
    const prices = path.prices || path.days.map((day: any) => day.price);
    const stockHold = Number(base.lots) * 100 * (prices[prices.length - 1] - prices[0]);
    const context = { sourceRun: id, path: index, base, originalPnl: path.pnl, stockHold, rules: RULES, strategyKeys: STRATEGY_KEYS,
      parameters: parametersFor(base).filter(p => STRATEGY_KEYS.includes(p.key.split(".")[0])) };
    this.contexts.set(cacheKey, context); return context;
  }

  async plan(id: string, index: number, body: Record<string, any>) {
    const context = await this.context(id, index);
    const ranges = body.ranges || [];
    if (!Array.isArray(ranges) || ranges.some((r: any) => !context.parameters.some(p => p.key === r.parameter))) throw new BadRequestException("Optimize strategy parameters; saved market assumptions and position size stay fixed.");
    const policies = body.policies || ["fixed", "adaptive"], strategies = body.strategies || ["wheel", "hold", "stock"], toggles = body.toggleRules || [];
    if (!Array.isArray(policies) || !policies.length || policies.some(p => !["fixed", "adaptive"].includes(p)) || !Array.isArray(strategies) || !strategies.length || strategies.some(s => !["wheel", "hold", "stock"].includes(s))) throw new BadRequestException("Choose valid strategies and call policies");
    if (!Array.isArray(toggles) || toggles.length > RULES.length || new Set(toggles).size !== toggles.length || toggles.some(r => !RULES.includes(r))) throw new BadRequestException("Choose valid optional rules");
    const objective = body.objective || "excess_hold", penalty = Number(body.drawdownPenalty ?? 1);
    if (!["excess_hold", "pnl", "risk_adjusted"].includes(objective) || !Number.isFinite(penalty) || penalty < 0) throw new BadRequestException("Choose a valid score and drawdown penalty");
    let plan;
    try { plan = planSimulations({ ...context.base, ranges, sweepMode: "separate" }); }
    catch (error) { throw new BadRequestException(error instanceof Error ? error.message : String(error)); }
    const candidates: Configuration[] = [], seen = new Set<string>();
    const add = (cfg: Configuration) => { const key = JSON.stringify(Object.keys(cfg).sort().map(k => [k, cfg[k]])); if (!seen.has(key)) { seen.add(key); candidates.push(cfg); } if (candidates.length > 1000) throw new BadRequestException("Search exceeds 1,000 candidates. Narrow ranges or vary fewer rules."); };
    add(planSimulations({ ...context.base, ranges: [], sweepMode: "separate" }).configurations[0]);
    for (const cfg of plan.configurations) for (const strategy of strategies) {
      if (strategy === "stock") { add({ ...plan.configurations[0], strategy, rules: "none", callPolicy: "fixed" }); continue; }
      for (const callPolicy of policies) for (let mask = 0; mask < 2 ** toggles.length; mask++) {
        const enabled = new Set(cfg.rules ? cfg.rules.split(",").filter(r => r !== "none") : RULES);
        toggles.forEach((rule, i) => { if (mask & (1 << i)) enabled.add(rule); else enabled.delete(rule); });
        add({ ...cfg, strategy, callPolicy, rules: RULES.filter(rule => enabled.has(rule)).join(",") || "none" });
      }
    }
    return { context, candidates, objective, penalty };
  }

  async preview(id: string, index: number, body: Record<string, unknown>) { const plan = await this.plan(id, index, body); return { candidates: plan.candidates.length, objective: plan.objective }; }
  async start(id: string, index: number, body: Record<string, unknown>) {
    const plan = await this.plan(id, index, body);
    return this.queue.enqueueTask("optimization", { sourceRun: id, path: index, candidates: plan.candidates, objective: plan.objective, drawdownPenalty: plan.penalty }, plan.context.base, plan.candidates.length);
  }

  result(id: string) {
    const file = join(this.root, "optimizations", id, "result.json");
    if (!valid(id) || !existsSync(file)) throw new NotFoundException("No completed optimization");
    return JSON.parse(readFileSync(file, "utf8"));
  }

  replay(id: string, body: Record<string, any>) {
    const result = this.result(id), candidate = result.candidates.find((c: any) => c.id === body.candidate);
    if (!candidate) throw new BadRequestException("Choose a searched candidate");
    const source = this.summary(result.source_run);
    const selected = body.scope === "path";
    if (!["path", "run"].includes(body.scope)) throw new BadRequestException("Choose selected path or whole simulation");
    return this.queue.enqueueTask("replay", { sourceRun: result.source_run, config: candidate.config, ...(selected ? { path: result.training_path } : {}), trainingPath: result.training_path, optimizationId: id, candidateId: candidate.id }, candidate.config, selected ? 1 : source.config.paths);
  }

  save(id: string, body: Record<string, any>) {
    const result = this.result(id), candidate = result.candidates.find((c: any) => c.id === body.candidate);
    if (!candidate) throw new BadRequestException("Choose a searched candidate");
    const entry = { id: randomUUID(), name: String(body.name || "Optimized strategy").trim().slice(0, 120), created: new Date().toISOString(), config: candidate.config, strategyKeys: STRATEGY_KEYS, optimization: id, trainingPath: result.training_path, sourceRun: result.source_run };
    const folder = join(this.root, "configurations"); mkdirSync(folder, { recursive: true });
    const file = join(folder, entry.id + ".json"); writeFileSync(file + ".tmp", JSON.stringify(entry)); renameSync(file + ".tmp", file);
    return entry;
  }
  configurations() { const folder = join(this.root, "configurations"); return existsSync(folder) ? readdirSync(folder).filter(f => f.endsWith(".json")).map(f => JSON.parse(readFileSync(join(folder, f), "utf8"))).sort((a, b) => b.created.localeCompare(a.created)) : []; }
}
