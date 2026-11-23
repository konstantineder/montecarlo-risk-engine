import { ChildProcess, spawn } from "child_process";
import { randomUUID } from "crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, unlinkSync, writeFileSync } from "fs";
import { join } from "path";
import { commandArguments, Configuration, planCombinedConfigurations, planSimulations, rangeValues } from "./simulation-plan";

export interface SimulationJob {
  id: string; batch: string; index: number; batchSize: number;
  status: "queued" | "running" | "pausing" | "paused" | "done" | "error";
  progress: number; phase: string; completedPaths: number; totalPaths: number;
  created: string; started?: string; finished?: string;
  config: Configuration; mode: string; configurationCount: number; configurations?: Configuration[];
  sourceRuns?: string[]; sourceBatch?: string; variations: Record<string, string>;
  run?: string; error?: string; log: string;
  kind?: "simulation" | "optimization" | "replay";
  payload?: Record<string, unknown>; optimization?: string;
}

export class SimulationQueue {
  private readonly jobs = new Map<string, SimulationJob>();
  private running = false;
  private stopping = false;
  private child?: ChildProcess;
  private lastWrite = 0;
  private readonly stateFile: string;

  constructor(private readonly repo: string, private readonly root: string, private readonly launch: typeof spawn = spawn, private readonly durable = launch === spawn) {
    this.stateFile = join(root, ".jobs", "jobs.json");
    if (durable && existsSync(this.stateFile)) {
      for (const job of JSON.parse(readFileSync(this.stateFile, "utf8")) as SimulationJob[]) {
        if (job.status === "running" || job.status === "pausing") {
          job.status = "paused";
          job.phase = "Interrupted; resume from the last completed path";
        }
        this.jobs.set(job.id, job);
      }
      this.persist(true);
      queueMicrotask(() => this.pump());
    }
  }

  private persist(force = false) {
    if (!this.durable || (!force && Date.now() - this.lastWrite < 1000)) return;
    mkdirSync(join(this.root, ".jobs"), { recursive: true });
    writeFileSync(this.stateFile + ".tmp", JSON.stringify([...this.jobs.values()]));
    renameSync(this.stateFile + ".tmp", this.stateFile);
    this.lastWrite = Date.now();
  }

  enqueue(input: Record<string, unknown>) { return this.enqueuePlan(planSimulations(input)); }
  enqueueCombined(configurations: Record<string, unknown>[]) { return this.enqueuePlan(planCombinedConfigurations(configurations)); }

  enqueueTask(kind: "optimization" | "replay", payload: Record<string, unknown>, config: Configuration, units: number) {
    return this.enqueuePlan(planSimulations({ ...config, paths: 1, ranges: [], sweepMode: "separate" }), { kind, payload, totalPaths: units, config, mode: kind });
  }

  combineBatch(batch: string) {
    const prior = this.list().find((job) => job.sourceBatch === batch && job.status !== "error");
    if (prior) return this.response(prior);
    const jobs = this.list().filter((job) => job.batch === batch).sort((a, b) => a.index - b.index);
    if (jobs.length < 2 || jobs.some((job) => job.status !== "done" || !job.run || job.mode === "combined")) throw new Error("Complete the separate simulations in this batch before combining their results.");
    const started = this.enqueueCombined(jobs.map((job) => job.config));
    Object.assign(this.jobs.get(started.job)!, { sourceRuns: jobs.map((job) => job.run!), sourceBatch: batch });
    this.persist(true);
    return started;
  }

  private response(job: SimulationJob) {
    return { job: job.id, batch: job.batch, total: 1, configurationCount: job.configurationCount, totalPaths: job.totalPaths, mode: job.mode };
  }

  private enqueuePlan(plan: ReturnType<typeof planSimulations>, metadata: Partial<SimulationJob> = {}) {
    const batch = randomUUID();
    const combined = plan.mode === "combined";
    const configs = combined ? [plan.configurations[0]] : plan.configurations;
    const ids: string[] = [];
    configs.forEach((config, index) => {
      const id = randomUUID(); ids.push(id);
      this.jobs.set(id, {
        id, batch, index: index + 1, batchSize: configs.length, kind: "simulation",
        mode: plan.mode, configurationCount: combined ? plan.total : 1,
        ...(combined ? { configurations: plan.configurations } : {}),
        status: "queued", progress: 0, phase: "Queued", completedPaths: 0,
        totalPaths: combined ? plan.totalPaths : Number(config.paths), created: new Date().toISOString(), config,
        variations: combined ? Object.fromEntries(plan.ranges.map((r) => [r.parameter, `${r.values[0]} to ${r.values[r.count - 1]} (${r.count} values)`])) : rangeValues(config, plan.ranges.map((r) => r.parameter)), log: "", ...metadata,
      });
    });
    this.persist(true);
    queueMicrotask(() => this.pump());
    return { job: ids[0], batch, total: configs.length, configurationCount: plan.total, totalPaths: plan.totalPaths, mode: plan.mode };
  }

  list() { return [...this.jobs.values()].reverse(); }
  get(id: string) { return this.jobs.get(id) || null; }

  pause(id: string) {
    const job = this.jobs.get(id);
    if (!job || !["queued", "running", "pausing"].includes(job.status)) throw new Error("Only queued or running jobs can be paused.");
    if (job.status === "queued") { job.status = "paused"; job.phase = "Paused before starting"; }
    else {
      const folder = join(this.root, ".jobs", id);
      mkdirSync(folder, { recursive: true });
      writeFileSync(join(folder, "pause.requested"), "pause");
      job.status = "pausing"; job.phase = "Pause requested; finishing the current path";
    }
    this.persist(true);
    return job;
  }

  resume(id: string) {
    const job = this.jobs.get(id);
    if (!job || !["paused", "error"].includes(job.status)) throw new Error("Only paused or failed jobs can be resumed.");
    const marker = join(this.root, ".jobs", id, "pause.requested");
    if (existsSync(marker)) unlinkSync(marker);
    job.status = "queued"; job.phase = "Queued to resume from checkpoint";
    delete job.error; delete job.finished;
    this.persist(true); queueMicrotask(() => this.pump());
    return job;
  }

  shutdown() {
    this.stopping = true;
    for (const job of this.jobs.values()) if (job.status === "running" || job.status === "pausing") {
      job.status = "paused"; job.phase = "Interrupted; resume from the last completed path";
    }
    this.persist(true);
    this.child?.kill("SIGTERM");
  }

  private pump() {
    if (this.running || this.stopping) return;
    const job = [...this.jobs.values()].find((j) => j.status === "queued");
    if (!job) return;
    this.running = true; job.status = "running"; job.phase = "Starting or resuming Python";
    job.started = new Date().toISOString(); this.persist(true);
    const venv = join(this.repo, ".venv", "bin", "python");
    const checkpoint = join(this.root, ".jobs", job.id);
    if (this.durable) mkdirSync(checkpoint, { recursive: true });
    let settled = false;
    const finish = (error?: string, paused = false) => {
      if (settled) return; settled = true;
      paused = paused || this.stopping;
      job.status = paused ? "paused" : error ? "error" : "done";
      job.phase = paused ? "Paused at the last completed path" : error ? "Failed; resume is available" : "Completed";
      if (error && !paused) job.error = error;
      if (!paused) job.finished = new Date().toISOString();
      if (!error && !paused) job.progress = 100;
      if (job.status === "done" && this.durable) {
        try { rmSync(checkpoint, { recursive: true, force: true }); } catch { /* Retain a cleanup failure for a later maintenance pass. */ }
      }
      this.child = undefined;
      this.running = false; this.persist(true); queueMicrotask(() => this.pump());
    };
    try {
      const combined = job.mode === "combined";
      const task = job.kind === "optimization" || job.kind === "replay";
      const args = task ? ["-m", "tqqq_book.optimizer", "--kind", job.kind!, "--id", job.id] : combined ? ["-m", "tqqq_book.combined", "--progress"] : commandArguments(job.config);
      if (job.sourceRuns) args.push("--saved-runs");
      const child = this.launch(existsSync(venv) ? venv : "python3", args, {
        cwd: this.repo, env: { ...process.env, PYTHONPATH: join(this.repo, "src"), RUNS_DIR: this.root, TQQQ_CHECKPOINT_DIR: checkpoint, PYTHONUNBUFFERED: "1" },
        stdio: [combined || task ? "pipe" : "ignore", "pipe", "pipe"],
      });
      this.child = child;
      let pending = "";
      const line = (text: string) => {
        if (text.startsWith("TQQQ_PROGRESS ")) {
          try {
            const event = JSON.parse(text.slice("TQQQ_PROGRESS ".length));
            if (Number.isFinite(event.percent)) job.progress = Math.max(job.progress, Math.min(99, event.percent));
            if (typeof event.phase === "string" && job.status !== "pausing") job.phase = event.phase;
            if (Number.isInteger(event.completed)) job.completedPaths = Math.max(job.completedPaths, event.completed);
            if (typeof event.run === "string" && /^[\w-]+$/.test(event.run)) job.run = event.run;
            if (typeof event.optimization === "string" && /^[\w-]+$/.test(event.optimization)) job.optimization = event.optimization;
            this.persist();
          } catch { /* Ignore malformed progress without losing the job. */ }
        } else job.log = (job.log + text + "\n").slice(-4000);
      };
      child.stdout?.on("data", (chunk: Buffer) => { pending += chunk.toString(); const lines = pending.split("\n"); pending = lines.pop() || ""; lines.forEach(line); });
      child.stderr?.on("data", (chunk: Buffer) => { job.log = (job.log + chunk.toString()).slice(-4000); });
      child.on("error", (error: Error) => finish(error.message));
      child.on("close", (code: number | null) => {
        if (pending) line(pending);
        if (code === 75) finish(undefined, true);
        else if (code !== 0) finish(job.log || `Simulation exited with code ${code}.`);
        else if (job.kind === "optimization" ? !existsSync(join(this.root, "optimizations", job.id, "result.json")) : !job.run || !existsSync(join(this.root, job.run, "summary.json"))) finish("Job finished without a saved result.");
        else finish();
      });
      if (combined || task) {
        child.stdin?.on("error", () => undefined);
        child.stdin?.end(JSON.stringify(task ? job.payload : job.sourceRuns ? { runs: job.sourceRuns } : { configurations: job.configurations }));
      }
    } catch (error) { finish(error instanceof Error ? error.message : String(error)); }
  }
}
