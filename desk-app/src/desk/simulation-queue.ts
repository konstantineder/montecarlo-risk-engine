import { spawn } from "child_process";
import { randomUUID } from "crypto";
import { existsSync } from "fs";
import { join } from "path";
import { commandArguments, Configuration, planSimulations, rangeValues } from "./simulation-plan";

export interface SimulationJob {
  id: string;
  batch: string;
  index: number;
  batchSize: number;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  phase: string;
  completedPaths: number;
  totalPaths: number;
  created: string;
  started?: string;
  finished?: string;
  config: Configuration;
  variations: Record<string, string>;
  run?: string;
  error?: string;
  log: string;
}

export class SimulationQueue {
  private readonly jobs = new Map<string, SimulationJob>();
  private running = false;

  constructor(private readonly repo: string, private readonly root: string, private readonly launch: typeof spawn = spawn) {}

  enqueue(input: Record<string, unknown>) {
    const plan = planSimulations(input);
    const batch = randomUUID();
    const ids: string[] = [];
    plan.configurations.forEach((config, index) => {
      const id = randomUUID();
      ids.push(id);
      this.jobs.set(id, {
        id, batch, index: index + 1, batchSize: plan.total,
        status: "queued", progress: 0, phase: "Queued",
        completedPaths: 0, totalPaths: Number(config.paths),
        created: new Date().toISOString(), config,
        variations: rangeValues(config, plan.ranges.map((r) => r.parameter)), log: "",
      });
    });
    queueMicrotask(() => this.pump());
    return { job: ids[0], batch, total: plan.total };
  }

  list() { return [...this.jobs.values()].reverse(); }
  get(id: string) { return this.jobs.get(id) || null; }

  private pump() {
    if (this.running) return;
    const job = [...this.jobs.values()].find((j) => j.status === "queued");
    if (!job) return;
    this.running = true;
    job.status = "running";
    job.phase = "Starting Python";
    job.started = new Date().toISOString();
    const venv = join(this.repo, ".venv", "bin", "python");
    let settled = false;
    const finish = (error?: string) => {
      if (settled) return;
      settled = true;
      job.finished = new Date().toISOString();
      job.status = error ? "error" : "done";
      job.phase = error ? "Failed" : "Completed";
      if (error) job.error = error;
      else job.progress = 100;
      this.running = false;
      queueMicrotask(() => this.pump());
    };
    try {
      const child = this.launch(existsSync(venv) ? venv : "python3", commandArguments(job.config), {
        cwd: this.repo,
        env: { ...process.env, PYTHONPATH: join(this.repo, "src"), RUNS_DIR: this.root, PYTHONUNBUFFERED: "1" },
        stdio: ["ignore", "pipe", "pipe"],
      });
      let pending = "";
      const line = (text: string) => {
        if (text.startsWith("TQQQ_PROGRESS ")) {
          try {
            const event = JSON.parse(text.slice("TQQQ_PROGRESS ".length));
            if (Number.isFinite(event.percent)) job.progress = Math.max(job.progress, Math.min(99, event.percent));
            if (typeof event.phase === "string") job.phase = event.phase;
            if (Number.isInteger(event.completed)) job.completedPaths = event.completed;
            if (typeof event.run === "string" && /^[\w-]+$/.test(event.run)) job.run = event.run;
          } catch { /* A partial or malformed event must not stop the queue. */ }
        } else job.log = (job.log + text + "\n").slice(-4000);
      };
      child.stdout?.on("data", (chunk: Buffer) => {
        pending += chunk.toString();
        const lines = pending.split("\n");
        pending = lines.pop() || "";
        lines.forEach(line);
      });
      child.stderr?.on("data", (chunk: Buffer) => { job.log = (job.log + chunk.toString()).slice(-4000); });
      child.on("error", (error: Error) => finish(error.message));
      child.on("close", (code: number | null) => {
        if (pending) line(pending);
        if (code !== 0) finish(job.log || `Simulation exited with code ${code}.`);
        else if (!job.run || !existsSync(join(this.root, job.run, "summary.json"))) finish("Simulation finished without a saved result.");
        else finish();
      });
    } catch (error) {
      finish(error instanceof Error ? error.message : String(error));
    }
  }
}
