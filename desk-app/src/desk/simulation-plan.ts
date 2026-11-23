export interface ScalarParameter {
  key: string;
  label: string;
  flag: string;
  value: number;
  group: string;
  min?: number;
  max?: number;
  integer?: boolean;
}

export const SCALARS: ScalarParameter[] = [
  { key: "lots", label: "Lots", flag: "--lots", value: 100, group: "Run", min: 1, integer: true },
  { key: "horizon", label: "Horizon days", flag: "--horizon", value: 100, group: "Run", min: 1, integer: true },
  { key: "paths", label: "Paths", flag: "--paths", value: 100, group: "Run", min: 1, integer: true },
  { key: "seed", label: "Seed", flag: "--seed", value: 13, group: "Run", min: 0, integer: true },
  { key: "spot", label: "Spot", flag: "--spot", value: 59.22, group: "Run", min: 0.01 },
  { key: "iv", label: "Opening IV", flag: "--iv", value: 0.71, group: "Run", min: 0.001 },
  { key: "vol", label: "Realized volatility", flag: "--vol", value: 0.66, group: "Model", min: 0 },
  { key: "drift", label: "QQQ annual drift", flag: "--drift", value: 0.12, group: "Model" },
  { key: "rate", label: "Interest rate", flag: "--rate", value: 0.04, group: "Model" },
  { key: "initialDte", label: "Initial put DTE label", flag: "--initial-dte", value: 180, group: "Model", min: 1, integer: true },
  { key: "ivElasticity", label: "IV elasticity", flag: "--iv-elasticity", value: 0.86, group: "Model", min: 0 },
  { key: "slippage", label: "Slippage per share", flag: "--slippage", value: 0.05, group: "Model", min: 0 },
  { key: "lambdaJ", label: "Jumps per year", flag: "--lambda-j", value: 0.7, group: "Model", min: 0 },
  { key: "muJ", label: "Mean log jump", flag: "--mu-j", value: -0.07, group: "Model" },
  { key: "sigmaJ", label: "Jump volatility", flag: "--sigma-j", value: 0.05, group: "Model", min: 0 },
  { key: "harvestDrop", label: "Harvest: price drop", flag: "--harvest-drop", value: 0.08, group: "Rules", min: 0, max: 1 },
  { key: "harvestKeep", label: "Harvest: buyback fraction", flag: "--harvest-keep", value: 0.5, group: "Rules", min: 0, max: 1 },
  { key: "rollFraction", label: "Crash roll: fraction", flag: "--roll-fraction", value: 0.5, group: "Rules", min: 0, max: 1 },
  { key: "crashLine", label: "Crash roll: trigger price", flag: "--crash-line", value: 45, group: "Rules", min: 0 },
  { key: "crashDown", label: "Crash roll: strike reduction", flag: "--crash-down", value: 10, group: "Rules", min: 0 },
  { key: "rollUpGap", label: "Roll up: strike gap", flag: "--roll-up-gap", value: 20, group: "Rules", min: 0 },
  { key: "rollDownGap", label: "Roll down: strike gap", flag: "--roll-down-gap", value: 12, group: "Rules", min: 0 },
  { key: "rollDownFloor", label: "Roll down: minimum strike", flag: "--roll-down-floor", value: 40, group: "Rules", min: 1 },
  { key: "writeEvery", label: "Write calls: every N days", flag: "--write-every", value: 5, group: "Rules", min: 1, integer: true },
  { key: "writeFloor", label: "Write calls: price floor", flag: "--write-floor", value: 52, group: "Rules", min: 0 },
  { key: "minSpotFrac", label: "Write calls: minimum spot fraction", flag: "--min-spot-frac", value: 0.95, group: "Rules", min: 0 },
  { key: "reserve", label: "Write calls: reserved shares", flag: "--reserve", value: 1000, group: "Rules", min: 0, integer: true },
  { key: "reenterOtm", label: "Replace assigned: put OTM", flag: "--reenter-otm", value: 0.05, group: "Rules", min: 0, max: 0.99 },
  { key: "reenterTenor", label: "Replace assigned: tenor days", flag: "--reenter-tenor", value: 10, group: "Rules", min: 1, integer: true },
];

export type Configuration = Record<string, string>;
export interface RangeSpec { parameter: string; from: number; to: number; step: number }
export interface Parameter { key: string; label: string; integer?: boolean; min?: number; max?: number }
export const MAX_SIMULATIONS = 1000;

export const DEFAULTS: Configuration = {
  ...Object.fromEntries(SCALARS.map((p) => [p.key, String(p.value)])),
  strategy: "wheel", puts: "10:60:30,40:90:45,50:180:90",
  otm: "0.03,0.05,0.08,0.10", contracts: "12,12,12,10",
  putRatioGrid: "0:1,15:1.1,30:1.2,50:1.5", rules: "",
};

function numeric(value: unknown, label: string): number {
  if ((typeof value !== "number" && typeof value !== "string") || String(value).trim() === "") {
    throw new Error(`${label} needs a number.`);
  }
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`${label} needs a finite number.`);
  return n;
}

function check(value: unknown, p: Parameter): number {
  const n = numeric(value, p.label);
  if (p.integer && !Number.isSafeInteger(n)) throw new Error(`${p.label} needs a whole number.`);
  if (p.min !== undefined && n < p.min) throw new Error(`${p.label} must be at least ${p.min}.`);
  if (p.max !== undefined && n > p.max) throw new Error(`${p.label} must be at most ${p.max}.`);
  return n;
}

function matrix(text: string, width: number, label: string): number[][] {
  const rows = text.split(",").map((part) => part.split(":"));
  if (rows.some((row) => row.length !== width)) throw new Error(`${label} needs ${width} values per entry, separated by colons.`);
  return rows.map((row) => row.map((v) => numeric(v, label)));
}

export function parametersFor(cfg: Configuration): Parameter[] {
  const params: Parameter[] = SCALARS.map(({ key, label, min, max, integer }) => ({ key, label, min, max, integer }));
  for (let i = 0; i < 4; i++) {
    params.push({ key: `otm.${i}`, label: `Call OTM · ${i + 1}w`, min: 0 });
    params.push({ key: `contracts.${i}`, label: `Call contracts · ${i + 1}w`, min: 0, integer: true });
  }
  cfg.puts.split(",").forEach((_, i) => {
    params.push({ key: `puts.${i}.0`, label: `Put sleeve ${i + 1} · percent`, min: 0.001, max: 100 });
    params.push({ key: `puts.${i}.1`, label: `Put sleeve ${i + 1} · DTE`, min: 2, integer: true });
    params.push({ key: `puts.${i}.2`, label: `Put sleeve ${i + 1} · roll DTE`, min: 1, integer: true });
  });
  cfg.putRatioGrid.split(",").forEach((_, i) => {
    params.push({ key: `putRatioGrid.${i}.0`, label: `Put ratio step ${i + 1} · drop percent`, min: 0 });
    params.push({ key: `putRatioGrid.${i}.1`, label: `Put ratio step ${i + 1} · ratio`, min: 0.001 });
  });
  return params;
}

function parameterValue(cfg: Configuration, key: string): string {
  const [name, row, column] = key.split(".");
  if (row === undefined) return cfg[name];
  const entry = cfg[name].split(",")[Number(row)];
  return column === undefined ? entry : entry.split(":")[Number(column)];
}

function setParameter(cfg: Configuration, key: string, value: number): Configuration {
  const [name, row, column] = key.split(".");
  if (row === undefined) return { ...cfg, [name]: String(value) };
  const rows = cfg[name].split(",");
  if (column === undefined) rows[Number(row)] = String(value);
  else {
    const parts = rows[Number(row)].split(":");
    parts[Number(column)] = String(value);
    rows[Number(row)] = parts.join(":");
  }
  return { ...cfg, [name]: rows.join(",") };
}

function validate(cfg: Configuration) {
  if (cfg.otm.split(",").length !== 4 || cfg.contracts.split(",").length !== 4) {
    throw new Error("Call OTM and contracts each need four values: 1w, 2w, 3w, 4w.");
  }
  const sleeves = matrix(cfg.puts, 3, "Put sleeves");
  matrix(cfg.putRatioGrid, 2, "Put ratio grid");
  for (const p of parametersFor(cfg)) check(parameterValue(cfg, p.key), p);
  const weightSum = sleeves.reduce((sum, row) => sum + row[0] / 100, 0);
  if (Math.abs(weightSum - 1) > 0.02) throw new Error("Put sleeve percents must sum to 100 for every combination.");
  if (sleeves.some(([, dte, roll]) => roll >= dte)) throw new Error("Each put sleeve's roll DTE must be shorter than its DTE.");
  if (!["wheel", "hold"].includes(cfg.strategy)) throw new Error("Strategy must be wheel or hold.");
  if (!/^[a-z_,]*$/.test(cfg.rules)) throw new Error("Invalid rule names.");
}

export function planSimulations(input: Record<string, unknown>) {
  const mode = input.sweepMode ?? "separate";
  if (mode !== "separate" && mode !== "combined") throw new Error("Choose separate simulations or one combined simulation.");
  const base = { ...DEFAULTS };
  for (const key of Object.keys(base)) {
    if (input[key] !== undefined) base[key] = String(input[key]).trim();
  }
  validate(base);
  for (const p of SCALARS) base[p.key] = String(Number(base[p.key]));
  base.contracts = base.contracts.split(",").map((v) => String(Number(v))).join(",");
  base.puts = base.puts.split(",").map((row) => row.split(":").map((v) => String(Number(v))).join(":")).join(",");
  const parameters = parametersFor(base);
  const rawRanges = input.ranges ?? [];
  if (!Array.isArray(rawRanges)) throw new Error("Ranges must be a list.");
  const seen = new Set<string>();
  let total = 1;
  const ranges = rawRanges.map((raw: RangeSpec) => {
    const parameter = parameters.find((p) => p.key === raw?.parameter);
    if (!parameter) throw new Error("Choose a valid parameter for each range.");
    if (seen.has(parameter.key)) throw new Error(`${parameter.label} has more than one range.`);
    seen.add(parameter.key);
    const from = check(raw.from, parameter), to = check(raw.to, parameter);
    const step = numeric(raw.step, `${parameter.label} increment`);
    if (step <= 0 || (parameter.integer && !Number.isSafeInteger(step))) throw new Error(`${parameter.label} needs a positive ${parameter.integer ? "whole number " : ""}increment.`);
    if (to < from) throw new Error(`${parameter.label}: end must be at least start.`);
    const count = Math.floor((to - from) / step + 1e-9) + 1;
    total *= count;
    if (!Number.isSafeInteger(count) || total > MAX_SIMULATIONS) throw new Error(`A batch can contain at most ${MAX_SIMULATIONS.toLocaleString("en-US")} simulations. Narrow the ranges or increase the increments.`);
    const values = Array.from({ length: count }, (_, i) => Number((from + i * step).toPrecision(12)));
    if (new Set(values).size !== count) throw new Error(`${parameter.label}: increment is too small.`);
    return { parameter: parameter.key, label: parameter.label, from, to, step, values, count };
  });
  let configurations = [base];
  for (const range of ranges) {
    configurations = configurations.flatMap((cfg) => range.values.map((v) => setParameter(cfg, range.parameter, v)));
  }
  for (const cfg of configurations) validate(cfg);
  return {
    configurations, ranges, total, mode,
    simulationCount: mode === "combined" ? 1 : total,
    totalPaths: configurations.reduce((n, cfg) => n + Number(cfg.paths), 0),
    parameters,
  };
}

export function planCombinedConfigurations(configurations: Record<string, unknown>[]) {
  if (!configurations.length || configurations.length > MAX_SIMULATIONS) throw new Error("A combined simulation needs between 1 and 1,000 configurations.");
  const plans = configurations.map((config) => planSimulations({ ...config, ranges: [], sweepMode: "separate" }));
  return {
    ...plans[0], mode: "combined", simulationCount: 1,
    total: plans.length, configurations: plans.map((plan) => plan.configurations[0]), ranges: [],
    totalPaths: plans.reduce((sum, plan) => sum + plan.totalPaths, 0),
  };
}

export function commandArguments(cfg: Configuration): string[] {
  return [
    "-m", "tqqq_book", "--progress",
    ...SCALARS.flatMap((p) => [p.flag, cfg[p.key]]),
    "--strategy", cfg.strategy, "--puts", cfg.puts, "--otm", cfg.otm,
    "--contracts", cfg.contracts, "--put-ratio", cfg.putRatioGrid,
    ...(cfg.rules ? ["--rules", cfg.rules] : []),
  ];
}

export function rangeValues(cfg: Configuration, keys: string[]) {
  return Object.fromEntries(keys.map((key) => [key, parameterValue(cfg, key)]));
}
