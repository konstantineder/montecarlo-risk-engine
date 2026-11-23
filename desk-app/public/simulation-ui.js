const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let schemaRequest;
let schemaData;
let stopJobsPolling;

function simulationSchema() {
  if (!schemaRequest) schemaRequest = get("/api/simulation/config").then((data) => { schemaData = data; return data; }).catch((error) => { schemaRequest = null; throw error; });
  return schemaRequest;
}

async function postJson(url, body) {
  const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const result = await response.json();
  if (!response.ok) throw new Error(Array.isArray(result.message) ? result.message.join(" ") : result.message || "The request failed.");
  return result;
}

function simulationTabs(selected) {
  const nav = document.createElement("nav");
  nav.className = "tabs";
  nav.setAttribute("aria-label", "Simulations");
  [["new", "#/", "New simulation"], ["active", "#/active", "Active runs"], ["saved", "#/saved", "Saved runs"]].forEach(([key, href, label]) => {
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label;
    if (key === selected) { a.className = "on"; a.setAttribute("aria-current", "page"); }
    nav.append(a);
  });
  return nav;
}

function rangeParameters(schema, values) {
  const parameters = schema.scalars.map((p) => ({ ...p }));
  for (let i = 0; i < 4; i++) {
    parameters.push({ key: `otm.${i}`, label: `Call OTM · ${i + 1}w` });
    parameters.push({ key: `contracts.${i}`, label: `Call contracts · ${i + 1}w`, integer: true });
  }
  String(values.puts).split(",").forEach((_, i) => {
    ["percent", "DTE", "roll DTE"].forEach((label, j) => parameters.push({ key: `puts.${i}.${j}`, label: `Put sleeve ${i + 1} · ${label}`, integer: j > 0 }));
  });
  String(values.putRatioGrid).split(",").forEach((_, i) => {
    ["drop percent", "ratio"].forEach((label, j) => parameters.push({ key: `putRatioGrid.${i}.${j}`, label: `Put ratio step ${i + 1} · ${label}` }));
  });
  return parameters;
}

async function simulationForm(saveOnly = false) {
  const schema = await simulationSchema();
  const cfg = { ...schema.defaults, ...rulePayload(), ...savedConfig() };
  const wrap = document.createElement("div");
  const form = document.createElement("form");
  form.className = "form";
  const field = (name, label, wide = false) => `<label class="${wide ? "wide" : ""}">${escapeHtml(label)}<input name="${name}" value="${escapeHtml(cfg[name])}"/></label>`;
  form.innerHTML = `
    ${field("lots", "Lots")}${field("horizon", "Horizon days")}${field("paths", "Paths per configuration")}${field("seed", "Seed")}
    ${field("spot", "Spot")}${field("iv", "Opening IV")}
    <label>Strategy<select name="strategy"><option value="wheel">wheel</option><option value="hold">hold</option></select></label>
    ${field("contracts", "Contracts 1w,2w,3w,4w")}
    ${field("puts", "Put sleeves percent:dte:roll", true)}${field("otm", "Call OTM 1w,2w,3w,4w", true)}
    <details class="advanced full"><summary>Advanced configuration &amp; parameter ranges</summary>
      <p class="note">Set model and rule values here. Decimals are fractions: 0.2 means 20%. Rule switches are on the Rules tab. The put sleeves determine the actual hedge DTEs.</p>
      <h3>Model settings</h3><div class="settings-grid">${schema.scalars.filter((p) => p.group === "Model").map((p) => field(p.key, p.label)).join("")}</div>
      <h3>Rule settings</h3><div class="settings-grid">${schema.scalars.filter((p) => p.group === "Rules").map((p) => field(p.key, p.label)).join("")}${field("putRatioGrid", "Put ratio grid: drop percent:ratio", true)}</div>
      <label class="check"><input type="checkbox" id="enable-ranges"/>Run a parameter sweep</label>
      <div class="range-editor" hidden>
        <label class="sweep-output">Sweep output<select name="sweepMode"><option value="separate">Separate simulations — one per configuration</option><option value="combined">One combined simulation — all configurations and paths</option></select></label>
        <p class="note">Choose a start, end and increment for any numeric parameter. Every combination contributes its configured number of paths. Choose separate results or one combined result above. For example, 0.2 to 0.5 in steps of 0.1 gives 4 values; another 3-value range gives 12 configurations. The end is included when an increment reaches it.</p>
        <div class="range-rows"></div><button type="button" class="ghost add-range">Add parameter range</button>
        <p class="note">Up to ${schema.maxSimulations.toLocaleString()} configurations per sweep. Combined mode saves one result and retains the configuration on every path.</p>
      </div>
    </details>
    <div class="launch-summary full" role="status"></div>
    <button class="run" type="submit" disabled>${saveOnly ? "Save configuration" : "Run simulation"}</button>`;
  form.elements.strategy.value = cfg.strategy;
  form.elements.sweepMode.value = cfg.sweepMode || "separate";
  const advanced = form.querySelector("details");
  const enabled = form.querySelector("#enable-ranges");
  enabled.checked = cfg.sweepEnabled === true;
  advanced.open = enabled.checked;
  const ranges = Array.isArray(cfg.ranges) ? cfg.ranges.map((r) => ({ ...r })) : [];
  const editor = form.querySelector(".range-editor");
  const rows = form.querySelector(".range-rows");
  const preview = form.querySelector(".launch-summary");
  const launch = form.querySelector("button.run");
  const status = document.createElement("p");
  status.className = "status";
  status.setAttribute("role", "status");
  const values = () => Object.fromEntries(new FormData(form).entries());
  const payload = () => ({ ...rulePayload(), ...values(), ranges: enabled.checked ? ranges : [], sweepMode: enabled.checked ? form.elements.sweepMode.value : "separate" });
  let debounce, request = 0;
  const updatePreview = async (version) => {
    try {
      const plan = await postJson("/api/simulation/preview", payload());
      if (version !== request || !form.isConnected) return;
      const combined = plan.mode === "combined";
      const count = combined ? `1 combined simulation · ${plan.total.toLocaleString()} configurations` : `${plan.total.toLocaleString()} simulation${plan.total === 1 ? "" : "s"}`;
      const factors = plan.ranges.map((r) => `${r.count} values for ${r.label}`).join(" × ");
      preview.textContent = `${count} · ${plan.totalPaths.toLocaleString()} paths in total${factors ? ` · ${factors}` : ""}`;
      preview.classList.remove("invalid");
      launch.disabled = false;
      launch.textContent = saveOnly ? "Save configuration" : combined ? "Run combined simulation" : `Run ${plan.total === 1 ? "simulation" : `${plan.total.toLocaleString()} simulations`}`;
    } catch (error) {
      if (version !== request || !form.isConnected) return;
      preview.textContent = error.message;
      preview.classList.add("invalid");
      launch.disabled = true;
    }
  };
  const schedulePreview = () => {
    clearTimeout(debounce);
    const version = ++request;
    launch.disabled = true;
    preview.textContent = "Calculating combinations…";
    debounce = setTimeout(() => { if (form.isConnected) updatePreview(version); }, 200);
  };
  const drawRanges = () => {
    editor.hidden = !enabled.checked;
    const parameters = rangeParameters(schema, values());
    rows.replaceChildren();
    ranges.forEach((range, index) => {
      const row = document.createElement("div");
      row.className = "range-row";
      row.innerHTML = `<label>Parameter<select aria-label="Range parameter">${parameters.map((p) => `<option value="${p.key}">${escapeHtml(p.label)}</option>`).join("")}</select></label>
        ${[["from", "Start"], ["to", "End"], ["step", "Increment"]].map(([key, label]) => `<label>${label}<input type="number" step="any" data-range="${key}" value="${escapeHtml(range[key])}"/></label>`).join("")}
        <button type="button" class="remove-range" aria-label="Remove range ${index + 1}">Remove</button>`;
      row.querySelector("select").value = range.parameter;
      row.querySelector("select").addEventListener("change", (event) => { range.parameter = event.target.value; schedulePreview(); });
      row.querySelectorAll("input").forEach((input) => input.addEventListener("input", () => { range[input.dataset.range] = input.value; }));
      row.querySelector("button").addEventListener("click", () => { ranges.splice(index, 1); drawRanges(); schedulePreview(); });
      rows.append(row);
    });
  };
  enabled.addEventListener("change", () => { drawRanges(); schedulePreview(); });
  form.querySelector(".add-range").addEventListener("click", () => {
    const parameters = rangeParameters(schema, values());
    const parameter = ["otm.0", "otm.1", ...parameters.map((p) => p.key)].find((key) => !ranges.some((r) => r.parameter === key));
    if (!parameter) return;
    const [name, index, column] = parameter.split(".");
    let value = values()[name];
    if (index !== undefined) value = value.split(",")[Number(index)];
    if (column !== undefined) value = value.split(":")[Number(column)];
    ranges.push({ parameter, from: value, to: value, step: parameters.find((p) => p.key === parameter)?.integer ? 1 : 0.01 });
    drawRanges(); schedulePreview();
  });
  form.addEventListener("input", schedulePreview);
  form.querySelectorAll('[name="puts"], [name="putRatioGrid"]').forEach((input) => input.addEventListener("change", drawRanges));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    launch.disabled = true;
    status.textContent = saveOnly ? "Saving…" : "Queuing simulations…";
    try {
      const body = payload();
      const result = saveOnly ? await postJson("/api/simulation/preview", body) : await postJson("/api/simulate", body);
      saveConfig({ ...values(), sweepEnabled: enabled.checked, ranges });
      status.textContent = saveOnly ? "Saved. The simulation form will use this configuration and these ranges." : `Queued ${result.mode === "combined" ? `1 combined simulation with ${result.configurationCount} configurations and ${result.totalPaths.toLocaleString()} paths` : `${result.total} simulation${result.total === 1 ? "" : "s"}`}. Follow the progress below or in Active runs.`;
      if (!saveOnly) wrap.dispatchEvent(new CustomEvent("simulation-started", { bubbles: true }));
    } catch (error) { status.textContent = error.message; }
    finally { schedulePreview(); }
  });
  drawRanges();
  wrap.append(form, status);
  setTimeout(schedulePreview, 0);
  return wrap;
}

function progressBar(percent, label) {
  const bar = document.createElement("progress");
  bar.max = 100;
  bar.value = percent;
  bar.setAttribute("aria-label", label);
  return bar;
}

async function watchJobs(host, includeCompleted = false) {
  let stopped = false, timer, inFlight = false;
  const seen = new Set();
  stopJobsPolling = () => { stopped = true; clearTimeout(timer); };
  const refresh = async () => {
    if (stopped || inFlight || !host.isConnected) return;
    inFlight = true;
    clearTimeout(timer);
    try {
      const jobs = await get("/api/jobs");
      if (stopped || !host.isConnected) return;
      const expanded = new Set([...host.querySelectorAll("details[open]")].map((d) => d.dataset.job));
      const groups = new Map();
      for (const job of jobs) {
        if (job.status === "queued" || job.status === "running") seen.add(job.batch);
        if (!groups.has(job.batch)) groups.set(job.batch, []);
        groups.get(job.batch).push(job);
      }
      host.replaceChildren();
      for (const [batch, group] of groups) {
        if (!includeCompleted && !seen.has(batch)) continue;
        group.sort((a, b) => a.index - b.index);
        const section = document.createElement("section");
        section.className = "batch";
        const done = group.filter((j) => j.status === "done").length;
        const failed = group.filter((j) => j.status === "error").length;
        const percent = group.reduce((sum, j) => sum + j.progress, 0) / group.length;
        const h = document.createElement("h2");
        h.textContent = group.length === 1 ? "Simulation progress" : `Parameter sweep · ${group.length} simulations`;
        section.append(h);
        if (group.length > 1) {
          const summary = document.createElement("p");
          summary.className = "note";
          summary.textContent = `${percent.toFixed(1)}% · ${done} of ${group.length} completed${failed ? ` · ${failed} failed` : ""}`;
          section.append(summary, progressBar(percent, "Batch progress"));
        }
        const cards = document.createElement("div");
        cards.className = "cards job-cards";
        for (const job of group) {
          const card = document.createElement("article");
          card.className = `sim job ${job.status}`;
          card.dataset.job = job.id;
          const title = document.createElement("h3");
          title.textContent = job.mode === "combined" ? `Combined simulation · ${job.configurationCount} configurations` : group.length === 1 ? `${job.config.strategy} simulation` : `Simulation ${job.index} of ${job.batchSize}`;
          const progress = document.createElement("strong");
          progress.textContent = job.status === "queued" ? "Queued · 0%" : `${job.progress.toFixed(1)}%${job.status === "done" ? " · Completed" : job.status === "error" ? " · Failed" : ""}`;
          const phase = document.createElement("p");
          phase.className = "meta";
          phase.textContent = `${job.phase} · ${job.completedPaths}/${job.totalPaths} paths`;
          const config = document.createElement("p");
          config.className = "meta";
          config.textContent = job.mode === "combined" ? `${job.configurationCount} configurations · ${job.totalPaths} total paths · settings below` : `${job.config.lots} lots · ${job.config.horizon} days · seed ${job.config.seed}`;
          card.append(title, progress, progressBar(job.progress, `Simulation ${job.index} progress`), phase, config);
          if (Object.keys(job.variations).length) {
            const choices = document.createElement("p");
            choices.className = "variation";
            const labels = rangeParameters(schemaData || { scalars: [] }, job.config);
            choices.textContent = Object.entries(job.variations).map(([key, value]) => `${labels.find((p) => p.key === key)?.label || key}: ${value}`).join(" · ");
            card.append(choices);
          }
          if (job.status === "done" && job.run) {
            const link = document.createElement("a");
            link.className = "ghost";
            link.href = `#/run/${job.run}`;
            link.textContent = "Open results";
            card.append(link);
          }
          if (job.error) {
            const error = document.createElement("p");
            error.className = "job-error";
            error.textContent = job.error;
            card.append(error);
          }
          const details = document.createElement("details");
          details.dataset.job = job.id;
          details.innerHTML = "<summary>Configuration</summary>";
          details.open = expanded.has(job.id);
          const fill = () => {
            if (details.children.length > 1 || !details.open) return;
            const list = document.createElement("dl");
            (job.configurations || [job.config]).forEach((configuration, index) => {
              if (job.mode === "combined") {
                const heading = document.createElement("h4");
                heading.textContent = `Configuration ${index + 1} · ${configuration.paths} paths`;
                list.append(heading);
              }
              Object.entries(configuration).forEach(([key, value]) => {
              const term = document.createElement("dt"), definition = document.createElement("dd");
              const names = { puts: "Put sleeves: percent:DTE:roll", otm: "Call OTM: 1w, 2w, 3w, 4w", contracts: "Call contracts at 100 lots", putRatioGrid: "Put ratio grid", rules: "Enabled rules", strategy: "Strategy" };
              term.textContent = schemaData?.scalars.find((p) => p.key === key)?.label || names[key] || key;
              definition.textContent = key === "rules" && !value ? "All rules" : value;
                list.append(term, definition);
              });
            });
            details.append(list);
          };
          details.addEventListener("toggle", fill); fill();
          card.append(details);
          cards.append(card);
        }
        section.append(cards);
        host.append(section);
      }
      if (!host.children.length && includeCompleted) host.innerHTML = '<p class="empty">No active simulations. Start one from the New simulation tab.</p>';
    } catch (error) {
      if (!stopped) {
        let note = host.querySelector(".poll-error");
        if (!note) { note = document.createElement("p"); note.className = "status poll-error"; host.prepend(note); }
        note.textContent = `Could not refresh progress: ${error.message}. Retrying…`;
      }
    } finally {
      inFlight = false;
      if (!stopped && host.isConnected) timer = setTimeout(refresh, 1500);
    }
  };
  host.refresh = refresh;
  await refresh();
}

async function showActive() {
  const view = state.view;
  crumb.textContent = "Active runs";
  app.innerHTML = "<h1>Active runs</h1><p class='lede'>Track queued and running simulations, inspect their configurations, and open completed results. Runs continue when you leave this tab.</p>";
  app.append(simulationTabs("active"));
  const host = document.createElement("div");
  app.append(host);
  await simulationSchema();
  if (view !== state.view) return;
  await watchJobs(host, true);
}

function combinedConfigurations(run) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = "Configurations in this simulation";
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = run.note;
  const groups = run.configurations || [];
  section.append(heading, note, table(
    ["Configuration", "Paths", "Lots", "Horizon", "Call OTM 1w–4w", "Median P&L", "Loss"],
    groups.map((g) => [String(g.id), String(g.config.paths), String(g.config.lots), `${g.config.horizon_days}d`, g.config.tenors.map((t) => t.otm).join(", "), money(g.monte_carlo.median), pct(g.monte_carlo.prob_loss)])
  ));
  groups.forEach((group) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Configuration ${group.id} · opening book, rules and settings`;
    const pre = document.createElement("pre");
    pre.className = "configuration-data";
    pre.textContent = JSON.stringify(group.config, null, 2);
    const legs = [...group.book.puts, ...group.book.calls];
    details.append(summary, facts([
      ["Deployed", money(group.book.deployed)], ["Put premium", money(group.book.put_premium)],
      ["Call premium", money(group.book.call_premium)], ["Floor", money(group.book.floor)],
    ]), table(["Side", "Qty", "Strike", "DTE", "Premium"], legs.map((leg) => [leg.side, String(leg.qty), px(leg.strike), String(leg.dte), px(leg.premium)])), ruleSnapshot(group.rules), pre);
    section.append(details);
  });
  return section;
}
