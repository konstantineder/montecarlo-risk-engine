const app = document.getElementById("app");
const crumb = document.getElementById("crumb");

const money = (n) => {
  const v = Number(n) || 0;
  const sign = v < 0 ? "-" : "";
  return sign + "$" + Math.abs(Math.round(v)).toLocaleString("en-US");
};
const px = (n) => Number(n).toFixed(2);
const pct = (n) => (Number(n) * 100).toFixed(1) + "%";
const when = (stamp) => {
  if (!stamp) return "";
  const m = String(stamp).match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} UTC` : stamp;
};

const cache = new Map();
async function get(url) {
  if (cache.has(url)) return cache.get(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  if (url.startsWith("/api/runs/") && !/\/paths\/\d+$/.test(url)) {
    if (cache.size >= 24) cache.delete(cache.keys().next().value);
    cache.set(url, data);
  }
  return data;
}

function facts(pairs) {
  const wrap = document.createElement("div");
  wrap.className = "facts";
  pairs.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.className = "fact";
    div.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    wrap.append(div);
  });
  return wrap;
}

function table(headers, rows, onClick) {
  const grid = document.createElement("div");
  grid.className = "grid";
  const el = document.createElement("table");
  const head = document.createElement("thead");
  const hr = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    hr.append(th);
  });
  head.append(hr);
  const body = document.createElement("tbody");
  rows.forEach((cells, i) => {
    const tr = document.createElement("tr");
    cells.forEach((text, j) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (headers[j] === "Trades") td.className = "trades";
      if (["P&L", "MTM", "Drawdown", "Mean P&L", "Median P&L"].includes(headers[j])) {
        td.classList.add("table-value");
        if (/^[-−]\$/.test(text)) td.classList.add("negative");
        else if (/^\$/.test(text) && text !== "$0") td.classList.add("positive");
      }
      tr.append(td);
    });
    if (onClick) {
      tr.tabIndex = 0;
      tr.addEventListener("click", () => onClick(i, tr));
      tr.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        onClick(i, tr);
      });
    }
    body.append(tr);
  });
  el.append(head, body);
  grid.append(el);
  return grid;
}

function putStrikes(day) {
  const longs = (day.position || []).filter((row) => row.name === "put" && row.side === "long");
  const qty = longs.reduce((n, row) => n + (Number(row.qty) || 0), 0);
  const floor = longs.length ? Math.min(...longs.map((row) => Number(row.strike) || 0)) : 0;
  const avg = qty ? longs.reduce((n, row) => n + (Number(row.strike) || 0) * (Number(row.qty) || 0), 0) / qty : 0;
  return { floor, avg };
}

function bookCounts(day) {
  let shares = 0, puts = 0, shortCalls = 0, shortPuts = 0;
  (day.position || []).forEach((row) => {
    const qty = Number(row.qty) || 0;
    if (row.name === "share") shares = qty;
    else if (row.name === "put" && row.side === "long") puts += qty;
    else if (row.name === "call" && row.side === "short") shortCalls += qty;
    else if (row.name === "put" && row.side === "short") shortPuts += qty;
  });
  return { shares: shares / 100, puts, shortCalls, shortPuts };
}

const state = { run: null, path: null, view: 0 };
let pathSelection;
let pathOrderCache;

async function route() {
  state.view++;
  disposeCharts();
  if (stopJobsPolling) { stopJobsPolling(); stopJobsPolling = null; }
  const [hash, query] = (location.hash.replace(/^#/, "") || "/").split("?");
  const parts = hash.split("/").filter(Boolean);
  const params = new URLSearchParams(query);
  crumb.textContent = "";
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("on", link.getAttribute("href") === `#/${parts[0] || ""}`.replace("#//", "#/"));
  });
  if (parts[0] === "rules") return showRules();
  if (parts[0] === "config") return showConfig();
  if (parts[0] === "active") return showActive();
  if (parts[0] === "saved") return showHome(true);
  if (parts[0] === "optimize") return showOptimizerForm(parts[1], Number(parts[2]));
  if (parts[0] === "optimizer") return showOptimizerResult(parts[1]);
  if (parts[0] === "run" && parts[2] === "path") return showPath(parts[1], Number(parts[3]), params);
  if (parts[0] === "run") return showRun(parts[1], params);
  return showHome();
}

const RULE_FIELDS = {
  harvest: [["drop", "harvestDrop", "Drop"], ["keep", "harvestKeep", "Buy back at"]],
  crash_roll: [["fraction", "rollFraction", "Fraction"], ["line", "crashLine", "Crash line"], ["down", "crashDown", "Roll down by"]],
  roll_up: [["gap", "rollUpGap", "Gap"]],
  roll_down: [["gap", "rollDownGap", "Gap"], ["floor", "rollDownFloor", "Min strike"]],
  put_ratio: [["grid", "putRatioGrid", "Grid"]],
  write_calls: [["every", "writeEvery", "Every"], ["floor", "writeFloor", "Floor"], ["min_spot_frac", "minSpotFrac", "Min spot"], ["reserve_shares", "reserve", "Reserve"]],
  replace_assigned: [["otm", "reenterOtm", "OTM"], ["tenor", "reenterTenor", "Tenor"]],
};

function chosenRules() {
  const saved = localStorage.getItem("desk-rules");
  if (!saved) return null;
  const parsed = JSON.parse(saved);
  return Array.isArray(parsed) ? null : parsed;
}

function saveRules(state) {
  localStorage.setItem("desk-rules", JSON.stringify(state));
  const cfg = savedConfig();
  Object.values(state).forEach((row) => Object.assign(cfg, row.values || {}));
  saveConfig(cfg);
}

function savedConfig() {
  const saved = localStorage.getItem("desk-config");
  return saved ? JSON.parse(saved) : {};
}

function saveConfig(values) {
  localStorage.setItem("desk-config", JSON.stringify(values));
}

function rulePayload() {
  const state = chosenRules() || {};
  const body = {};
  const cfg = savedConfig();
  Object.values(RULE_FIELDS).flat().forEach(([, key]) => { if (cfg[key] !== undefined) body[key] = cfg[key]; });
  const on = [];
  Object.entries(state).forEach(([name, row]) => {
    if (row.on !== false) on.push(name);
    Object.entries(row.values || {}).forEach(([key, value]) => { body[key] = cfg[key] ?? value; });
  });
  if (chosenRules()) body.rules = on.join(",") || "none";
  return body;
}

async function showRules() {
  const view = state.view;
  crumb.textContent = "Rules";
  app.innerHTML = "<h1>Rules</h1><p class='lede'>A rule is on or off, and its numbers are editable. Off means it is not called. These values are used for the next simulation and for a re-read.</p>";
  const rules = await get("/api/rules");
  if (view !== state.view) return;
  const saved = chosenRules() || {};
  const ruleState = {};
  rules.forEach((rule) => {
    const prior = saved[rule.name] || {};
    const values = {};
    (RULE_FIELDS[rule.name] || []).forEach(([key, bodyKey]) => {
      values[bodyKey] = savedConfig()[bodyKey] ?? prior.values?.[bodyKey] ?? rule[key];
    });
    ruleState[rule.name] = { on: prior.on !== false, values };
  });
  const list = document.createElement("div");
  const draw = () => {
    list.replaceChildren();
    rules.forEach((rule) => {
      const row = ruleState[rule.name];
      const card = document.createElement("article");
      card.className = `rule ${row.on ? "on" : "off"}`;
      const fields = (RULE_FIELDS[rule.name] || []).map(([key, bodyKey, label]) =>
        `<label>${label}<input data-key="${bodyKey}" value="${row.values[bodyKey] ?? ""}"/></label>`
      ).join("");
      card.innerHTML = `
        <button class="switch">${row.on ? "ON" : "OFF"}</button>
        <div>
          <h3>${rule.name}</h3>
          <p>${rule.says}</p>
          <div class="knobs">${fields}</div>
        </div>`;
      card.querySelector("button").addEventListener("click", () => {
        row.on = !row.on;
        saveRules(ruleState);
        draw();
      });
      card.querySelectorAll("input").forEach((input) => {
        input.addEventListener("change", () => {
          row.values[input.dataset.key] = input.value;
          saveRules(ruleState);
        });
      });
      list.append(card);
    });
  };
  draw();
  app.append(list);
}

async function showConfig() {
  crumb.textContent = "Configuration";
  app.innerHTML = "<h1>Configuration</h1><p class='lede'>Save the fixed values and advanced ranges used by the simulation form.</p>";
  const view = state.view;
  const form = await simulationForm(true);
  if (view === state.view) app.append(form);
}

function exportBar(id) {
  const bar = document.createElement("div");
  bar.className = "exports";
  bar.innerHTML = `
    <a class="ghost" href="/api/runs/${id}/export.json">JSON full</a>
    <a class="ghost" href="/api/runs/${id}/export.mtm.json">JSON trades + MTM</a>
    <a class="ghost" href="/api/runs/${id}/export.book.json">JSON trades + positions</a>
    <a class="ghost" href="/api/runs/${id}/export.csv">CSV</a>
    <span class="note">Full keeps the before and after of every rule. Trades + MTM is the daily mark and the trades. Trades + positions adds the end-of-day book. CSV is the zip.</span>`;
  return bar;
}

function ruleSnapshot(rules) {
  const wrap = document.createElement("div");
  (rules || []).forEach((rule) => {
    const card = document.createElement("article");
    card.className = `rule ${rule.on === false ? "off" : "on"}`;
    const acted = rule.paths != null ? ` · acted on ${rule.paths} paths` : "";
    card.innerHTML = `<div class="switch">${rule.on === false ? "OFF" : "ON"}</div><div><h3>${rule.name}${acted}</h3><p>${rule.says || ""}</p></div>`;
    wrap.append(card);
  });
  return wrap;
}

async function showHome(savedOnly = false) {
  const view = state.view;
  crumb.textContent = "Simulations";
  app.innerHTML = "<h1>Simulations</h1><p class='lede'>Start a run, track its progress, or open saved results.</p>";
  app.append(simulationTabs(savedOnly ? "saved" : "new"));
  if (!savedOnly) {
    const form = await simulationForm();
    if (view !== state.view) return;
    const progress = document.createElement("div");
    app.append(form, progress);
    form.addEventListener("simulation-started", () => progress.refresh?.());
    await watchJobs(progress);
    if (view !== state.view) return;
  }
  const title = document.createElement("h2");
  title.textContent = "Saved runs";
  app.append(title);
  const cards = document.createElement("div");
  cards.className = "cards";
  const more = document.createElement("button");
  more.className = "ghost";
  let offset = 0;
  const limit = 20;
  const load = async () => {
    more.disabled = true;
    const page = await get(`/api/runs?limit=${limit}&offset=${offset}`);
    if (view !== state.view) return;
    const items = page.items || page;
    if (!items.length && offset === 0) {
      app.insertAdjacentHTML("beforeend", "<p class='empty'>No runs yet.</p>");
      more.remove();
      return;
    }
    items.forEach((run) => {
      const btn = document.createElement("button");
      btn.className = "sim";
      const cls = run.median < 0 ? "loss" : "gain";
      btn.innerHTML = `
        <div class="when">${when(run.created)}${run.mode === "combined" ? " · combined run" : ` · seed ${run.seed}`}</div>
        <strong class="${cls}">${money(run.median)}</strong>
        <div class="meta">${run.mode === "combined" ? `Combined · ${run.configuration_count} configurations` : `${run.strategy} · ${run.lots} lots`} · ${run.paths} paths · ${run.horizon_days}d</div>
        <div class="meta">p05 ${money(run.p05)} · p95 ${money(run.p95)} · loss ${pct(run.prob_loss)}</div>`;
      btn.addEventListener("click", () => { location.hash = `#/run/${run.id}`; });
      cards.append(btn);
    });
    offset += items.length;
    const left = (page.total || items.length) - offset;
    more.hidden = left <= 0;
    more.disabled = false;
    more.textContent = `More (${left} left)`;
  };
  app.append(cards, more);
  more.addEventListener("click", load);
  await load();
}

async function showRun(id, params) {
  const view = state.view;
  state.run = id;
  crumb.innerHTML = `<a href="#/">Simulations</a> / <span>${id}</span>`;
  app.innerHTML = "<p class='lede'>Loading the run…</p>";
  const [run, paths] = await Promise.all([
    get(`/api/runs/${id}`),
    get(`/api/runs/${id}/paths`),
  ]);
  if (view !== state.view) return;
  const mc = run.monte_carlo || {};
  const book = run.book || {};
  const cfg = run.config || {};
  const combined = cfg.mode === "combined";
  app.innerHTML = "";
  const h = document.createElement("h1");
  h.textContent = combined ? `Combined simulation · ${cfg.configuration_count} configurations` : `${cfg.strategy || "wheel"} · ${cfg.lots || ""} lots`;
  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = combined ? `${when(run.created)} · ${mc.paths} pooled paths · horizons ${cfg.horizon_range.join("–")}d` : `${when(run.created)} · seed ${cfg.seed} · ${mc.paths} paths · horizon ${cfg.horizon_days}d`;
  const actions = exportBar(id);
  const explore = document.createElement("button");
  explore.className = "ghost";
  explore.textContent = "Explore paths";
  explore.addEventListener("click", () => {
    const pathsHeading = document.querySelector("#run-paths");
    pathsHeading.focus({ preventScroll: true });
    pathsHeading.scrollIntoView?.({ block: "start" });
  });
  actions.append(explore);
  app.append(h, lede, actions);
  const optimizeLink = document.createElement("a"); optimizeLink.className = "ghost";
  const configurations = new Map((run.configurations || []).map((group) => [group.id, group.config]));
  const excessHold = (path) => {
    const own = configurations.get(path.configuration) || cfg;
    return path.pnl - (path.ending_spot - own.spot) * own.lots * 100;
  };
  const worst = paths.reduce((a,b) => excessHold(a) < excessHold(b) ? a : b, paths[0]);
  optimizeLink.href = `#/optimize/${id}/${worst?.path ?? 0}`; optimizeLink.textContent = "Optimize weakest path versus stock";
  if (worst) app.append(optimizeLink);
  if (run.analysis) app.append(replayComparison(run.analysis));
  app.append(facts([
    ["Median P&L", money(mc.median)],
    ["Mean", money(mc.mean)],
    ["5th", money(mc.p05)],
    ["95th", money(mc.p95)],
    ["Chance of a loss", pct(mc.prob_loss)],
    ["Median drawdown", money(mc.median_drawdown)],
    [combined ? "Average deployed" : "Deployed", money(book.deployed)],
    [combined ? "Average opening floor" : "Floor", money(book.floor)],
  ]));

  const rulesTitle = document.createElement("h2");
  rulesTitle.textContent = "Rules on this run";
  const rulesNote = document.createElement("p");
  rulesNote.className = "note";
  rulesNote.textContent = "This is the snapshot taken when the run was made. ON was called. OFF was not. The count is how many paths a rule actually traded on.";
  const reread = document.createElement("button");
  reread.className = "ghost";
  reread.textContent = "Re-read this simulation with my rules";
  const status = document.createElement("p");
  status.className = "status";
  reread.addEventListener("click", async () => {
    reread.disabled = true;
    status.textContent = "Queuing the same seed with your selected rules…";
    try {
      await postJson(`/api/runs/${id}/reread`, rulePayload());
      location.hash = "#/active";
    } catch (error) {
      status.textContent = error.message;
      reread.disabled = false;
    }
  });
  const openPath = (path) => { location.hash = `#/run/${id}/path/${path}`; };
  let selectedBin = params.has("bin") && pathSelection?.run === id ? pathSelection : undefined;
  const selectBin = (label, matches) => {
    selectedBin = { label, matches };
    filter.value = "";
    page = 0;
    draw();
    chartModal?.close();
    pathTitle.focus({ preventScroll: true });
    pathTitle.scrollIntoView?.({ block: "start" });
  };
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  const ribbon = run.ribbon || [];
  if (ribbon.length) {
    grid.append(
      chartCard("Spot band", combined ? "5th to 95th of paths still active on each day. Line is the median." : "5th to 95th across every path. Line is the median.", bandChart(ribbon, "spot_p05", "spot_p50", "spot_p95")),
      chartCard("P&L band", "5th to 95th. Dashed line is zero.", bandChart(ribbon, "pnl_p05", "pnl_p50", "pnl_p95", { zero: true, money: true })),
      chartCard("Mark to market", "5th to 95th of cash plus marks.", bandChart(ribbon, "mtm_p05", "mtm_p50", "mtm_p95", { money: true })),
    );
  }
  const seriesHost = document.createElement("div");
  seriesHost.className = "chart-series";
  get(`/api/runs/${id}/series`).then((series) => {
    if (view !== state.view || !series?.length) return;
    seriesHost.append(
      chartCard(
        combined ? "Position, pooled median" : "Position, median path",
        "Contracts. Shares are divided by 100 so they sit next to the puts.",
        multiChart(series, [
          { key: "shares", color: "#5b4b8a", label: "shares ÷ 100" },
          { key: "puts", color: "#3d6b4f", label: "long puts" },
          { key: "shortCalls", color: "#8a6a2f", label: "short calls" },
          { key: "shortPuts", color: "#a33b32", label: "short puts" },
        ]),
      ),
      chartCard(
        "Put strike",
        "The floor is the lowest long put. The average is weighted by contracts. A step is a roll.",
        multiChart(series, [
          { key: "floor", color: "#3d6b4f", label: "floor strike" },
          { key: "avg", color: "#5b4b8a", label: "average strike" },
        ]),
      ),
    );
  }).catch(() => undefined);
  grid.append(
    chartCard("Where paths landed", paths.length > 2500 ? "Sampled paths and extremes. Select a point to open its path." : "Select a point to open its path.", scatter(paths, openPath)),
    chartCard("Spot ranges × P&L", "Each cell counts all paths in that spot and P&L range. Select a cell to explore its paths.", spotPnlHeatmap(paths, selectBin)),
    chartCard("P&L distribution", "Select a bar to filter paths by P&L range.", histogram(paths, selectBin)),
    chartCard("Drawdown vs. final P&L", "Compare the largest interim drawdown with the final result. Select a point to open its path.", scatter(paths, openPath, "max_dd", "Maximum drawdown")),
  );
  grid.append(seriesHost);
  app.append(grid);
  if (combined) app.append(combinedConfigurations(run, paths), reread, status);
  else app.append(rulesTitle, rulesNote, ruleSnapshot(run.rules), reread, status);

  const legs = [...(book.puts || []), ...(book.calls || [])];
  if (legs.length) {
    const t = document.createElement("h2");
    t.textContent = "Opening book";
    app.append(t, table(
      ["Side", "Qty", "Strike", "DTE", "Premium", "Tag"],
      legs.map((leg) => [leg.side, String(leg.qty), px(leg.strike), String(leg.dte), Number(leg.premium).toFixed(2), leg.tag])
    ));
  }

  const pathTitle = document.createElement("h2");
  pathTitle.id = "run-paths";
  pathTitle.tabIndex = -1;
  pathTitle.textContent = "Paths";
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "Select a column to sort or open any path by ID.";
  const jump = document.createElement("form");
  jump.className = "path-controls";
  jump.innerHTML = '<label>Path ID <input type="number" min="0" step="1" required></label><button class="ghost">Open path</button><span class="note" role="status"></span>';
  const pathIds = new Set(paths.map((row) => row.path));
  jump.addEventListener("submit", (event) => {
    event.preventDefault();
    const index = jump.querySelector("input").valueAsNumber;
    if (pathIds.has(index)) openPath(index);
    else jump.querySelector("[role=status]").textContent = "No path with that ID.";
  });
  const activeBin = document.createElement("div");
  activeBin.className = "path-controls";
  activeBin.hidden = true;
  app.append(pathTitle, note, jump, activeBin);
  const columns = ["path", ...(combined ? ["configuration"] : []), "ending_spot", "pnl", "min_spot", "max_dd", "ending_shares", "trades"];
  let key = columns.includes(params.get("sort")) ? params.get("sort") : "pnl";
  let dir = params.get("dir") === "1" ? 1 : -1;
  const filter = document.createElement("select");
  if (combined) {
    filter.setAttribute("aria-label", "Path configuration");
    filter.innerHTML = '<option value="">All configurations</option>' + (run.configurations || []).map((g) => `<option value="${g.id}">Configuration ${g.id}</option>`).join("");
    filter.value = params.get("config") || "";
    app.append(filter);
  }
  const host = document.createElement("div");
  const pager = document.createElement("div");
  pager.className = "exports";
  app.append(host, pager);
  let page = 0;
  const pageSize = 100;
  let ordered;
  const draw = (reuseOrder = false) => {
    if (!reuseOrder) ordered = paths.filter((path) =>
      (!combined || !filter.value || path.configuration === Number(filter.value)) &&
      (!selectedBin || selectedBin.matches(path))
    ).sort((a, b) => (a[key] - b[key]) * dir || a.path - b.path);
    activeBin.hidden = !selectedBin;
    if (selectedBin) {
      const label = document.createElement("strong");
      label.textContent = selectedBin.label;
      const clear = document.createElement("button");
      clear.className = "ghost";
      clear.textContent = "Clear chart filter";
      clear.addEventListener("click", () => { selectedBin = undefined; page = 0; draw(); });
      activeBin.replaceChildren(label, clear);
    }
    const pages = Math.max(1, Math.ceil(ordered.length / pageSize));
    page = Math.min(page, pages - 1);
    const slice = ordered.slice(page * pageSize, (page + 1) * pageSize);
    host.replaceChildren(table(
      ["Index", ...(combined ? ["Configuration"] : []), "Spot", "P&L", "Min spot", "Drawdown", "Shares", "Trades"],
      slice.map((r) => [String(r.path), ...(combined ? [String(r.configuration)] : []), px(r.ending_spot), money(r.pnl), px(r.min_spot), money(r.max_dd), String(r.ending_shares), String(r.trades)]),
      (i) => {
        const params = new URLSearchParams({ sort: key, dir });
        if (filter.value) params.set("config", filter.value);
        if (selectedBin) {
          pathSelection = { run: id, ...selectedBin };
          params.set("bin", "1");
        }
        location.hash = `#/run/${id}/path/${slice[i].path}?${params}`;
      }
    ));
    host.querySelectorAll("th").forEach((th, i) => {
      const button = document.createElement("button");
      button.textContent = th.textContent;
      if (key === columns[i]) th.setAttribute("aria-sort", dir === 1 ? "ascending" : "descending");
      button.addEventListener("click", () => {
        dir = key === columns[i] ? dir * -1 : (columns[i] === "path" ? 1 : -1);
        key = columns[i];
        page = 0;
        draw();
        host.querySelectorAll("th button")[i].focus();
      });
      th.replaceChildren(button);
    });
    pager.innerHTML = "";
    const label = document.createElement("span");
    label.className = "note";
    label.setAttribute("role", "status");
    label.textContent = ordered.length ? `${ordered.length.toLocaleString()} paths · ${page * pageSize + 1}–${Math.min(ordered.length, (page + 1) * pageSize)}` : "No matching paths";
    const prev = document.createElement("button");
    prev.className = "ghost";
    prev.textContent = "Previous";
    prev.disabled = page === 0;
    prev.addEventListener("click", () => { page -= 1; draw(true); });
    const next = document.createElement("button");
    next.className = "ghost";
    next.textContent = "Next";
    next.disabled = page >= pages - 1;
    next.addEventListener("click", () => { page += 1; draw(true); });
    pager.append(label, prev, next);
  };
  filter.addEventListener("change", () => { page = 0; draw(); });
  draw();
  if (params.has("paths")) pathTitle.scrollIntoView?.({ block: "start" });
}

function tradeText(trades) {
  return (trades || []).map((t) => {
    const rule = t.rule ? t.rule + " " : "";
    const qty = t.qty != null ? t.qty + "x " : "";
    const strike = t.strike != null ? t.strike : "";
    return `${rule}${t.action} ${qty}${strike} ${t.tag || ""}`.trim();
  }).join("; ");
}

async function showPath(id, index, params) {
  const view = state.view;
  state.run = id;
  state.path = index;
  crumb.innerHTML = `<a href="#/">Simulations</a> / <a href="#/run/${id}">${id}</a> / <span>Path ${index}</span>`;
  app.innerHTML = "<p class='lede'>Loading the path…</p>";
  const [row, paths] = await Promise.all([
    get(`/api/runs/${id}/paths/${index}`),
    get(`/api/runs/${id}/paths`),
  ]);
  if (view !== state.view) return;
  if (!row) throw new Error("Path unavailable.");
  const days = row.days || [];
  app.innerHTML = "";
  const h = document.createElement("h1");
  h.textContent = `Path ${index}${row.configuration ? ` · configuration ${row.configuration}` : ""}`;
  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = row.config ? `${row.config.lots} lots · ${row.config.horizon_days} days · seed ${row.config.seed} · call OTM ${row.config.tenors.map((t) => t.otm).join(", ")}. Click a day for the book before and after each rule.` : "Spot, P&L, the cover, and the open shorts. Click a day for the book before and after each rule.";
  app.append(h, lede, exportBar(id), facts([
    ["End spot", px(row.ending_spot)],
    ["P&L", money(row.pnl)],
    ["Min spot", px(row.min_spot)],
    ["Drawdown", money(row.max_dd)],
    ["Shares", String(row.ending_shares)],
    ["Assignments", String(row.assignments)],
    ["New calls", String(row.new_calls)],
    ["Roll cash", money(row.roll_cash)],
  ]));
  const optimizeLink = document.createElement("a"); optimizeLink.className = "ghost"; optimizeLink.href = `#/optimize/${id}/${index}`; optimizeLink.textContent = "Optimize this path";
  app.append(optimizeLink);
  const pathOrder = ["path", "configuration", "ending_spot", "pnl", "min_spot", "max_dd", "ending_shares", "trades"];
  const sortKey = pathOrder.includes(params.get("sort")) ? params.get("sort") : "path";
  const sortDir = params.get("dir") === "-1" ? -1 : 1;
  const bin = params.has("bin") && pathSelection?.run === id ? pathSelection : undefined;
  const configuration = params.get("config");
  const orderKey = JSON.stringify([id, sortKey, sortDir, configuration]);
  if (pathOrderCache?.key !== orderKey || pathOrderCache.paths !== paths || pathOrderCache.bin !== bin) {
    const ids = paths.filter((path) => (!configuration || String(path.configuration) === configuration) &&
      (!bin || bin.matches(path))).sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir || a.path - b.path).map((path) => path.path);
    pathOrderCache = { key: orderKey, paths, bin, ids, positions: new Map(ids.map((path, i) => [path, i])) };
  }
  const { ids: navPaths, positions } = pathOrderCache;
  const navIndex = positions.get(index) ?? -1;
  const nav = document.createElement("nav");
  nav.className = "path-controls";
  nav.setAttribute("aria-label", "Path navigation");
  const link = (label, href) => {
    const control = document.createElement(href ? "a" : "button");
    control.className = "ghost";
    control.textContent = label;
    if (href) control.href = href;
    else control.disabled = true;
    return control;
  };
  const destination = (path) => path == null ? null : `#/run/${id}/path/${path}?${params}`;
  const back = new URLSearchParams(params);
  back.set("paths", "1");
  const position = document.createElement("span");
  position.textContent = navIndex >= 0 ? `${navIndex + 1} of ${navPaths.length} paths` : `Path ${index}`;
  nav.append(link("Previous path", destination(navPaths[navIndex - 1])), position,
    link("Next path", destination(navIndex === -1 ? null : navPaths[navIndex + 1])),
    link("All paths", `#/run/${id}?${back}`));
  app.append(nav);
  if (!days.length) return;
  const initialShares = (row.config?.lots || 0) * 100 || days[0]?.position?.find((leg) => leg.name === "share")?.qty || 0;
  const holdPnl = initialShares * (days[days.length - 1].price - days[0].price);
  app.append(facts([["Stock-only P&L", money(holdPnl)], ["P&L above holding stock", money(row.pnl - holdPnl)]]));
  let dayIndex = days.length - 1;
  let selected = days[dayIndex];
  const dayIndices = new Map(days.map((day, i) => [day.day, i]));
  const tradeIndices = days.flatMap((day, i) => day.trades?.length ? [i] : []);
  const controls = document.createElement("div");
  controls.className = "path-controls day-controls";
  controls.innerHTML = `<button class="ghost" data-step="previous-trade">Previous trade</button>
    <button class="ghost" data-step="previous">Previous day</button>
    <label>Trading day <input type="range" min="0" max="${days.length - 1}" value="${dayIndex}"></label>
    <output aria-live="polite"></output>
    <button class="ghost" data-step="next">Next day</button>
    <button class="ghost" data-step="next-trade">Next trade</button>
    <button class="ghost" data-step="book">View day book</button>`;
  const slider = controls.querySelector("input");
  const output = controls.querySelector("output");
  const dayOptions = { day: () => selected.day, selectDay: (day) => selectDay(dayIndices.get(day)) };
  const charts = document.createElement("div");
  charts.className = "chart-grid";
  let peak = -Infinity;
  const cover = days.map((d) => {
    peak = Math.max(peak, d.pnl);
    return {
      ...d, spot: d.price, tradeSpot: d.trades?.length ? d.price : null,
      stockHold: initialShares * (d.price - days[0].price), drawdown: d.pnl - peak,
      ...bookCounts(d), ...putStrikes(d),
    };
  });
  charts.append(
    chartCard("Spot", "Select a day on the chart to inspect its book. Gold dots mark trades.", multiChart(cover, [
      { key: "spot", label: "Spot", color: chartColors.accent },
      { key: "tradeSpot", label: "Trade day", type: "scatter", color: chartColors.gain },
    ], dayOptions)),
    chartCard("P&L vs. holding shares", "Compare strategy P&L with a stock-only position on the same prices.", multiChart(cover, [
      { key: "pnl", label: "Strategy", color: chartColors.accent },
      { key: "stockHold", label: "Stock only", color: "#248080" },
    ], { ...dayOptions, zero: true, money: true })),
    chartCard("Mark to market", "Cash plus marks on shares and options.", multiChart(cover, [
      { key: "mtm", label: "Mark to market", color: chartColors.accent },
    ], { ...dayOptions, money: true })),
    chartCard("Drawdown over time", "P&L below its running peak, starting at day zero.", multiChart(cover, [
      { key: "drawdown", label: "Drawdown", color: chartColors.loss, areaStyle: { opacity: 0.12 } },
    ], { ...dayOptions, zero: true, money: true })),
    chartCard("Cover", "Shares ÷ 100 against the long puts. They should stay near each other.", multiChart(cover, [
      { key: "shares", color: "#5b4b8a", label: "shares ÷ 100" },
      { key: "puts", color: "#3d6b4f", label: "long puts" },
    ], dayOptions)),
    chartCard("Put strike", "Spot against the floor and the average long put. A step up is the roll.", multiChart(cover, [
      { key: "spot", color: "#8a837c", label: "spot" },
      { key: "floor", color: "#3d6b4f", label: "floor strike" },
      { key: "avg", color: "#5b4b8a", label: "average strike" },
    ], dayOptions)),
    chartCard("Short options open", "Calls still sold, and puts sold to get the shares back.", multiChart(cover, [
      { key: "shortCalls", color: "#8a6a2f", label: "short calls" },
      { key: "shortPuts", color: "#a33b32", label: "short puts" },
    ], dayOptions)),
  );
  app.append(controls, charts);

  const dayTitle = document.createElement("h2");
  dayTitle.textContent = "Days";
  const dayHost = document.createElement("div");
  const bookHost = document.createElement("div");
  bookHost.id = "book";
  app.append(bookHost, dayHost);
  const dayTable = table(
    ["Day", "Spot", "P&L", "Shares", "Trades"],
    days.map((d) => [String(d.day), px(d.price), money(d.pnl), String(d.shares), tradeText(d.trades)]),
    (i) => selectDay(i)
  );
  dayHost.append(dayTitle, dayTable);
  const dayRows = dayTable.querySelectorAll("tbody tr");
  const previousTrade = () => tradeIndices.findLast((i) => i < dayIndex);
  const nextTrade = () => tradeIndices.find((i) => i > dayIndex);
  const selectDay = (i) => {
    if (!days[i]) return;
    dayRows[dayIndex].classList.remove("on");
    dayRows[dayIndex].removeAttribute("aria-current");
    dayIndex = i;
    selected = days[i];
    dayRows[i].classList.add("on");
    dayRows[i].setAttribute("aria-current", "true");
    slider.value = i;
    slider.setAttribute("aria-valuetext", `Day ${selected.day}`);
    output.textContent = `Day ${selected.day} · ${money(selected.pnl)} P&L · ${selected.trades?.length || 0} trades`;
    controls.querySelector('[data-step="previous"]').disabled = i === 0;
    controls.querySelector('[data-step="next"]').disabled = i === days.length - 1;
    controls.querySelector('[data-step="previous-trade"]').disabled = previousTrade() == null;
    controls.querySelector('[data-step="next-trade"]').disabled = nextTrade() == null;
    const surfaces = [...charts.querySelectorAll(".chart-surface"), ...(chartModal?.querySelectorAll(".chart-surface") || [])];
    for (const surface of surfaces) surface.dispatchEvent(new Event("desk:day"));
    drawBook();
  };
  slider.addEventListener("input", () => selectDay(slider.valueAsNumber));
  controls.querySelector('[data-step="previous"]').addEventListener("click", () => selectDay(dayIndex - 1));
  controls.querySelector('[data-step="next"]').addEventListener("click", () => selectDay(dayIndex + 1));
  controls.querySelector('[data-step="previous-trade"]').addEventListener("click", () => selectDay(previousTrade()));
  controls.querySelector('[data-step="next-trade"]').addEventListener("click", () => selectDay(nextTrade()));
  controls.querySelector('[data-step="book"]').addEventListener("click", () => bookHost.scrollIntoView?.({ block: "start" }));
  const drawBook = () => {
    bookHost.replaceChildren();
    const title = document.createElement("h2");
    title.textContent = `Day ${selected.day} close`;
    const ticks = document.createElement("div");
    ticks.className = "ticks";
    (selected.rules || []).forEach((rule) => {
      const span = document.createElement("span");
      span.className = `tick ${rule.acted ? "acted" : "idle"}`;
      span.textContent = `${rule.name} ${rule.acted ? "acted" : "idle"}`;
      ticks.append(span);
    });
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "Gold means the rule traded that day. Idle means it was on, looked, and did nothing.";
    bookHost.append(title, ticks, note, positionTable(selected.position));
    (selected.steps || []).forEach((step) => {
      const details = document.createElement("details");
      details.open = true;
      const summary = document.createElement("summary");
      summary.textContent = `${step.rule} · ${step.trades.length} trade${step.trades.length === 1 ? "" : "s"}`;
      const trades = document.createElement("p");
      trades.className = "note";
      trades.textContent = tradeText(step.trades);
      const before = document.createElement("h3");
      before.textContent = "Before";
      const after = document.createElement("h3");
      after.textContent = "After";
      details.append(summary, trades, before, positionTable(step.before), after, positionTable(step.after));
      bookHost.append(details);
    });
  };
  selectDay(dayIndex);
}

function positionTable(rows) {
  const lines = rows || [];
  const total = lines.reduce((n, r) => n + (Number(r.mtm) || 0), 0);
  const cells = lines.map((r) => [
    r.name, r.side || "", String(r.qty),
    r.strike == null ? "" : px(r.strike),
    r.expiry == null ? "" : String(r.expiry),
    r.name === "cash" ? "" : (r.name === "share" ? px(r.cost) : Number(r.cost).toFixed(2)),
    r.name === "cash" ? money(r.mark) : (r.name === "share" ? px(r.mark) : Number(r.mark).toFixed(2)),
    money(r.mtm),
  ]);
  cells.push(["Book", "", "", "", "", "", "", money(total)]);
  return table(["Name", "Side", "Qty", "Strike", "Expiry", "Cost", "Mark", "MTM"], cells);
}

function navigate() {
  const view = state.view + 1;
  route().catch((error) => {
    if (view !== state.view) return;
    app.replaceChildren();
    const note = document.createElement("p");
    note.className = "status";
    note.textContent = error.message;
    app.append(note);
  });
}
window.addEventListener("hashchange", navigate);
navigate();
