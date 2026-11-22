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

async function get(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
      tr.append(td);
    });
    if (onClick) tr.addEventListener("click", () => onClick(i, tr));
    body.append(tr);
  });
  el.append(head, body);
  grid.append(el);
  return grid;
}

function frame(width, height, left, right, top, bottom, xMin, xMax, yMin, yMax, yLabel) {
  let grid = "";
  for (let i = 0; i < 5; i++) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const yy = top + (1 - i / 4) * (height - top - bottom);
    grid += `<line x1="${left}" y1="${yy.toFixed(1)}" x2="${width - right}" y2="${yy.toFixed(1)}" stroke="#3a342b"/>`;
    grid += `<text x="${left - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${yLabel(value)}</text>`;
  }
  const ticks = 6;
  for (let i = 0; i < ticks; i++) {
    const day = xMin + ((xMax - xMin) * i) / (ticks - 1);
    const xx = left + (i / (ticks - 1)) * (width - left - right);
    grid += `<line x1="${xx.toFixed(1)}" y1="${top}" x2="${xx.toFixed(1)}" y2="${height - bottom}" stroke="#2a261f"/>`;
    grid += `<text x="${xx.toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${Math.round(day)}</text>`;
  }
  grid += `<text x="${(left + width - right) / 2}" y="${height - 2}" text-anchor="middle" fill="#6f675c" font-size="10" font-family="ui-monospace,monospace">day</text>`;
  return { grid, x: (day) => left + ((day - xMin) / (xMax - xMin || 1)) * (width - left - right), y: (v) => top + (1 - (v - yMin) / (yMax - yMin || 1)) * (height - top - bottom) };
}

function bandChart(rows, lowKey, midKey, highKey, opts = {}) {
  const width = 720, height = 280, left = 78, right = 16, top = 16, bottom = 36;
  const lows = rows.map((d) => d[lowKey]), mids = rows.map((d) => d[midKey]), highs = rows.map((d) => d[highKey]);
  let y0 = Math.min(...lows), y1 = Math.max(...highs);
  if (opts.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  const span = y1 - y0 || 1;
  y0 -= span * 0.06; y1 += span * 0.06;
  const last = rows[rows.length - 1].day || 1;
  const f = frame(width, height, left, right, top, bottom, 0, last, y0, y1, opts.money ? money : (v) => v.toFixed(1));
  const upper = rows.map((d) => `${f.x(d.day).toFixed(1)},${f.y(d[highKey]).toFixed(1)}`).join(" ");
  const lower = rows.slice().reverse().map((d) => `${f.x(d.day).toFixed(1)},${f.y(d[lowKey]).toFixed(1)}`).join(" ");
  const mid = rows.map((d) => `${f.x(d.day).toFixed(1)},${f.y(d[midKey]).toFixed(1)}`).join(" ");
  const zero = opts.zero && y0 <= 0 && y1 >= 0
    ? `<line x1="${left}" y1="${f.y(0).toFixed(1)}" x2="${width - right}" y2="${f.y(0).toFixed(1)}" stroke="#8a8172" stroke-dasharray="3 4"/>`
    : "";
  return `<svg viewBox="0 0 ${width} ${height}">${f.grid}${zero}<polygon points="${upper} ${lower}" fill="#e6d3a3" opacity="0.16"/><polyline points="${mid}" fill="none" stroke="#e6d3a3" stroke-width="1.8"/></svg>`;
}

function multiChart(rows, series, opts = {}) {
  const width = 720, height = 280, left = 78, right = 16, top = 28, bottom = 36;
  const all = [0];
  series.forEach((s) => rows.forEach((r) => all.push(r[s.key])));
  let y0 = Math.min(...all), y1 = Math.max(...all);
  if (y1 === y0) y1 += 1;
  const span = y1 - y0 || 1;
  y0 -= span * 0.06; y1 += span * 0.08;
  const last = rows[rows.length - 1].day || 1;
  const f = frame(width, height, left, right, top, bottom, 0, last, y0, y1, opts.money ? money : (v) => String(Math.round(v)));
  let legend = "";
  series.forEach((s, i) => {
    const x0 = left + i * 150;
    legend += `<line x1="${x0}" y1="12" x2="${x0 + 18}" y2="12" stroke="${s.color}" stroke-width="2"/>`;
    legend += `<text x="${x0 + 24}" y="16" fill="#c8c0b4" font-size="11" font-family="ui-monospace,monospace">${s.label}</text>`;
  });
  const lines = series.map((s) => {
    const points = rows.map((r) => `${f.x(r.day).toFixed(1)},${f.y(r[s.key]).toFixed(1)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}">${f.grid}${legend}${lines}</svg>`;
}
function chart(days, key, opts = {}) {
  const ordered = days.slice().sort((a, b) => a.day - b.day);
  const width = 720, height = 280, left = 78, right = 16, top = 16, bottom = 36;
  const values = ordered.map((d) => d[key]);
  let low = Math.min(...values), high = Math.max(...values);
  if (opts.zero) { low = Math.min(low, 0); high = Math.max(high, 0); }
  const span = high - low || 1;
  low -= span * 0.06; high += span * 0.06;
  const last = ordered[ordered.length - 1].day || 1;
  const f = frame(width, height, left, right, top, bottom, 0, last, low, high, opts.money ? money : (v) => v.toFixed(1));
  const line = ordered.map((d) => `${f.x(d.day).toFixed(1)},${f.y(d[key]).toFixed(1)}`).join(" ");
  const dots = ordered.filter((d) => d.trades && d.trades.length).map((d) =>
    `<circle cx="${f.x(d.day).toFixed(1)}" cy="${f.y(d[key]).toFixed(1)}" r="3.2" fill="#f3efe4"/>`
  ).join("");
  const zero = opts.zero && low <= 0 && high >= 0
    ? `<line x1="${left}" y1="${f.y(0).toFixed(1)}" x2="${width - right}" y2="${f.y(0).toFixed(1)}" stroke="#8a8172" stroke-dasharray="3 4"/>`
    : "";
  return `<svg viewBox="0 0 ${width} ${height}">${f.grid}${zero}<polyline points="${line}" fill="none" stroke="#e6d3a3" stroke-width="1.8"/>${dots}</svg>`;
}

function linesChart(days, series) {
  const ordered = days.slice().sort((a, b) => a.day - b.day);
  const counts = ordered.map(bookCounts);
  const width = 1040, height = 260, left = 78, right = 16, top = 28, bottom = 32;
  const all = [0];
  series.forEach((s) => counts.forEach((c) => all.push(c[s.key])));
  let low = Math.min(...all), high = Math.max(...all);
  if (high === low) high += 1;
  const span = high - low || 1;
  low -= span * 0.06; high += span * 0.08;
  const last = ordered[ordered.length - 1].day || 1;
  const x = (day) => left + (day / last) * (width - left - right);
  const y = (v) => top + (1 - (v - low) / (high - low)) * (height - top - bottom);
  let grid = "";
  for (let i = 0; i < 5; i++) {
    const value = low + ((high - low) * i) / 4;
    const yy = y(value);
    grid += `<line x1="${left}" y1="${yy.toFixed(1)}" x2="${width - right}" y2="${yy.toFixed(1)}" stroke="#3a342b"/>`;
    grid += `<text x="${left - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${Math.round(value)}</text>`;
  }
  for (let i = 0; i < 6; i++) {
    const day = Math.round((last * i) / 5);
    const xx = x(day);
    grid += `<line x1="${xx.toFixed(1)}" y1="${top}" x2="${xx.toFixed(1)}" y2="${height - bottom}" stroke="#2a261f"/>`;
    grid += `<text x="${xx.toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${day}</text>`;
  }
  let legend = "";
  series.forEach((s, i) => {
    const x0 = left + i * 170;
    legend += `<line x1="${x0}" y1="12" x2="${x0 + 22}" y2="12" stroke="${s.color}" stroke-width="2"/>`;
    legend += `<text x="${x0 + 28}" y="16" fill="#c8c0b4" font-size="12" font-family="ui-monospace,monospace">${s.label}</text>`;
  });
  const lines = series.map((s) => {
    const points = ordered.map((d, i) => `${x(d.day).toFixed(1)},${y(counts[i][s.key]).toFixed(1)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}">${grid}${legend}${lines}</svg>`;
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

function scatter(rows) {
  const width = 720, height = 280, left = 78, right = 16, top = 16, bottom = 36;
  const xs = rows.map((r) => r.ending_spot), ys = rows.map((r) => r.pnl);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys, 0), y1 = Math.max(...ys, 0);
  const sx = x1 - x0 || 1, sy = y1 - y0 || 1;
  x0 -= sx * 0.06; x1 += sx * 0.06; y0 -= sy * 0.06; y1 += sy * 0.06;
  const f = frame(width, height, left, right, top, bottom, x0, x1, y0, y1, money);
  const dots = rows.map((r) =>
    `<a href="#/run/${state.run}/path/${r.path}"><circle cx="${f.x(r.ending_spot).toFixed(1)}" cy="${f.y(r.pnl).toFixed(1)}" r="4" fill="${r.pnl < 0 ? "#d07a62" : "#e6d3a3"}"><title>Path ${r.path} ${money(r.pnl)}</title></circle></a>`
  ).join("");
  const zero = y0 <= 0 && y1 >= 0
    ? `<line x1="${left}" y1="${f.y(0).toFixed(1)}" x2="${width - right}" y2="${f.y(0).toFixed(1)}" stroke="#8a8172" stroke-dasharray="3 4"/>`
    : "";
  return `<svg viewBox="0 0 ${width} ${height}">${f.grid}${zero}${dots}</svg>`;
}

function histogram(rows) {
  const values = rows.map((r) => r.pnl);
  const bins = 18;
  const low = Math.min(...values), high = Math.max(...values);
  const span = high - low || 1;
  const counts = Array(bins).fill(0);
  values.forEach((v) => {
    const i = Math.min(bins - 1, Math.floor(((v - low) / span) * bins));
    counts[i] += 1;
  });
  const max = Math.max(...counts, 1);
  const width = 720, height = 240, left = 78, right = 16, top = 16, bottom = 36;
  const f = frame(width, height, left, right, top, bottom, low, high, 0, max, (v) => String(Math.round(v)));
  const bw = (width - left - right) / bins;
  let bars = "";
  counts.forEach((c, i) => {
    const h = (c / max) * (height - top - bottom);
    const x = left + i * bw;
    const y = height - bottom - h;
    const mid = low + ((i + 0.5) / bins) * span;
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${mid < 0 ? "#d07a62" : "#e6d3a3"}" opacity="0.85"><title>${money(mid)} · ${c} paths</title></rect>`;
  });
  return `<svg viewBox="0 0 ${width} ${height}">${f.grid}${bars}</svg>`;
}

function chartCard(title, note, svg) {
  const card = document.createElement("article");
  card.className = "chart-card";
  const head = document.createElement("header");
  const text = document.createElement("div");
  text.innerHTML = `<h3>${title}</h3><p>${note}</p>`;
  const btn = document.createElement("button");
  btn.className = "max";
  btn.textContent = "Maximize";
  btn.addEventListener("click", () => maximize(title, note, svg));
  head.append(text, btn);
  const body = document.createElement("div");
  body.innerHTML = svg;
  card.append(head, body);
  return card;
}

function maximize(title, note, svg) {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `<div class="sheet"><header><div><h2>${title}</h2><p class="note">${note}</p></div><button>Close</button></header>${svg}</div>`;
  modal.querySelector("button").addEventListener("click", () => modal.remove());
  modal.addEventListener("click", (event) => { if (event.target === modal) modal.remove(); });
  document.body.append(modal);
}

const state = { run: null, path: null, day: null };

function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  crumb.textContent = "";
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("on", link.getAttribute("href") === `#/${parts[0] || ""}`.replace("#//", "#/"));
  });
  if (parts[0] === "rules") return showRules();
  if (parts[0] === "config") return showConfig();
  if (parts[0] === "run" && parts[2] === "path") return showPath(parts[1], Number(parts[3]));
  if (parts[0] === "run") return showRun(parts[1]);
  return showHome();
}

function chosenRules() {
  const saved = localStorage.getItem("desk-rules");
  return saved ? JSON.parse(saved) : null;
}

function saveRules(names) {
  localStorage.setItem("desk-rules", JSON.stringify(names));
}

async function showRules() {
  crumb.textContent = "Rules";
  app.innerHTML = "<h1>Rules</h1><p class='lede'>A rule is on or off. Off means it is not called, so it cannot act. This set is used for the next simulation, and for re-reading a saved one.</p>";
  const rules = await get("/api/rules");
  const picked = new Set(chosenRules() || rules.map((rule) => rule.name));
  const list = document.createElement("div");
  const draw = () => {
    list.replaceChildren();
    rules.forEach((rule) => {
      const on = picked.has(rule.name);
      const card = document.createElement("article");
      card.className = `rule ${on ? "on" : "off"}`;
      const knobs = Object.entries(rule).filter(([key]) => !["name", "says", "on", "paths"].includes(key))
        .map(([key, value]) => `${key} ${value}`).join(" · ");
      card.innerHTML = `
        <button class="switch">${on ? "ON" : "OFF"}</button>
        <div><h3>${rule.name}</h3><p>${rule.says}</p>${knobs ? `<div class="knobs">${knobs}</div>` : ""}</div>`;
      card.querySelector("button").addEventListener("click", () => {
        if (picked.has(rule.name)) picked.delete(rule.name);
        else picked.add(rule.name);
        saveRules([...picked]);
        draw();
      });
      list.append(card);
    });
  };
  draw();
  app.append(list);
}

async function showConfig() {
  crumb.textContent = "Configuration";
  app.innerHTML = "<h1>Configuration</h1><p class='lede'>These are the defaults a new simulation starts from. A saved run keeps the values it was made with.</p>";
  const fields = [
    ["Lots", "100"],
    ["Horizon", "100 days"],
    ["Paths", "100"],
    ["Seed", "13"],
    ["Spot", "59.22"],
    ["IV", "0.71"],
    ["Put sleeves", "10% at 60d, 40% at 90d, 50% at 180d"],
    ["Call OTM", "3%, 5%, 8%, 10%"],
    ["Contracts", "12, 12, 12, 10"],
    ["Strategy", "wheel"],
  ];
  app.append(facts(fields));
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "Change them on the Simulations page when you start a run. The rules switches are on the Rules page.";
  app.append(note);
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

async function showHome() {
  crumb.textContent = "Simulations";
  app.innerHTML = "<h1>Simulations</h1><p class='lede'>Start a run, or open one that is already saved.</p>";
  const form = document.createElement("form");
  form.className = "form";
  form.innerHTML = `
    <label>Lots<input name="lots" value="100"/></label>
    <label>Horizon days<input name="horizon" value="100"/></label>
    <label>Paths<input name="paths" value="100"/></label>
    <label>Seed<input name="seed" value="13"/></label>
    <label>Spot<input name="spot" value="59.22"/></label>
    <label>IV<input name="iv" value="0.71"/></label>
    <label>Strategy<select name="strategy"><option>wheel</option><option>hold</option></select></label>
    <label>Contracts<input name="contracts" value="12,12,12,10"/></label>
    <label class="wide">Put sleeves percent:dte:roll<input name="puts" value="10:60:30,40:90:45,50:180:90"/></label>
    <label class="wide">Call OTM 1w,2w,3w,4w<input name="otm" value="0.03,0.05,0.08,0.10"/></label>
    <button class="run" type="submit">Run simulation</button>`;
  const status = document.createElement("p");
  status.className = "status";
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = Object.fromEntries(new FormData(form).entries());
    const picked = chosenRules();
    if (picked) body.rules = picked.join(",");
    form.querySelector("button").disabled = true;
    status.textContent = "Starting…";
    const started = await fetch("/api/simulate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
    const tick = async () => {
      const job = await get(`/api/jobs/${started.job}`);
      if (job.status === "done") {
        status.textContent = "Done.";
        location.hash = `#/run/${job.run}`;
        return;
      }
      if (job.status === "error") {
        status.textContent = job.error || "The run failed.";
        form.querySelector("button").disabled = false;
        return;
      }
      status.textContent = "Running. A hundred paths takes a little while.";
      setTimeout(tick, 2000);
    };
    tick();
  });
  app.append(form, status);
  const runs = await get("/api/runs");
  if (!runs.length) {
    app.insertAdjacentHTML("beforeend", "<p class='empty'>No runs yet.</p>");
    return;
  }
  const cards = document.createElement("div");
  cards.className = "cards";
  runs.forEach((run) => {
    const btn = document.createElement("button");
    btn.className = "sim";
    const cls = run.median < 0 ? "loss" : "gain";
    btn.innerHTML = `
      <div class="when">${when(run.created)} · seed ${run.seed}</div>
      <strong class="${cls}">${money(run.median)}</strong>
      <div class="meta">${run.strategy} · ${run.paths} paths · ${run.lots} lots · ${run.horizon_days}d</div>
      <div class="meta">p05 ${money(run.p05)} · p95 ${money(run.p95)} · loss ${pct(run.prob_loss)}</div>`;
    btn.addEventListener("click", () => { location.hash = `#/run/${run.id}`; });
    cards.append(btn);
  });
  app.append(cards);
}

async function showRun(id) {
  state.run = id;
  crumb.innerHTML = `<a href="#/">Simulations</a> / <span>${id}</span>`;
  app.innerHTML = "<p class='lede'>Loading the run…</p>";
  const [run, paths, series] = await Promise.all([
    get(`/api/runs/${id}`),
    get(`/api/runs/${id}/paths`),
    get(`/api/runs/${id}/series`),
  ]);
  const mc = run.monte_carlo || {};
  const book = run.book || {};
  const cfg = run.config || {};
  app.innerHTML = "";
  const h = document.createElement("h1");
  h.textContent = `${cfg.strategy || "wheel"} · ${cfg.lots || ""} lots`;
  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = `${when(run.created)} · seed ${cfg.seed} · ${mc.paths} paths · horizon ${cfg.horizon_days}d. Maximize any chart. Click a point or a row for the path.`;
  app.append(h, lede);
  app.append(facts([
    ["Median P&L", money(mc.median)],
    ["Mean", money(mc.mean)],
    ["5th", money(mc.p05)],
    ["95th", money(mc.p95)],
    ["Chance of a loss", pct(mc.prob_loss)],
    ["Median drawdown", money(mc.median_drawdown)],
    ["Deployed", money(book.deployed)],
    ["Floor", money(book.floor)],
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
    status.textContent = "Re-reading the same seed under the rules you have switched on…";
    const picked = chosenRules();
    const started = await fetch(`/api/runs/${id}/reread`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ rules: picked ? picked.join(",") : "" }),
    }).then((r) => r.json());
    const tick = async () => {
      const job = await get(`/api/jobs/${started.job}`);
      if (job.status === "done") { location.hash = `#/run/${job.run}`; return; }
      if (job.status === "error") { status.textContent = job.error || "The re-read failed."; reread.disabled = false; return; }
      setTimeout(tick, 2000);
    };
    tick();
  });
  app.append(rulesTitle, rulesNote, ruleSnapshot(run.rules), reread, status);

  const grid = document.createElement("div");
  grid.className = "chart-grid";
  const ribbon = run.ribbon || [];
  if (ribbon.length) {
    grid.append(
      chartCard("Spot band", "5th to 95th across every path. Line is the median.", bandChart(ribbon, "spot_p05", "spot_p50", "spot_p95")),
      chartCard("P&L band", "5th to 95th. Dashed line is zero.", bandChart(ribbon, "pnl_p05", "pnl_p50", "pnl_p95", { zero: true, money: true })),
      chartCard("Mark to market", "5th to 95th of cash plus marks.", bandChart(ribbon, "mtm_p05", "mtm_p50", "mtm_p95", { money: true })),
    );
  }
  if (series.length) {
    grid.append(chartCard(
      "Position, median path",
      "Contracts. Shares are divided by 100 so they sit next to the puts.",
      multiChart(series, [
        { key: "shares", color: "#e6d3a3", label: "shares ÷ 100" },
        { key: "puts", color: "#7d9a84", label: "long puts" },
        { key: "shortCalls", color: "#c8b48a", label: "short calls" },
        { key: "shortPuts", color: "#d07a62", label: "short puts" },
      ]),
    ));
  }
  grid.append(
    chartCard("Where paths landed", "Horizontal is ending spot, vertical is P&L. Click a point.", scatter(paths)),
    chartCard("P&L distribution", "Brick is a loss. Gold is a gain. Left axis is the path count.", histogram(paths)),
  );
  app.append(grid);

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
  pathTitle.textContent = "Paths";
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "Sorted by P&L. Click a row for the day-by-day book.";
  app.append(pathTitle, note);
  let key = "pnl", dir = -1;
  const host = document.createElement("div");
  app.append(host);
  const draw = () => {
    const ordered = paths.slice().sort((a, b) => (a[key] - b[key]) * dir);
    host.replaceChildren(table(
      ["Index", "Spot", "P&L", "Min spot", "Drawdown", "Shares", "Trades"],
      ordered.map((r) => [String(r.path), px(r.ending_spot), money(r.pnl), px(r.min_spot), money(r.max_dd), String(r.ending_shares), String(r.trades)]),
      (i) => { location.hash = `#/run/${id}/path/${ordered[i].path}`; }
    ));
    host.querySelectorAll("th").forEach((th, i) => {
      const map = ["path", "ending_spot", "pnl", "min_spot", "max_dd", "ending_shares", "trades"];
      th.addEventListener("click", () => {
        dir = key === map[i] ? dir * -1 : (map[i] === "path" ? 1 : -1);
        key = map[i];
        draw();
      });
    });
  };
  draw();
}

function ribbonChart(ribbon) {
  const width = 1040, height = 240, left = 78, right = 16, top = 16, bottom = 32;
  const lows = ribbon.map((d) => d.spot_p05), mids = ribbon.map((d) => d.spot_p50), highs = ribbon.map((d) => d.spot_p95);
  let low = Math.min(...lows), high = Math.max(...highs);
  const span = high - low || 1;
  low -= span * 0.06; high += span * 0.06;
  const last = ribbon[ribbon.length - 1].day || 1;
  const x = (day) => left + (day / last) * (width - left - right);
  const y = (v) => top + (1 - (v - low) / (high - low)) * (height - top - bottom);
  const band = ribbon.map((d) => `${x(d.day).toFixed(1)},${y(d.spot_p95).toFixed(1)}`).join(" ")
    + " " + ribbon.slice().reverse().map((d) => `${x(d.day).toFixed(1)},${y(d.spot_p05).toFixed(1)}`).join(" ");
  const mid = ribbon.map((d) => `${x(d.day).toFixed(1)},${y(d.spot_p50).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${width} ${height}"><polygon points="${band}" fill="#e6d3a3" opacity="0.18"/><polyline points="${mid}" fill="none" stroke="#e6d3a3" stroke-width="1.8"/></svg>`;
}

function tradeText(trades) {
  return (trades || []).map((t) => {
    const rule = t.rule ? t.rule + " " : "";
    const qty = t.qty != null ? t.qty + "x " : "";
    const strike = t.strike != null ? t.strike : "";
    return `${rule}${t.action} ${qty}${strike} ${t.tag || ""}`.trim();
  }).join("; ");
}

async function showPath(id, index) {
  state.run = id;
  state.path = index;
  crumb.innerHTML = `<a href="#/">Simulations</a> / <a href="#/run/${id}">${id}</a> / <span>Path ${index}</span>`;
  app.innerHTML = "<p class='lede'>Loading the path…</p>";
  const row = await get(`/api/runs/${id}/paths/${index}`);
  const days = row.days || [];
  app.innerHTML = "";
  const h = document.createElement("h1");
  h.textContent = `Path ${index}`;
  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = "Spot, P&L, the cover, and the open shorts. Click a day for the book before and after each rule.";
  app.append(h, lede, facts([
    ["End spot", px(row.ending_spot)],
    ["P&L", money(row.pnl)],
    ["Min spot", px(row.min_spot)],
    ["Drawdown", money(row.max_dd)],
    ["Shares", String(row.ending_shares)],
    ["Assignments", String(row.assignments)],
    ["New calls", String(row.new_calls)],
    ["Roll cash", money(row.roll_cash)],
  ]));
  const charts = document.createElement("div");
  charts.className = "chart-grid";
  const cover = days.map((d) => ({ day: d.day, ...bookCounts(d) }));
  charts.append(
    chartCard("Spot", "Left is the price. Dots are trade days.", chart(days, "price")),
    chartCard("P&L", "Dashed line is zero.", chart(days, "pnl", { zero: true, money: true })),
    chartCard("Mark to market", "Cash plus the marks on shares and options.", chart(days, "mtm", { money: true })),
    chartCard("Cover", "Shares ÷ 100 against the long puts. They should stay near each other.", multiChart(cover, [
      { key: "shares", color: "#e6d3a3", label: "shares ÷ 100" },
      { key: "puts", color: "#7d9a84", label: "long puts" },
    ])),
    chartCard("Short options open", "Calls still sold, and puts sold to get the shares back.", multiChart(cover, [
      { key: "shortCalls", color: "#e6d3a3", label: "short calls" },
      { key: "shortPuts", color: "#d07a62", label: "short puts" },
    ])),
  );
  app.append(charts);

  const dayTitle = document.createElement("h2");
  dayTitle.textContent = "Days";
  app.append(dayTitle);
  const dayHost = document.createElement("div");
  const bookHost = document.createElement("div");
  bookHost.id = "book";
  app.append(dayHost, bookHost);
  let selected = days[days.length - 1];
  const drawDays = () => {
    dayHost.replaceChildren(table(
      ["Day", "Spot", "P&L", "Shares", "Trades"],
      days.map((d) => [String(d.day), px(d.price), money(d.pnl), String(d.shares), tradeText(d.trades)]),
      (i) => { selected = days[i]; drawDays(); drawBook(); }
    ));
    dayHost.querySelectorAll("tbody tr").forEach((tr, i) => {
      if (days[i].day === selected.day) tr.className = "on";
    });
  };
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
  drawDays();
  drawBook();
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

window.addEventListener("hashchange", route);
route();
