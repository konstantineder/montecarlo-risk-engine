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

function chart(days, key, opts = {}) {
  const ordered = days.slice().sort((a, b) => a.day - b.day);
  const width = 1040, height = 260, left = 78, right = 16, top = 16, bottom = 32;
  const values = ordered.map((d) => d[key]);
  let low = Math.min(...values), high = Math.max(...values);
  if (opts.zero) { low = Math.min(low, 0); high = Math.max(high, 0); }
  const span = high - low || 1;
  low -= span * 0.06; high += span * 0.06;
  const last = ordered[ordered.length - 1].day || 1;
  const x = (day) => left + (day / last) * (width - left - right);
  const y = (v) => top + (1 - (v - low) / (high - low)) * (height - top - bottom);
  const label = opts.money ? (v) => money(v) : (v) => v.toFixed(1);
  let grid = "";
  for (let i = 0; i < 5; i++) {
    const value = low + ((high - low) * i) / 4;
    const yy = y(value);
    grid += `<line x1="${left}" y1="${yy.toFixed(1)}" x2="${width - right}" y2="${yy.toFixed(1)}" stroke="#3a342b"/>`;
    grid += `<text x="${left - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${label(value)}</text>`;
  }
  for (let i = 0; i < 6; i++) {
    const day = Math.round((last * i) / 5);
    const xx = x(day);
    grid += `<line x1="${xx.toFixed(1)}" y1="${top}" x2="${xx.toFixed(1)}" y2="${height - bottom}" stroke="#2a261f"/>`;
    grid += `<text x="${xx.toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${day}</text>`;
  }
  const line = ordered.map((d) => `${x(d.day).toFixed(1)},${y(d[key]).toFixed(1)}`).join(" ");
  const dots = ordered.filter((d) => d.trades && d.trades.length).map((d) =>
    `<circle cx="${x(d.day).toFixed(1)}" cy="${y(d[key]).toFixed(1)}" r="3.2" fill="#f3efe4"/>`
  ).join("");
  const zero = opts.zero && low <= 0 && high >= 0
    ? `<line x1="${left}" y1="${y(0).toFixed(1)}" x2="${width - right}" y2="${y(0).toFixed(1)}" stroke="#8a8172" stroke-dasharray="3 4"/>`
    : "";
  return `<svg viewBox="0 0 ${width} ${height}">${grid}${zero}<polyline points="${line}" fill="none" stroke="#e6d3a3" stroke-width="1.8"/>${dots}</svg>`;
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
  const width = 1040, height = 280, left = 78, right = 16, top = 16, bottom = 32;
  const xs = rows.map((r) => r.ending_spot), ys = rows.map((r) => r.pnl);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys, 0), y1 = Math.max(...ys, 0);
  const sx = x1 - x0 || 1, sy = y1 - y0 || 1;
  x0 -= sx * 0.06; x1 += sx * 0.06; y0 -= sy * 0.06; y1 += sy * 0.06;
  const X = (v) => left + ((v - x0) / (x1 - x0)) * (width - left - right);
  const Y = (v) => top + (1 - (v - y0) / (y1 - y0)) * (height - top - bottom);
  let grid = "";
  for (let i = 0; i < 5; i++) {
    const value = y0 + ((y1 - y0) * i) / 4;
    const yy = Y(value);
    grid += `<line x1="${left}" y1="${yy.toFixed(1)}" x2="${width - right}" y2="${yy.toFixed(1)}" stroke="#3a342b"/>`;
    grid += `<text x="${left - 8}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${money(value)}</text>`;
  }
  const dots = rows.map((r) =>
    `<a href="#/run/${state.run}/path/${r.path}"><circle cx="${X(r.ending_spot).toFixed(1)}" cy="${Y(r.pnl).toFixed(1)}" r="4.2" fill="${r.pnl < 0 ? "#d07a62" : "#e6d3a3"}"><title>Path ${r.path} ${money(r.pnl)}</title></circle></a>`
  ).join("");
  const zero = y0 <= 0 && y1 >= 0
    ? `<line x1="${left}" y1="${Y(0).toFixed(1)}" x2="${width - right}" y2="${Y(0).toFixed(1)}" stroke="#8a8172" stroke-dasharray="3 4"/>`
    : "";
  return `<svg viewBox="0 0 ${width} ${height}">${grid}${zero}${dots}</svg>`;
}

function histogram(rows) {
  const values = rows.map((r) => r.pnl);
  const bins = 16;
  const low = Math.min(...values), high = Math.max(...values);
  const span = high - low || 1;
  const counts = Array(bins).fill(0);
  values.forEach((v) => {
    const i = Math.min(bins - 1, Math.floor(((v - low) / span) * bins));
    counts[i] += 1;
  });
  const max = Math.max(...counts, 1);
  const width = 1040, height = 200, left = 78, right = 16, top = 12, bottom = 28;
  const bw = (width - left - right) / bins;
  let bars = "";
  counts.forEach((c, i) => {
    const h = (c / max) * (height - top - bottom);
    const x = left + i * bw;
    const y = height - bottom - h;
    const mid = low + ((i + 0.5) / bins) * span;
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw - 3).toFixed(1)}" height="${h.toFixed(1)}" fill="${mid < 0 ? "#d07a62" : "#e6d3a3"}" opacity="0.85"><title>${money(mid)} · ${c}</title></rect>`;
  });
  return `<svg viewBox="0 0 ${width} ${height}">${bars}<text x="${left}" y="${height - 8}" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${money(low)}</text><text x="${width - right}" y="${height - 8}" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace,monospace">${money(high)}</text></svg>`;
}

const state = { run: null, path: null, day: null };

function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  crumb.textContent = "";
  if (parts[0] === "run" && parts[2] === "path") return showPath(parts[1], Number(parts[3]));
  if (parts[0] === "run") return showRun(parts[1]);
  return showHome();
}

async function showHome() {
  crumb.textContent = "Simulations";
  app.innerHTML = "<h1>Simulations</h1><p class='lede'>Each card is a saved Monte Carlo run. Open one to see the book, the path cloud, and every path.</p>";
  const runs = await get("/api/runs");
  if (!runs.length) {
    app.insertAdjacentHTML("beforeend", "<p class='empty'>No runs yet. Run the desk, then come back.</p>");
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
  const [run, paths] = await Promise.all([get(`/api/runs/${id}`), get(`/api/runs/${id}/paths`)]);
  const mc = run.monte_carlo || {};
  const book = run.book || {};
  const cfg = run.config || {};
  app.innerHTML = "";
  const h = document.createElement("h1");
  h.textContent = `${cfg.strategy || "wheel"} · ${cfg.lots || ""} lots`;
  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = `${when(run.created)} · seed ${cfg.seed} · ${mc.paths} paths · horizon ${cfg.horizon_days}d. Click a path, or a point on the cloud.`;
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

  const cloudTitle = document.createElement("h2");
  cloudTitle.textContent = "Where the paths landed";
  const cloudNote = document.createElement("p");
  cloudNote.className = "note";
  cloudNote.textContent = "Each point is a path. Horizontal is the ending spot, vertical is the P&L. Gold finished up, brick finished down. Click a point.";
  const cloud = document.createElement("div");
  cloud.innerHTML = scatter(paths);
  const histTitle = document.createElement("h2");
  histTitle.textContent = "P&L distribution";
  const hist = document.createElement("div");
  hist.innerHTML = histogram(paths);
  app.append(cloudTitle, cloudNote, cloud, histTitle, hist);

  if (run.ribbon && run.ribbon.length) {
    const band = document.createElement("h2");
    band.textContent = "Spot band across every path";
    const svg = document.createElement("div");
    svg.innerHTML = ribbonChart(run.ribbon);
    app.append(band, svg);
  }

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
  charts.className = "charts";
  charts.innerHTML = `
    <div><h3>Spot. Dots are trade days.</h3>${chart(days, "price")}</div>
    <div><h3>P&L. Dashed line is zero.</h3>${chart(days, "pnl", { zero: true, money: true })}</div>
    <div><h3>Cover. Shares ÷ 100 against the long puts.</h3>${linesChart(days, [
      { key: "shares", color: "#e6d3a3", label: "shares ÷ 100" },
      { key: "puts", color: "#7d9a84", label: "long puts" },
    ])}</div>
    <div><h3>Short options still open.</h3>${linesChart(days, [
      { key: "shortCalls", color: "#e6d3a3", label: "short calls" },
      { key: "shortPuts", color: "#d07a62", label: "short puts" },
    ])}</div>`;
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
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "MTM is the value of that line. The book row is the sum, which is the P&L.";
    bookHost.append(title, note, positionTable(selected.position));
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
