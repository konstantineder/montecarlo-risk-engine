"""One-page desk report. No network. The grids are filled from the saved paths."""

from __future__ import annotations

import json
from pathlib import Path


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _band(rows: list[dict], low: str, mid: str, high: str, width: int = 760, height: int = 240) -> str:
    if not rows:
        return ""
    xs = [row["day"] for row in rows]
    ys = [row[low] for row in rows] + [row[high] for row in rows]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    pad = 28

    def px(day: float) -> float:
        return pad + (day - x0) / (x1 - x0) * (width - pad * 2)

    def py(value: float) -> float:
        return height - pad - (value - y0) / (y1 - y0) * (height - pad * 2)

    upper = " ".join(f"{px(row['day']):.1f},{py(row[high]):.1f}" for row in rows)
    lower = " ".join(f"{px(row['day']):.1f},{py(row[low]):.1f}" for row in reversed(rows))
    middle = " ".join(f"{px(row['day']):.1f},{py(row[mid]):.1f}" for row in rows)
    zero = py(0.0) if y0 <= 0 <= y1 else None
    zero_line = f'<line x1="{pad}" y1="{zero:.1f}" x2="{width - pad}" y2="{zero:.1f}" stroke="#5c5648" stroke-dasharray="3 4"/>' if zero is not None else ""
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<polygon points="{upper} {lower}" fill="#c4a574" opacity="0.18"/>'
        f'<polyline points="{middle}" fill="none" stroke="#e6d3a3" stroke-width="1.6"/>'
        f"{zero_line}</svg>"
    )


def _load_paths(result: dict) -> list[dict]:
    saved = result.get("saved") or {}
    source = saved.get("paths")
    if source and Path(source).exists():
        return [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    median = result.get("median_path")
    if isinstance(median, dict) and median.get("days"):
        return [median]
    return []


def _slim(paths: list[dict]) -> list[dict]:
    kept = []
    for row in paths:
        last = row["days"][-1] if row.get("days") else {}
        kept.append({
            "path": row["path"],
            "pnl": row["pnl"],
            "ending_spot": row["ending_spot"],
            "min_spot": row["min_spot"],
            "max_dd": row["max_dd"],
            "ending_shares": row["ending_shares"],
            "assignments": row["assignments"],
            "new_calls": row["new_calls"],
            "roll_cash": row["roll_cash"],
            "mtm": last.get("mtm", 0.0),
            "days": row["days"],
        })
    return kept


def write_report(result: dict, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    book = result["book"]
    mc = result["monte_carlo"]
    ribbon = result["ribbon"]
    vol = result.get("vol") or {}
    paths = _slim(_load_paths(result))
    selected = (result.get("median_path") or {}).get("path", 0)
    legs = "".join(
        "<tr>"
        f"<td>{leg['side']} {leg.get('tag')}</td>"
        f"<td>{leg['qty']}</td><td>{leg['strike']:.0f}</td><td>{leg['dte']}</td>"
        f"<td>{leg['days_left_at_horizon']}</td><td>{leg['premium']:.2f}</td>"
        f"<td>{leg['priced_by']}</td></tr>"
        for leg in [*book["puts"], *book["calls"]]
    )
    facts = [
        ("Deployed", _money(book["deployed"])),
        ("Put premium", _money(book["put_premium"])),
        ("Call premium", _money(book["call_premium"])),
        ("Floor", _money(book["floor"])),
        ("BS vs fill", _money(book["model_gap_vs_fill"])),
        ("Initial put DTE", str(cfg.get("initial_dte", cfg["put_dte"]))),
        ("Horizon", f"{cfg['path_days']} of {cfg['horizon_days']}"),
        ("Put days left", str(cfg["put_days_left"])),
        ("Median P&L", _money(mc["median"])),
        ("5th pct", _money(mc["p05"])),
        ("95th pct", _money(mc["p95"])),
        ("Chance of a loss", f"{mc['prob_loss']:.0%}"),
        ("Median drawdown", _money(mc["median_drawdown"])),
    ]
    cards = "".join(f'<div class="fact"><span>{label}</span><strong>{value}</strong></div>' for label, value in facts)
    opening = "".join(
        f"<tr><td>{trade['action']}</td><td>{trade['qty']}</td><td>{trade['strike']:.0f}</td><td>{trade['expiry']}</td><td>{trade['premium']:.2f}</td><td>{trade['tag']}</td></tr>"
        for trade in book.get("opening_trades", [])
    )
    rule_rows = []
    for rule in result.get("rules") or [{"name": "hold", "says": "Keep the opening book. No new orders.",}]:
        knobs = " ".join(f"{key} {value}" for key, value in rule.items() if key not in ("name", "says"))
        rule_rows.append(f"<tr><td>{rule['name']}</td><td>{rule['says']}</td><td>{knobs}</td></tr>")
    payload = json.dumps({"paths": paths, "selected": selected}, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TQQQ book · __DTE__ DTE · __DAYS__ day path</title>
<style>
  body { margin: 0; background: #14120e; color: #f3efe4; font: 15px/1.45 Georgia, serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-weight: 500; font-size: 36px; margin: 0 0 8px; }
  p { color: #b7ae9d; }
  .facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #3a342b; border-radius: 12px; overflow: hidden; }
  .fact { background: #1c1915; padding: 14px 16px; }
  .fact span { display: block; color: #9c9486; font: 11px/1.2 ui-monospace, monospace; letter-spacing: 0.04em; text-transform: uppercase; }
  .fact strong { display: block; margin-top: 6px; font: 20px/1.2 ui-monospace, monospace; }
  section { margin-top: 28px; }
  h2 { font-size: 18px; font-weight: 500; margin: 0 0 8px; }
  table { width: 100%; border-collapse: collapse; font: 13px/1.4 ui-monospace, monospace; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #3a342b; white-space: nowrap; }
  th { color: #9c9486; font-weight: 500; }
  svg { width: 100%; height: auto; background: #1c1915; border-radius: 12px; }
  #charts { display: grid; gap: 14px; margin: 14px 0; }
  #charts h3 { margin: 0 0 6px; font-size: 15px; font-weight: 500; }
  .grid { max-height: 440px; overflow: auto; border: 1px solid #3a342b; border-radius: 12px; }
  .grid th { position: sticky; top: 0; background: #221e19; cursor: pointer; z-index: 1; }
  .grid tbody tr { cursor: pointer; }
  .grid tbody tr:hover { background: #241f19; }
  .grid tr.on { background: #2c261e; }
  td.trades { color: #e6d3a3; white-space: normal; }
  .detail { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #3a342b; border-radius: 12px; overflow: hidden; margin: 12px 0; }
</style>
</head>
<body>
<main>
  <p>TQQQ · __LOTS__ lots · model marks, not an order</p>
  <h1>Path desk</h1>
  <p>__NOTE__</p>
  <p>__VOL__</p>
  <div class="facts">__CARDS__</div>
  <section>
    <h2>Rules</h2>
    <p>strategy.apply walks these in order. A trade on a path names the rule that made it.</p>
    <table>
      <thead><tr><th>Rule</th><th>What it decides</th><th>Knobs</th></tr></thead>
      <tbody>__RULES__</tbody>
    </table>
  </section>
  <section>
    <h2>Spot, 5th to 95th percentile</h2>
    __SPOT__
  </section>
  <section>
    <h2>P&L, 5th to 95th percentile</h2>
    __PNL__
  </section>
  <section>
    <h2>MTM, 5th to 95th percentile</h2>
    __MTM__
  </section>
  <section>
    <h2>Paths</h2>
    <p id="path-sort">Click Index, P&L, or Spot to sort. Click a row for every day of that path.</p>
    <div class="grid">
      <table>
        <thead>
          <tr>
            <th data-grid="paths" data-key="path">Index</th>
            <th data-grid="paths" data-key="ending_spot">Spot</th>
            <th data-grid="paths" data-key="pnl">P&L</th>
            <th data-grid="paths" data-key="mtm">MTM</th>
            <th data-grid="paths" data-key="min_spot">Min spot</th>
            <th data-grid="paths" data-key="max_dd">Drawdown</th>
            <th data-grid="paths" data-key="trades">Trades</th>
          </tr>
        </thead>
        <tbody id="paths"></tbody>
      </table>
    </div>
  </section>
  <section>
    <h2 id="detail-title">Path</h2>
    <div class="detail" id="detail-facts"></div>
    <div id="charts"></div>
    <h2>Trades on this path</h2>
    <div class="grid">
      <table>
        <thead><tr><th>Day</th><th>Spot</th><th>P&L</th><th>Trades</th></tr></thead>
        <tbody id="trades"></tbody>
      </table>
    </div>
    <p id="day-sort">Every day, from day 0 to the end. Sort by day, spot, or P&L.</p>
    <div class="grid">
      <table>
        <thead>
          <tr>
            <th data-grid="days" data-key="day">Day</th>
            <th data-grid="days" data-key="price">Spot</th>
            <th data-grid="days" data-key="pnl">P&L</th>
            <th data-grid="days" data-key="mtm">MTM</th>
            <th data-grid="days" data-key="iv">IV</th>
            <th data-grid="days" data-key="shares">Shares</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody id="days"></tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Opening trades</h2>
    <table>
      <thead><tr><th>Action</th><th>Qty</th><th>Strike</th><th>Expiry</th><th>Premium</th><th>Tag</th></tr></thead>
      <tbody>__OPENING__</tbody>
    </table>
  </section>
  <section>
    <h2>Opening prices used</h2>
    <table>
      <thead><tr><th>Leg</th><th>Qty</th><th>Strike</th><th>DTE</th><th>Left at horizon</th><th>Premium</th><th>Priced by</th></tr></thead>
      <tbody>__LEGS__</tbody>
    </table>
  </section>
</main>
<script type="application/json" id="data">__JSON__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const paths = data.paths;
let pathKey = "path";
let pathDir = 1;
let dayKey = "day";
let dayDir = 1;
let current = paths.find((row) => row.path === data.selected) || paths[0];

function money(value) {
  const sign = value < 0 ? "-" : "";
  return sign + "$" + Math.abs(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
}
function px(value) {
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function tradeCount(row) {
  return row.days.reduce((sum, day) => sum + day.trades.length, 0);
}
function tradeText(trades) {
  return trades.map((trade) => {
    const bits = [trade.rule, trade.action, (trade.qty ?? "") + "x", String(trade.strike ?? "")].filter((part) => part !== undefined && part !== "");
    if (typeof trade.premium === "number") bits.push("@" + trade.premium.toFixed(2));
    if (trade.tag) bits.push(String(trade.tag));
    if (trade.expiry != null) bits.push("exp " + trade.expiry);
    if (typeof trade.intrinsic === "number" && trade.intrinsic > 0) bits.push("intrinsic " + trade.intrinsic.toFixed(2));
    if (trade.reason) bits.push(trade.reason);
    return bits.join(" ");
  }).join("; ");
}
function cell(text, className) {
  const node = document.createElement("td");
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function mark(grid, key, dir) {
  document.querySelectorAll('th[data-grid="' + grid + '"]').forEach((th) => {
    const name = th.dataset.label || th.textContent.replace(/[↑↓]$/, "");
    th.dataset.label = name;
    th.textContent = th.dataset.key === key ? name + (dir > 0 ? " ↑" : " ↓") : name;
  });
}
function renderPaths() {
  const body = document.getElementById("paths");
  body.replaceChildren();
  const rows = paths.slice().sort((a, b) => {
    const left = pathKey === "trades" ? tradeCount(a) : a[pathKey];
    const right = pathKey === "trades" ? tradeCount(b) : b[pathKey];
    return (left - right) * pathDir;
  });
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (current && row.path === current.path) tr.className = "on";
    tr.append(
      cell(String(row.path)),
      cell(px(row.ending_spot)),
      cell(money(row.pnl)),
      cell(money(row.mtm)),
      cell(px(row.min_spot)),
      cell(money(row.max_dd)),
      cell(String(tradeCount(row)))
    );
    tr.addEventListener("click", () => {
      current = row;
      dayKey = "day";
      dayDir = 1;
      document.querySelectorAll("#paths tr").forEach((el) => el.classList.remove("on"));
      tr.classList.add("on");
      renderDays();
    });
    body.append(tr);
  });
  mark("paths", pathKey, pathDir);
  document.getElementById("path-sort").textContent =
    paths.length + " paths. Sorted by " + pathKey + (pathDir > 0 ? ", low to high." : ", high to low.") + " Click a row for every day.";
}
function axisMoney(value) {
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  if (abs >= 1000) return sign + "$" + Math.round(abs).toLocaleString("en-US");
  return sign + "$" + abs.toFixed(0);
}
function axisSpot(value) {
  return value.toFixed(1);
}
function chart(days, key) {
  const ordered = days.slice().sort((a, b) => a.day - b.day);
  const width = 1040;
  const height = 280;
  const left = 78;
  const right = 16;
  const top = 16;
  const bottom = 32;
  const values = ordered.map((day) => day[key]);
  let low = Math.min.apply(null, values);
  let high = Math.max.apply(null, values);
  if (key === "pnl") {
    low = Math.min(low, 0);
    high = Math.max(high, 0);
  }
  const span = high - low || 1;
  low -= span * 0.06;
  high += span * 0.06;
  const lastDay = ordered[ordered.length - 1].day || 1;
  const x = (day) => left + (day / lastDay) * (width - left - right);
  const y = (value) => top + (1 - (value - low) / (high - low)) * (height - top - bottom);
  const label = key === "pnl" ? axisMoney : axisSpot;
  const yTicks = 5;
  const xTicks = Math.min(6, lastDay);
  let grid = "";
  for (let i = 0; i < yTicks; i += 1) {
    const value = low + ((high - low) * i) / (yTicks - 1);
    const yy = y(value);
    grid += '<line x1="' + left + '" y1="' + yy.toFixed(1) + '" x2="' + (width - right) + '" y2="' + yy.toFixed(1) + '" stroke="#3a342b"/>';
    grid += '<text x="' + (left - 8) + '" y="' + (yy + 4).toFixed(1) + '" text-anchor="end" fill="#9c9486" font-size="11" font-family="ui-monospace, monospace">' + label(value) + "</text>";
  }
  for (let i = 0; i < xTicks; i += 1) {
    const day = Math.round((lastDay * i) / (xTicks - 1));
    const xx = x(day);
    grid += '<line x1="' + xx.toFixed(1) + '" y1="' + top + '" x2="' + xx.toFixed(1) + '" y2="' + (height - bottom) + '" stroke="#2a261f"/>';
    grid += '<text x="' + xx.toFixed(1) + '" y="' + (height - 10) + '" text-anchor="middle" fill="#9c9486" font-size="11" font-family="ui-monospace, monospace">' + day + "</text>";
  }
  const line = ordered.map((day) => x(day.day).toFixed(1) + "," + y(day[key]).toFixed(1)).join(" ");
  const dots = ordered.filter((day) => day.trades.length).map((day) =>
    '<circle cx="' + x(day.day).toFixed(1) + '" cy="' + y(day[key]).toFixed(1) + '" r="3.4" fill="#f3efe4" stroke="#14120e" stroke-width="1"><title>Day ' + day.day + " " + label(day[key]) + "</title></circle>"
  ).join("");
  const zero = key === "pnl" && low <= 0 && high >= 0
    ? '<line x1="' + left + '" y1="' + y(0).toFixed(1) + '" x2="' + (width - right) + '" y2="' + y(0).toFixed(1) + '" stroke="#8a8172" stroke-dasharray="3 4"/>'
    : "";
  return '<svg viewBox="0 0 ' + width + " " + height + '" role="img">' + grid + zero +
    '<polyline points="' + line + '" fill="none" stroke="#e6d3a3" stroke-width="1.8"/>' + dots + "</svg>";
}
function renderDays() {
  const title = document.getElementById("detail-title");
  const facts = document.getElementById("detail-facts");
  const charts = document.getElementById("charts");
  const trades = document.getElementById("trades");
  const body = document.getElementById("days");
  body.replaceChildren();
  facts.replaceChildren();
  trades.replaceChildren();
  charts.replaceChildren();
  if (!current) {
    title.textContent = "Path";
    return;
  }
  title.textContent = "Path " + current.path;
  const spotCard = document.createElement("div");
  const pnlCard = document.createElement("div");
  spotCard.innerHTML = "<h3>Spot. Left is the price, bottom is the day. Dots are trades.</h3>" + chart(current.days, "price");
  pnlCard.innerHTML = "<h3>P&L. Left is dollars, bottom is the same day. Dashed line is zero.</h3>" + chart(current.days, "pnl");
  charts.append(spotCard, pnlCard);
  [
    ["Index", String(current.path)],
    ["End spot", px(current.ending_spot)],
    ["P&L", money(current.pnl)],
    ["MTM", money(current.mtm)],
    ["Min spot", px(current.min_spot)],
    ["Drawdown", money(current.max_dd)],
    ["Shares", String(current.ending_shares)],
    ["Assignments", String(current.assignments)],
    ["New calls", String(current.new_calls)],
    ["Roll cash", money(current.roll_cash)],
    ["Trades", String(tradeCount(current))],
    ["Days", String(current.days.length)]
  ].forEach((pair) => {
    const fact = document.createElement("div");
    fact.className = "fact";
    const span = document.createElement("span");
    span.textContent = pair[0];
    const strong = document.createElement("strong");
    strong.textContent = pair[1];
    fact.append(span, strong);
    facts.append(fact);
  });
  const ordered = current.days.slice().sort((a, b) => a.day - b.day);
  ordered.filter((day) => day.trades.length).forEach((day) => {
    const tr = document.createElement("tr");
    tr.append(cell(String(day.day)), cell(px(day.price)), cell(money(day.pnl)), cell(tradeText(day.trades), "trades"));
    trades.append(tr);
  });
  const days = current.days.slice().sort((a, b) => (a[dayKey] - b[dayKey]) * dayDir);
  days.forEach((day) => {
    const tr = document.createElement("tr");
    tr.append(
      cell(String(day.day)),
      cell(px(day.price)),
      cell(money(day.pnl)),
      cell(money(day.mtm)),
      cell(Number(day.iv).toFixed(2)),
      cell(String(day.shares)),
      cell(tradeText(day.trades), "trades")
    );
    body.append(tr);
  });
  mark("days", dayKey, dayDir);
}
document.querySelectorAll("th[data-key]").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (th.dataset.grid === "paths") {
      pathDir = pathKey === key ? pathDir * -1 : (key === "path" ? 1 : -1);
      pathKey = key;
      renderPaths();
    } else {
      dayDir = dayKey === key ? dayDir * -1 : (key === "day" ? 1 : -1);
      dayKey = key;
      renderDays();
    }
  });
});
renderPaths();
renderDays();
</script>
</body>
</html>
"""
    replacements = {
        "__DTE__": str(cfg["put_dte"]),
        "__DAYS__": str(cfg["path_days"]),
        "__LOTS__": str(cfg["lots"]),
        "__NOTE__": result["note"],
        "__VOL__": vol.get("note", ""),
        "__CARDS__": cards,
        "__SPOT__": _band(ribbon, "spot_p05", "spot_p50", "spot_p95"),
        "__PNL__": _band(ribbon, "pnl_p05", "pnl_p50", "pnl_p95"),
        "__MTM__": _band(ribbon, "mtm_p05", "mtm_p50", "mtm_p95"),
        "__OPENING__": opening,
        "__RULES__": "".join(rule_rows),
        "__LEGS__": legs,
        "__JSON__": payload,
    }
    for token, value in replacements.items():
        page = page.replace(token, value)
    target.write_text(page)
    return str(target.resolve())
