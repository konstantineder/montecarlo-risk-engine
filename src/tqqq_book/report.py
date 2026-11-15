"""One-page desk report. No network. The chart data is the Monte Carlo ribbon."""

from __future__ import annotations

import json
from pathlib import Path


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _px(value: float) -> str:
    return f"{value:,.2f}"


def _trade_text(trades: list[dict]) -> str:
    bits = []
    for trade in trades:
        premium = trade.get("premium")
        price = f" @ {premium:.2f}" if isinstance(premium, (int, float)) else ""
        bits.append(f"{trade['action']} {trade.get('qty', '')}x {trade.get('strike', '')}{price}")
    return "; ".join(bits)


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


def write_report(result: dict, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    book = result["book"]
    mc = result["monte_carlo"]
    ribbon = result["ribbon"]
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
    vol = result.get("vol") or {}
    opening = "".join(
        f"<tr><td>{trade['action']}</td><td>{trade['qty']}</td><td>{trade['strike']:.0f}</td><td>{trade['expiry']}</td><td>{trade['premium']:.2f}</td><td>{trade['tag']}</td></tr>"
        for trade in book.get("opening_trades", [])
    )
    median = result.get("median_path") or {"path": "", "days": []}
    day_rows = "".join(
        "<tr>"
        f"<td>{day['day']}</td><td>{_px(day['price'])}</td><td>{_money(day['pnl'])}</td>"
        f"<td>{_money(day['mtm'])}</td><td>{day['iv']:.2f}</td><td>{day['shares']}</td>"
        f"<td class=\"trades\">{_trade_text(day['trades'])}</td></tr>"
        for day in median.get("days", [])
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TQQQ book · {cfg['put_dte']} DTE · {cfg['path_days']} day path</title>
<style>
  body {{ margin: 0; background: #14120e; color: #f3efe4; font: 15px/1.45 Georgia, serif; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-weight: 500; font-size: 36px; margin: 0 0 8px; }}
  p {{ color: #b7ae9d; }}
  .facts {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #3a342b; border-radius: 12px; overflow: hidden; }}
  .fact {{ background: #1c1915; padding: 14px 16px; }}
  .fact span {{ display: block; color: #9c9486; font: 11px/1.2 ui-monospace, monospace; letter-spacing: 0.04em; text-transform: uppercase; }}
  .fact strong {{ display: block; margin-top: 6px; font: 20px/1.2 ui-monospace, monospace; }}
  section {{ margin-top: 28px; }}
  h2 {{ font-size: 18px; font-weight: 500; margin: 0 0 8px; }}
  table {{ width: 100%; border-collapse: collapse; font: 13px/1.4 ui-monospace, monospace; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #3a342b; }}
  th {{ color: #9c9486; font-weight: 500; }}
  svg {{ width: 100%; height: auto; background: #1c1915; border-radius: 12px; }}
  .days {{ max-height: 540px; overflow: auto; border: 1px solid #3a342b; border-radius: 12px; }}
  td.trades {{ color: #e6d3a3; }}
</style>
</head>
<body>
<main>
  <p>TQQQ · {cfg['lots']} lots · model marks, not an order</p>
  <h1>Path desk</h1>
  <p>{result['note']}</p>
  <p>{vol.get('note', '')}</p>
  <div class="facts">{cards}</div>
  <section>
    <h2>Spot, 5th to 95th percentile</h2>
    {_band(ribbon, "spot_p05", "spot_p50", "spot_p95")}
  </section>
  <section>
    <h2>P&L, 5th to 95th percentile</h2>
    {_band(ribbon, "pnl_p05", "pnl_p50", "pnl_p95")}
  </section>
  <section>
    <h2>MTM, 5th to 95th percentile</h2>
    {_band(ribbon, "mtm_p05", "mtm_p50", "mtm_p95")}
  </section>
  <section>
    <h2>Opening trades</h2>
    <table>
      <thead><tr><th>Action</th><th>Qty</th><th>Strike</th><th>Expiry</th><th>Premium</th><th>Tag</th></tr></thead>
      <tbody>{opening}</tbody>
    </table>
  </section>
  <section>
    <h2>Median path {median.get('path', '')}, each day</h2>
    <div class="days">
    <table>
      <thead><tr><th>Day</th><th>Price</th><th>P&L</th><th>MTM</th><th>IV</th><th>Shares</th><th>Trades</th></tr></thead>
      <tbody>{day_rows}</tbody>
    </table>
    </div>
  </section>
  <section>
    <h2>Opening prices used</h2>
    <table>
      <thead><tr><th>Leg</th><th>Qty</th><th>Strike</th><th>DTE</th><th>Left at horizon</th><th>Premium</th><th>Priced by</th></tr></thead>
      <tbody>{legs}</tbody>
    </table>
  </section>
</main>
<script type="application/json" id="run">{json.dumps({"config": cfg, "book": book, "monte_carlo": mc})}</script>
</body>
</html>
"""
    target.write_text(page)
    return str(target.resolve())
