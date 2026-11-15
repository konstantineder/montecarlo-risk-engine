---
name: tqqq-book
description: Price and simulate the TQQQ protective-put book. Use when the user asks for the path desk, put DTE, horizon, lots, weekly call OTM, or the Monte Carlo ribbon.
---

# TQQQ book

Call the local `tqqq-book` MCP. Do not invent premiums. Nothing is ordered.

- `tqqq_book` prices the opening puts and calls. Use this when the user wants the premiums, deployed cash, floor, or the gap versus the old fill.
- `tqqq_monte_carlo` runs the paths. The `ribbon` array is the chart: one row per day, with spot and P&L at the 5th, 50th, and 95th percentile.
- `tqqq_report` does the same run and writes `runs/tqqq-desk.html`.

`putDte` is the life of the protective put. `horizonDays` is how long the path is. They are not the same number. If the horizon is shorter, the put is marked with the days that remain. It does not expire on the horizon.

Defaults: 100 lots, spot 59.22, put DTE 180, horizon 100, IV 0.71, vol 0.66, QQQ drift 0.12, weekly calls 12/12/12/10 contracts at 3/5/8/10 percent OTM, strategy wheel, 400 paths, seed 13.

A premium is this repo's analytic Black-Scholes price. It is not the IBKR fill.
