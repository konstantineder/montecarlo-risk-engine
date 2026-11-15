---
name: tqqq-book
description: Price and simulate the TQQQ protective-put book. Use when the user asks for the path desk, put DTE, horizon, lots, weekly call OTM, or the Monte Carlo ribbon.
---

# TQQQ book

Call the local `tqqq-book` MCP. Do not invent premiums. Nothing is ordered.

- `tqqq_book` prices the opening puts and calls. Use this when the user wants the premiums, deployed cash, floor, or the gap versus the old fill.
- `tqqq_monte_carlo` runs the paths and saves the run. The median path has every day: price, P&L, MTM, IV, shares, and the trades that day. Every path is in the saved `paths.jsonl`.
- `tqqq_report` does the same run and writes `desk.html` in that run folder.
- `tqqq_runs` lists saved runs. `tqqq_run` reads one. Pass `path` to see that path's days.

`initialDte` is the life of the protective puts you buy. `putDte` is the same knob. `horizonDays` is how long the path is. They are not the same number. If the horizon is shorter, the put is marked with the days that remain. It does not expire on the horizon.

Realized vol does not drift. It stays at `vol` (default 0.66). The daily shock is symmetric, so price goes up and down. Jumps are biased down. Implied vol rises when spot falls and falls when spot rises.

Defaults: 100 lots, spot 59.22, put DTE 180, horizon 100, IV 0.71, vol 0.66, QQQ drift 0.12, weekly calls 12/12/12/10 contracts at 3/5/8/10 percent OTM, strategy wheel, 400 paths, seed 13.

A premium is this repo's analytic Black-Scholes price. It is not the IBKR fill.
