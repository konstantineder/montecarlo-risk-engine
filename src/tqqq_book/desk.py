"""Run the 13 Nov TQQQ book with the knobs from the path desk.

Lots, protective-put DTE, and the 1w/2w/3w/4w call OTM ladder are inputs.
Everything else has the desk default. Put and call premiums are this repo's
Black-Scholes price at that DTE and IV. They are not snapped to the IBKR fill.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from models.black_scholes import BlackScholesModel
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType

DT = 1.0 / 252.0
FEE = 0.0088
CONTRACT = 100
REF_SPOT = 59.22
REF_LOTS = 100
FILL_PUT = 106_600.0
FILL_CALL = 27_780.0


@dataclass(frozen=True)
class Tenor:
    days: int
    otm: float
    contracts: int
    tag: str


@dataclass(frozen=True)
class BookConfig:
    lots: int = REF_LOTS
    spot: float = REF_SPOT
    qqq_mu: float = 0.12
    tqqq_vol: float = 0.66
    iv: float = 0.71
    iv_elasticity: float = 0.86
    rate: float = 0.04
    harvest_drop: float = 0.08
    roll_fraction: float = 0.5
    put_monetize_drop: float = 0.10
    new_put_otm: float = 0.02
    write_floor_after_monetize: float = 0.90
    stagger_min_spot_frac: float = 0.95
    horizon_days: int = 100
    put_dte: int = 180
    paths: int = 400
    seed: int = 13
    strategy: str = "wheel"
    slippage: float = 0.05
    lambda_j: float = 0.7
    mu_j: float = -0.07
    sigma_j: float = 0.05
    tenors: tuple[Tenor, ...] = field(
        default_factory=lambda: (
            Tenor(5, 0.03, 12, "w1"),
            Tenor(10, 0.05, 12, "w2"),
            Tenor(15, 0.08, 12, "w3"),
            Tenor(20, 0.10, 10, "w4"),
        )
    )


@dataclass
class Leg:
    kind: str
    side: str
    strike: float
    qty: int
    expiry: int
    entry: float
    tag: str
    open: bool = True
    otm: float = 0.0
    tenor: int = 0


def _engine_price(spot: float, strike: float, t_years: float, rate: float, sigma: float, kind: str) -> float:
    """One price from the fork's analytic Black-Scholes, not a second formula."""
    if t_years <= 1e-8 or sigma <= 1e-8 or spot <= 0 or strike <= 0:
        if kind == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    model = BlackScholesModel(0.0, spot, rate, sigma, asset_id="TQQQ")
    option = EuropeanOption(
        underlying=Equity(asset_id="TQQQ"),
        exercise_date=t_years,
        strike=strike,
        option_type=OptionType.CALL if kind == "call" else OptionType.PUT,
        asset_id="TQQQ",
    )
    return float(option.compute_pv_analytically(model)[0])


def _fast_price(spot: float, strike: float, t_years: float, rate: float, sigma: float, kind: str) -> float:
    """Same Black-Scholes closed form, without building a tensor per mark."""
    if spot <= 0 or strike <= 0:
        return 0.0
    if t_years <= 1e-8 or sigma <= 1e-8:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    vol = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / vol
    d2 = d1 - vol
    discount = math.exp(-rate * t_years)

    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if kind == "call":
        return spot * ncdf(d1) - strike * discount * ncdf(d2)
    return strike * discount * ncdf(-d2) - spot * ncdf(-d1)


def _scaled_strike(strike: float, spot: float) -> float:
    return float(max(1, round(strike * spot / REF_SPOT)))


def _qty(contracts_at_100: int, lots: int) -> int:
    return int(max(0, round(contracts_at_100 * lots / REF_LOTS)))


def _iv_at(spot: float, spot0: float, iv0: float, elasticity: float) -> float:
    if spot <= 0:
        return min(1.6, iv0 * 3)
    raw = iv0 * (spot0 / spot) ** elasticity
    return min(1.6, max(0.2, raw))


def _floor(legs: list[Leg]) -> float:
    return sum(leg.qty * CONTRACT * leg.strike for leg in legs if leg.open and leg.side == "long" and leg.kind == "put")


def build_book(cfg: BookConfig) -> dict:
    if cfg.lots < 1:
        raise ValueError("lots must be at least 1")
    if cfg.put_dte < 1:
        raise ValueError("put DTE must be at least 1")
    puts = (
        (_qty(60, cfg.lots), cfg.put_dte, _scaled_strike(60.0, cfg.spot), "p60"),
        (_qty(40, cfg.lots), cfg.put_dte, _scaled_strike(65.0, cfg.spot), "p65"),
    )
    # Longer calls stay on the 13 Nov strikes. The four weekly lines are
    # struck from the OTM inputs, so those knobs change the opening credit.
    extra = (
        (_qty(12, cfg.lots), 25, _scaled_strike(68.0, cfg.spot), "w5", 0.0),
        (_qty(10, cfg.lots), 30, _scaled_strike(70.0, cfg.spot), "w6", 0.0),
        (_qty(10, cfg.lots), cfg.put_dte, _scaled_strike(65.0, cfg.spot), "mar65", 0.0),
        (_qty(10, cfg.lots), cfg.put_dte, _scaled_strike(68.0, cfg.spot), "mar68", 0.0),
    )
    weekly = []
    for tenor in cfg.tenors:
        strike = float(max(1, round(cfg.spot * (1.0 + tenor.otm))))
        weekly.append((_qty(tenor.contracts, cfg.lots), tenor.days, strike, tenor.tag, tenor.otm))

    legs: list[Leg] = []
    put_premium = 0.0
    call_premium = 0.0
    checked = False
    for qty, expiry, strike, tag in puts:
        if qty <= 0:
            continue
        t_years = expiry * DT
        entry = _engine_price(cfg.spot, strike, t_years, cfg.rate, cfg.iv, "put")
        if not checked:
            mirror = _fast_price(cfg.spot, strike, t_years, cfg.rate, cfg.iv, "put")
            if abs(entry - mirror) > 1e-6:
                raise RuntimeError(f"fast pricer drifted from the engine: {entry} vs {mirror}")
            checked = True
        put_premium += entry * qty * CONTRACT
        legs.append(Leg("put", "long", strike, qty, expiry, entry, tag, tenor=expiry))
    for qty, expiry, strike, tag, otm in [*weekly, *extra]:
        if qty <= 0 or expiry <= 0:
            continue
        t_years = expiry * DT
        entry = _engine_price(cfg.spot, strike, t_years, cfg.rate, cfg.iv, "call")
        call_premium += entry * qty * CONTRACT
        legs.append(Leg("call", "short", strike, qty, expiry, entry, tag, otm=otm or max(0.0, strike / cfg.spot - 1.0), tenor=expiry))

    shares = cfg.lots * CONTRACT
    deployed = shares * cfg.spot + put_premium - call_premium
    scale = cfg.lots / REF_LOTS
    model_gap = (put_premium - FILL_PUT * scale) - (call_premium - FILL_CALL * scale)
    return {
        "legs": legs,
        "shares": shares,
        "put_premium": put_premium,
        "call_premium": call_premium,
        "deployed": deployed,
        "floor": _floor(legs),
        "model_gap": model_gap,
        "cash": -deployed,
    }


def _paths(cfg: BookConfig, horizon: int) -> np.ndarray:
    """QQQ diffusion, then TQQQ's 3x daily reset, fee, and financing."""
    rng = np.random.default_rng(cfg.seed)
    sig = cfg.tqqq_vol / 3.0
    out = np.empty((cfg.paths, horizon + 1), dtype=np.float64)
    out[:, 0] = cfg.spot
    fee_daily = FEE * DT
    fin_daily = 2.0 * cfg.rate * DT
    sqrt_dt = math.sqrt(DT)
    for i in range(cfg.paths):
        spot = cfg.spot
        for _t in range(horizon):
            z = float(rng.standard_normal())
            simple = cfg.qqq_mu * DT + sig * sqrt_dt * z
            lam = cfg.lambda_j * DT
            jumps = 0 if lam <= 0 else int(rng.poisson(lam))
            log_jump = 0.0
            if jumps:
                log_jump = float(np.sum(cfg.mu_j + cfg.sigma_j * rng.standard_normal(jumps)))
            r_q = (1.0 + simple) * math.exp(log_jump) - 1.0
            r_q = max(r_q, -0.95)
            r_t = 3.0 * r_q - fee_daily - fin_daily
            r_t = max(r_t, -0.95)
            spot = max(0.5, spot * (1.0 + r_t))
            out[i, _t + 1] = spot
    return out


def _cover(legs: list[Leg]) -> int:
    return sum(leg.qty * CONTRACT for leg in legs if leg.open and leg.side == "short" and leg.kind == "call")


def _mark(leg: Leg, spot: float, t: int, rate: float, sigma: float) -> float:
    return _fast_price(spot, leg.strike, max(0, leg.expiry - t) * DT, rate, sigma, leg.kind)


def _simulate(path: np.ndarray, book: dict, cfg: BookConfig) -> dict:
    legs = [Leg(**{k: getattr(leg, k) for k in Leg.__dataclass_fields__}) for leg in book["legs"]]
    cash = book["cash"]
    shares = book["shares"]
    buybacks = 0
    new_calls = 0
    assignments = 0
    monetized = False
    roll_cash = 0.0
    min_spot = float(path[0])
    wheel: list[tuple[int, float, float, int]] = []
    write_floor = 52.0 * cfg.spot / REF_SPOT
    uncapped = int(round(1000 * cfg.lots / REF_LOTS))
    last = len(path) - 1
    peak = book["deployed"]
    max_dd = 0.0
    last_nav = 0.0
    navs: list[float] = []
    days: list[dict] = []
    day_trades: list[dict] = []
    strategy = cfg.strategy

    def settle(t: int, spot: float) -> None:
        nonlocal cash, shares, assignments
        for leg in legs:
            if not leg.open or leg.expiry != t:
                continue
            intrinsic = max(leg.strike - spot, 0.0) if leg.kind == "put" else max(spot - leg.strike, 0.0)
            qty_shares = leg.qty * CONTRACT
            action = "expire"
            if leg.side == "long":
                cash += qty_shares * intrinsic
                action = "exercise" if intrinsic > 0 else "expire"
            elif leg.kind == "call" and intrinsic > 0:
                deliver = min(shares, qty_shares)
                cash += deliver * leg.strike
                shares -= deliver
                shortfall = qty_shares - deliver
                if shortfall:
                    cash -= shortfall * (spot - leg.strike)
                assignments += 1
                action = "assign_call"
                delivered = int(deliver // CONTRACT)
                if delivered > 0:
                    wheel.append((delivered, leg.strike, leg.otm, leg.tenor or 10))
            elif leg.kind == "put" and intrinsic > 0:
                cash -= qty_shares * leg.strike
                shares += qty_shares
                action = "assign_put"
            day_trades.append({
                "action": action,
                "kind": leg.kind,
                "side": leg.side,
                "qty": leg.qty,
                "strike": leg.strike,
                "tag": leg.tag,
                "intrinsic": round(intrinsic, 4),
            })
            leg.open = False

    def sell_call(t: int, spot: float, sigma: float, qty: int, strike: float, expiry: int, tag: str, otm: float, tenor: int) -> None:
        nonlocal cash, new_calls
        if qty <= 0 or expiry <= t:
            return
        mark = _fast_price(spot, strike, (expiry - t) * DT, cfg.rate, sigma, "call")
        credit = max(0.0, mark - cfg.slippage)
        cash += qty * CONTRACT * credit
        legs.append(Leg("call", "short", strike, qty, expiry, credit, tag, otm=otm, tenor=tenor))
        new_calls += qty
        day_trades.append({
            "action": "sell_call",
            "qty": qty,
            "strike": strike,
            "expiry": expiry,
            "premium": round(credit, 4),
            "tag": tag,
        })

    def monetize(t: int, spot: float, sigma: float) -> None:
        nonlocal cash, monetized, roll_cash, write_floor
        if monetized:
            return
        closed: list[Leg] = []
        realized = 0.0
        for leg in list(legs):
            if not (leg.open and leg.side == "long" and leg.kind == "put"):
                continue
            mark = _mark(leg, spot, t, cfg.rate, sigma)
            premium = max(0.0, mark - cfg.slippage)
            proceeds = leg.qty * CONTRACT * premium
            cash += proceeds
            realized += proceeds
            leg.open = False
            closed.append(leg)
            day_trades.append({
                "action": "sell_put",
                "qty": leg.qty,
                "strike": leg.strike,
                "premium": round(premium, 4),
                "tag": leg.tag,
                "reason": "monetize",
            })
        monetized = True
        if not closed:
            return
        total_qty = sum(leg.qty for leg in closed)
        expiry = closed[0].expiry
        new_strike = float(max(1, round(spot * (1.0 - cfg.new_put_otm))))
        new_mark = _fast_price(spot, new_strike, max(0, expiry - t) * DT, cfg.rate, sigma, "put")
        premium = new_mark + cfg.slippage
        cost = total_qty * CONTRACT * premium
        cash -= cost
        legs.append(Leg("put", "long", new_strike, total_qty, expiry, premium, f"monetized{int(new_strike)}", otm=cfg.new_put_otm, tenor=expiry))
        day_trades.append({
            "action": "buy_put",
            "qty": total_qty,
            "strike": new_strike,
            "expiry": expiry,
            "premium": round(premium, 4),
            "tag": f"monetized{int(new_strike)}",
            "reason": "monetize",
        })
        roll_cash += realized - cost
        write_floor = min(write_floor, spot * cfg.write_floor_after_monetize)

    for t, spot in enumerate(path):
        spot = float(spot)
        day_trades = []
        min_spot = min(min_spot, spot)
        sigma = _iv_at(spot, cfg.spot, cfg.iv, cfg.iv_elasticity)
        settle(t, spot)
        if t < last and strategy == "wheel":
            if not monetized and cfg.put_monetize_drop > 0 and spot <= cfg.spot * (1.0 - cfg.put_monetize_drop):
                monetize(t, spot, sigma)
            due = t > 0 and t % 5 == 0 and spot >= min(write_floor, spot * cfg.stagger_min_spot_frac)
            if due:
                room = max(0, int((shares - _cover(legs) - uncapped) // CONTRACT))
                for tenor in cfg.tenors:
                    have = sum(leg.qty for leg in legs if leg.open and leg.side == "short" and leg.kind == "call" and leg.tag == tenor.tag)
                    need = min(max(0, _qty(tenor.contracts, cfg.lots) - have), room)
                    if need <= 0 or t + tenor.days >= len(path):
                        continue
                    strike = float(max(1, round(spot * (1.0 + tenor.otm))))
                    sell_call(t, spot, sigma, need, strike, t + tenor.days, tenor.tag, tenor.otm, tenor.days)
                    room -= need
                    if room <= 0:
                        break
            if wheel and t < last:
                protected = sum(leg.qty for leg in legs if leg.open and leg.kind == "put" and leg.side == "long")
                short_puts = sum(leg.qty for leg in legs if leg.open and leg.kind == "put" and leg.side == "short")
                room_puts = protected - int(shares // CONTRACT) - short_puts
                still: list[tuple[int, float, float, int]] = []
                for qty, call_strike, otm, tenor_days in wheel:
                    take = min(qty, max(0, room_puts))
                    if take <= 0:
                        still.append((qty, call_strike, otm, tenor_days))
                        continue
                    mirror = otm if otm > 0 else 0.05
                    strike = float(max(1, round(call_strike * (1.0 - mirror))))
                    if strike >= call_strike:
                        strike = max(1.0, call_strike - 1.0)
                    expiry = min(last, t + max(5, tenor_days))
                    if expiry <= t:
                        continue
                    mark = _fast_price(spot, strike, (expiry - t) * DT, cfg.rate, sigma, "put")
                    credit = max(0.0, mark - cfg.slippage)
                    cash += take * CONTRACT * credit
                    legs.append(Leg("put", "short", strike, take, expiry, credit, "reenter", otm=mirror, tenor=tenor_days))
                    day_trades.append({
                        "action": "sell_put",
                        "qty": take,
                        "strike": strike,
                        "expiry": expiry,
                        "premium": round(credit, 4),
                        "tag": "reenter",
                    })
                    room_puts -= take
                    if take < qty:
                        still.append((qty - take, call_strike, otm, tenor_days))
                wheel = still
        mtm = cash + shares * spot
        for leg in legs:
            if not leg.open:
                continue
            mark = _mark(leg, spot, t, cfg.rate, sigma)
            mtm += mark * leg.qty * CONTRACT if leg.side == "long" else -mark * leg.qty * CONTRACT
        last_nav = mtm
        navs.append(last_nav)
        wealth = book["deployed"] + last_nav
        peak = max(peak, wealth)
        max_dd = max(max_dd, peak - wealth)
        days.append({
            "day": t,
            "price": round(spot, 4),
            "pnl": round(last_nav, 2),
            "mtm": round(wealth, 2),
            "iv": round(sigma, 4),
            "shares": shares,
            "trades": day_trades,
        })

    return {
        "pnl": last_nav,
        "max_dd": max_dd,
        "ending_shares": shares,
        "ending_spot": float(path[-1]),
        "min_spot": min_spot,
        "assignments": assignments,
        "new_calls": new_calls,
        "buybacks": buybacks,
        "roll_cash": roll_cash,
        "floor_after": _floor(legs),
        "nav": navs,
        "days": days,
    }


def _summary(name: str, rows: list[dict]) -> dict:
    pnl = np.array([row["pnl"] for row in rows], dtype=float)
    dd = np.array([row["max_dd"] for row in rows], dtype=float)
    return {
        "strategy": name,
        "paths": int(pnl.size),
        "mean": float(pnl.mean()),
        "median": float(np.median(pnl)),
        "p05": float(np.quantile(pnl, 0.05)),
        "p25": float(np.quantile(pnl, 0.25)),
        "p75": float(np.quantile(pnl, 0.75)),
        "p95": float(np.quantile(pnl, 0.95)),
        "prob_profit": float(np.mean(pnl > 0)),
        "prob_loss": float(np.mean(pnl < 0)),
        "median_drawdown": float(np.median(dd)),
        "mean_assignments": float(np.mean([row["assignments"] for row in rows])),
        "mean_new_calls": float(np.mean([row["new_calls"] for row in rows])),
        "mean_roll_cash": float(np.mean([row["roll_cash"] for row in rows])),
    }


def _ribbon(spots: np.ndarray, navs: np.ndarray, trade_counts: np.ndarray) -> list[dict]:
    rows = []
    for day in range(spots.shape[1]):
        spot = spots[:, day]
        pnl = navs[:, day]
        trades = trade_counts[:, day]
        rows.append({
            "day": day,
            "price_p05": float(np.quantile(spot, 0.05)),
            "price_p50": float(np.quantile(spot, 0.50)),
            "price_p95": float(np.quantile(spot, 0.95)),
            "spot_p05": float(np.quantile(spot, 0.05)),
            "spot_p50": float(np.quantile(spot, 0.50)),
            "spot_p95": float(np.quantile(spot, 0.95)),
            "pnl_p05": float(np.quantile(pnl, 0.05)),
            "pnl_p50": float(np.quantile(pnl, 0.50)),
            "pnl_p95": float(np.quantile(pnl, 0.95)),
            "trade_count": int(trades.sum()),
            "paths_with_trades": int(np.sum(trades > 0)),
        })
    return rows


def _median_path(paths: list[dict]) -> dict:
    target = float(np.median([row["pnl"] for row in paths]))
    chosen = min(paths, key=lambda row: abs(row["pnl"] - target))
    return {
        "path": chosen["path"],
        "pnl": chosen["pnl"],
        "max_dd": chosen["max_dd"],
        "ending_shares": chosen["ending_shares"],
        "ending_spot": chosen["ending_spot"],
        "days": chosen["days"],
    }


def _opening_trades(book: dict) -> list[dict]:
    trades = []
    for leg in book["legs"]:
        trades.append({
            "day": 0,
            "action": "buy_put" if leg.kind == "put" else "sell_call",
            "qty": leg.qty,
            "strike": leg.strike,
            "expiry": leg.expiry,
            "premium": round(leg.entry, 4),
            "tag": leg.tag,
        })
    return trades


def _vol_view(cfg: BookConfig) -> dict:
    up = _iv_at(cfg.spot * 1.2, cfg.spot, cfg.iv, cfg.iv_elasticity)
    down = _iv_at(cfg.spot * 0.8, cfg.spot, cfg.iv, cfg.iv_elasticity)
    return {
        "realized_vol": cfg.tqqq_vol,
        "realized_vol_drifts": False,
        "qqq_vol": cfg.tqqq_vol / 3.0,
        "daily_shocks": "symmetric normal, so the price goes up and down",
        "jumps": {
            "per_year": cfg.lambda_j,
            "mean_log": cfg.mu_j,
            "sigma": cfg.sigma_j,
            "bias": "down",
        },
        "iv_rule": "opening IV times (start spot / spot) to the power of the elasticity, floored at 0.20 and capped at 1.60",
        "iv_elasticity": cfg.iv_elasticity,
        "iv_if_spot_up_20pct": round(up, 4),
        "iv_if_spot_down_20pct": round(down, 4),
        "note": (
            "There is no separate vol drift. Realized vol stays at the input. "
            "Implied vol rises when the price falls and falls when the price rises, so both directions are covered. "
            "Daily price ups and downs come from the normal shock. Jumps add extra down days."
        ),
    }


def run(cfg: BookConfig | None = None) -> dict:
    cfg = cfg or BookConfig()
    path_days = min(cfg.horizon_days, cfg.put_dte)
    book = build_book(cfg)
    spots = _paths(cfg, path_days)
    rows = []
    navs = np.empty((cfg.paths, path_days + 1), dtype=float)
    trade_counts = np.zeros((cfg.paths, path_days + 1), dtype=int)
    paths = []
    for i in range(cfg.paths):
        stats = _simulate(spots[i], book, cfg)
        navs[i] = stats["nav"]
        days = stats.pop("days")
        stats.pop("nav")
        trade_counts[i] = [len(day["trades"]) for day in days]
        paths.append({"path": i, **stats, "days": days})
        rows.append(stats)
    call_lines = [
        {
            "tag": leg.tag,
            "side": leg.side,
            "qty": leg.qty,
            "strike": leg.strike,
            "dte": leg.expiry,
            "years": round(leg.expiry * DT, 4),
            "otm": round(leg.otm, 4),
            "premium": round(leg.entry, 4),
            "priced_by": "BlackScholesModel analytic",
            "days_left_at_horizon": max(0, leg.expiry - path_days),
        }
        for leg in book["legs"]
        if leg.kind == "call"
    ]
    put_lines = [
        {
            "tag": leg.tag,
            "side": leg.side,
            "qty": leg.qty,
            "strike": leg.strike,
            "dte": leg.expiry,
            "years": round(leg.expiry * DT, 4),
            "premium": round(leg.entry, 4),
            "priced_by": "BlackScholesModel analytic",
            "days_left_at_horizon": max(0, leg.expiry - path_days),
        }
        for leg in book["legs"]
        if leg.kind == "put"
    ]
    result = {
        "note": (
            "Opening premiums are this repo's analytic Black-Scholes at the opening spot, "
            "IV, and each leg's own DTE. They are not the IBKR fill. "
            "The path is shorter than the put when horizon_days < put_dte, so the put is still marked, not expired."
        ),
        "config": {
            "lots": cfg.lots,
            "shares": book["shares"],
            "spot": cfg.spot,
            "initial_dte": cfg.put_dte,
            "put_dte": cfg.put_dte,
            "horizon_days": cfg.horizon_days,
            "path_days": path_days,
            "put_days_left": max(0, cfg.put_dte - path_days),
            "iv": cfg.iv,
            "rate": cfg.rate,
            "tqqq_vol": cfg.tqqq_vol,
            "qqq_mu": cfg.qqq_mu,
            "strategy": cfg.strategy,
            "paths": cfg.paths,
            "seed": cfg.seed,
            "tenors": [{"tag": t.tag, "days": t.days, "otm": t.otm, "contracts": _qty(t.contracts, cfg.lots)} for t in cfg.tenors],
        },
        "book": {
            "deployed": book["deployed"],
            "put_premium": book["put_premium"],
            "call_premium": book["call_premium"],
            "floor": book["floor"],
            "model_gap_vs_fill": book["model_gap"],
            "opening_trades": _opening_trades(book),
            "puts": put_lines,
            "calls": call_lines,
        },
        "monte_carlo": _summary(cfg.strategy, rows),
        "vol": _vol_view(cfg),
        "ribbon": _ribbon(spots, navs, trade_counts),
        "median_path": _median_path(paths),
    }
    mtm = navs + book["deployed"]
    for day, row in enumerate(result["ribbon"]):
        row["mtm_p05"] = float(np.quantile(mtm[:, day], 0.05))
        row["mtm_p50"] = float(np.quantile(mtm[:, day], 0.50))
        row["mtm_p95"] = float(np.quantile(mtm[:, day], 0.95))
    from tqqq_book.report import write_report
    from tqqq_book.store import save_run
    result["saved"] = save_run(result, paths)
    html = write_report(result, Path(result["saved"]["dir"]) / "desk.html")
    write_report(result, Path(result["saved"]["dir"]).parent / "tqqq-desk.html")
    result["saved"]["html"] = html
    return result


def config_from_mapping(raw: dict | None = None) -> BookConfig:
    """Accept the desk names (putDte, horizonDays) or the Python names."""
    raw = raw or {}
    base = BookConfig()
    otm = raw.get("otm", [tenor.otm for tenor in base.tenors])
    contracts = raw.get("contracts", [tenor.contracts for tenor in base.tenors])
    if isinstance(otm, str):
        otm = [float(part) for part in otm.split(",")]
    if isinstance(contracts, str):
        contracts = [int(part) for part in contracts.split(",")]
    if len(otm) != 4 or len(contracts) != 4:
        raise ValueError("otm and contracts need four values: 1w, 2w, 3w, 4w.")
    tenors = tuple(
        Tenor(day, float(pct), int(qty), tag)
        for day, pct, qty, tag in zip((5, 10, 15, 20), otm, contracts, ("w1", "w2", "w3", "w4"))
    )

    def pick(*names, default):
        for name in names:
            if name in raw and raw[name] is not None:
                return raw[name]
        return default

    return BookConfig(
        lots=int(pick("lots", default=base.lots)),
        spot=float(pick("spot", default=base.spot)),
        qqq_mu=float(pick("qqqMu", "qqq_mu", "drift", default=base.qqq_mu)),
        tqqq_vol=float(pick("tqqqVol", "tqqq_vol", "vol", default=base.tqqq_vol)),
        iv=float(pick("iv", default=base.iv)),
        rate=float(pick("rate", default=base.rate)),
        horizon_days=int(pick("horizonDays", "horizon_days", "horizon", default=base.horizon_days)),
        put_dte=int(pick("initialDte", "initial_dte", "putDte", "put_dte", "dte", default=base.put_dte)),
        paths=int(pick("paths", "nPaths", default=base.paths)),
        seed=int(pick("seed", default=base.seed)),
        strategy=str(pick("strategy", default=base.strategy)),
        tenors=tenors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Price and simulate the configurable TQQQ book.")
    parser.add_argument("--lots", type=int, default=100)
    parser.add_argument("--put-dte", type=int, default=180, help="Initial DTE of the protective puts you buy.")
    parser.add_argument("--initial-dte", type=int, default=None, help="Same as --put-dte.")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--otm", default="0.03,0.05,0.08,0.10", help="1w,2w,3w,4w percent OTM as decimals")
    parser.add_argument("--contracts", default="12,12,12,10")
    parser.add_argument("--spot", type=float, default=59.22)
    parser.add_argument("--iv", type=float, default=0.71)
    parser.add_argument("--vol", type=float, default=0.66)
    parser.add_argument("--drift", type=float, default=0.12)
    parser.add_argument("--paths", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strategy", default="wheel", choices=("wheel", "hold"))
    parser.add_argument("--report", default="", help="Write an HTML desk report to this path.")
    args = parser.parse_args()
    put_dte = args.initial_dte if args.initial_dte is not None else args.put_dte
    result = run(config_from_mapping(vars(args) | {"horizon_days": args.horizon, "put_dte": put_dte, "initial_dte": put_dte, "vol": args.vol, "drift": args.drift}))
    if args.report:
        from tqqq_book.report import write_report
        print(write_report(result, args.report))
    book = result["book"]
    mc = result["monte_carlo"]
    cfg = result["config"]
    print(
        f"lots {cfg['lots']}  shares {cfg['shares']}  "
        f"put DTE {cfg['initial_dte']}  horizon {cfg['horizon_days']}  "
        f"path {cfg['path_days']}d  put still has {cfg['put_days_left']}d"
    )
    print(f"deployed ${book['deployed']:,.0f}  put ${book['put_premium']:,.0f}  call ${book['call_premium']:,.0f}  floor ${book['floor']:,.0f}")
    print(f"BS versus the scaled fill ${book['model_gap_vs_fill']:,.0f}")
    for line in book["puts"]:
        print(f"  put {line['tag']} {line['qty']} x {line['strike']:.0f}  {line['dte']}d  ${line['premium']:.2f}  left {line['days_left_at_horizon']}d")
    for line in book["calls"]:
        print(f"  call {line['tag']} {line['qty']} x {line['strike']:.0f}  {line['dte']}d  otm {line['otm']:.1%}  ${line['premium']:.2f}")
    print(
        f"{mc['strategy']}  {mc['paths']} paths  median ${mc['median']:,.0f}  "
        f"mean ${mc['mean']:,.0f}  p05 ${mc['p05']:,.0f}  p95 ${mc['p95']:,.0f}  "
        f"P(loss) {mc['prob_loss']:.1%}  median DD ${mc['median_drawdown']:,.0f}"
    )
    vol = result["vol"]
    saved = result["saved"]
    print(f"saved {saved['id']}  {saved['dir']}")
    print(
        f"vol stays {vol['realized_vol']:.2f}. No vol drift. "
        f"IV at +20% spot {vol['iv_if_spot_up_20pct']:.2f}, at -20% spot {vol['iv_if_spot_down_20pct']:.2f}."
    )


if __name__ == "__main__":
    main()
