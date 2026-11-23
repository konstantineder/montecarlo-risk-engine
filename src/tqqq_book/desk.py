"""Run the 13 Nov TQQQ book with the knobs from the path desk.

Lots, protective-put DTE, and the 1w/2w/3w/4w call OTM ladder are inputs.
Everything else has the desk default. Put and call premiums are this repo's
Black-Scholes price at that DTE and IV. They are not snapped to the IBKR fill.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

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
class PutSleeve:
    """A slice of the protective puts. weight is a fraction of the contracts."""

    weight: float
    dte: int
    roll_at: int
    tag: str


def default_put_sleeves() -> tuple[PutSleeve, ...]:
    """10% is the uncapped lots: 60 days, rolled at 30. The rest is 90 and 180."""
    return (
        PutSleeve(0.10, 60, 30, "d60"),
        PutSleeve(0.40, 90, 45, "d90"),
        PutSleeve(0.50, 180, 90, "d180"),
    )


@dataclass(frozen=True)
class Tenor:
    days: int
    otm: float
    contracts: int
    tag: str
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
    harvest_keep: float = 0.5
    roll_fraction: float = 0.5
    crash_line: float = 45.0
    crash_down: float = 10.0
    roll_up_gap: float = 20.0
    roll_down_gap: float = 12.0
    roll_down_floor: float = 40.0
    put_ratio_grid: tuple = ((0.0, 1.0), (0.15, 1.1), (0.30, 1.2), (0.50, 1.5))
    write_every: int = 5
    write_floor: float = 52.0
    reserve_shares: int = 1000
    reenter_otm: float = 0.05
    reenter_tenor: int = 10
    put_monetize_drop: float = 0.10
    new_put_otm: float = 0.02
    write_floor_after_monetize: float = 0.90
    stagger_min_spot_frac: float = 0.95
    horizon_days: int = 100
    put_dte: int = 180
    paths: int = 400
    seed: int = 13
    strategy: str = "wheel"
    rules: tuple[str, ...] = ()
    call_policy: str = "fixed"
    call_vol_sensitivity: float = 1.0
    call_trend_sensitivity: float = 0.5
    call_drawdown_sensitivity: float = 0.5
    call_otm_min: float = 0.01
    call_otm_max: float = 0.50
    call_lookback: int = 20
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
    put_sleeves: tuple[PutSleeve, ...] = field(default_factory=default_put_sleeves)


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
    roll_at: int = 0


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


def _split_qty(qty: int, weights: list[float]) -> list[int]:
    """Whole contracts that sum to qty. The leftover goes to the largest remainder."""
    if qty <= 0:
        return [0 for _ in weights]
    raw = [qty * weight for weight in weights]
    floors = [int(math.floor(value)) for value in raw]
    left = qty - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i], weights[i]), reverse=True)
    for index in order[:left]:
        floors[index] += 1
    return floors


def build_book(cfg: BookConfig) -> dict:
    if cfg.lots < 1:
        raise ValueError("lots must be at least 1")
    if cfg.put_dte < 1:
        raise ValueError("put DTE must be at least 1")
    sleeves = cfg.put_sleeves
    if not sleeves:
        raise ValueError("put sleeves are empty")
    weight_sum = sum(sleeve.weight for sleeve in sleeves)
    if abs(weight_sum - 1.0) > 0.02:
        raise ValueError(f"put sleeve weights must sum to 1, not {weight_sum:.3f}")
    weights = [sleeve.weight / weight_sum for sleeve in sleeves]
    lines = (
        (_qty(60, cfg.lots), _scaled_strike(60.0, cfg.spot), "p60"),
        (_qty(40, cfg.lots), _scaled_strike(65.0, cfg.spot), "p65"),
    )
    puts = []
    for qty, strike, name in lines:
        for sleeve, part in zip(sleeves, _split_qty(qty, weights)):
            if part <= 0:
                continue
            puts.append((part, sleeve.dte, strike, f"{name}-{sleeve.tag}", sleeve.dte, sleeve.roll_at))
    weekly = []
    from tqqq_book.policy import CallPolicy
    policy = CallPolicy.from_config(cfg)
    for tenor in cfg.tenors:
        opening_otm = policy.otm(tenor.otm, cfg.iv, cfg.iv)
        strike = float(max(1, round(cfg.spot * (1.0 + opening_otm))))
        weekly.append((_qty(tenor.contracts, cfg.lots), tenor.days, strike, tenor.tag, opening_otm))

    if cfg.strategy == "stock":
        puts = []; weekly = []
    legs: list[Leg] = []
    put_premium = 0.0
    call_premium = 0.0
    checked = False
    for qty, expiry, strike, tag, tenor, roll_at in puts:
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
        legs.append(Leg("put", "long", strike, qty, expiry, entry, tag, tenor=tenor, roll_at=roll_at))
    for qty, expiry, strike, tag, otm in weekly:
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


def _simulate(path: np.ndarray, book: dict, cfg: BookConfig, *, record_history: bool = True) -> dict:
    legs = [Leg(**{k: getattr(leg, k) for k in Leg.__dataclass_fields__}) for leg in book["legs"]]
    cash = book["cash"]
    shares = book["shares"]
    buybacks = 0
    new_calls = 0
    assignments = 0
    rolled = False
    roll_cash = 0.0
    min_spot = float(path[0])
    wheel: list[tuple[int, float, float, int]] = []
    last = len(path) - 1
    peak = book["deployed"]
    max_dd = 0.0
    last_nav = 0.0
    navs: list[float] = []
    days: list[dict] = []
    day_trades: list[dict] = []
    strategy = cfg.strategy
    market_peak = float(path[0])
    max_shares = cfg.lots * CONTRACT

    def check_inventory() -> None:
        obligated = sum(leg.qty * CONTRACT for leg in legs if leg.open and leg.kind == "put" and leg.side == "short")
        if shares < 0 or shares + obligated > max_shares:
            raise ValueError(f"Inventory limit exceeded: {shares} shares + {obligated} short-put shares > {max_shares}")

    check_inventory()

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
                "rule": "settle",
                "action": action,
                "kind": leg.kind,
                "side": leg.side,
                "qty": leg.qty,
                "strike": leg.strike,
                "tag": leg.tag,
                "intrinsic": round(intrinsic, 4),
            })
            leg.open = False

    def buy_back(order, t: int, spot: float, sigma: float) -> None:
        nonlocal cash, buybacks
        leg = order.leg
        if leg is None or not leg.open or leg.qty <= 0:
            return
        mark = _mark(leg, spot, t, cfg.rate, sigma)
        cash -= leg.qty * CONTRACT * (mark + cfg.slippage)
        buybacks += 1
        day_trades.append({
            "rule": order.rule,
            "action": "buy_call",
            "qty": leg.qty,
            "strike": leg.strike,
            "premium": round(mark + cfg.slippage, 4),
            "tag": leg.tag,
        })
        leg.open = False

    def execute(order) -> None:
        nonlocal cash, roll_cash, new_calls
        if order.action == "buy_call":
            buy_back(order, t, spot, sigma)
            return
        if order.action == "roll_put":
            leg = order.leg
            if leg is None or not leg.open or order.qty <= 0:
                return
            mark = _mark(leg, spot, t, cfg.rate, sigma)
            proceeds = order.qty * CONTRACT * max(0.0, mark - cfg.slippage)
            cash += proceeds
            new_mark = _fast_price(spot, order.strike, max(0, leg.expiry - t) * DT, cfg.rate, sigma, "put")
            premium = new_mark + cfg.slippage
            cash -= order.qty * CONTRACT * premium
            roll_cash += proceeds - order.qty * CONTRACT * premium
            leg.qty -= order.qty
            if leg.qty <= 0:
                leg.open = False
            legs.append(Leg("put", "long", order.strike, order.qty, leg.expiry, premium, order.tag, tenor=leg.tenor, roll_at=leg.roll_at))
            day_trades.append({
                "rule": order.rule,
                "action": "roll_put",
                "qty": order.qty,
                "strike": order.strike,
                "premium": round(premium, 4),
                "tag": order.tag,
                "reason": order.reason,
            })
            return
        if order.action == "sell_call":
            if order.qty <= 0 or order.expiry <= t:
                return
            mark = _fast_price(spot, order.strike, (order.expiry - t) * DT, cfg.rate, sigma, "call")
            credit = max(0.0, mark - cfg.slippage)
            cash += order.qty * CONTRACT * credit
            legs.append(Leg("call", "short", order.strike, order.qty, order.expiry, credit, order.tag, otm=order.otm, tenor=order.tenor))
            new_calls += order.qty
            day_trades.append({
                "rule": order.rule,
                "action": "sell_call",
                "qty": order.qty,
                "strike": order.strike,
                "expiry": order.expiry,
                "premium": round(credit, 4),
                "tag": order.tag,
                "reason": order.reason,
            })
            return
        if order.action == "sell_put":
            qty = min(order.qty, short_put_capacity(shares, legs, max_shares))
            if qty <= 0 or order.expiry <= t:
                return
            mark = _fast_price(spot, order.strike, (order.expiry - t) * DT, cfg.rate, sigma, "put")
            credit = max(0.0, mark - cfg.slippage)
            cash += qty * CONTRACT * credit
            legs.append(Leg("put", "short", order.strike, qty, order.expiry, credit, order.tag, otm=order.otm, tenor=order.tenor))
            day_trades.append({
                "rule": order.rule,
                "action": "sell_put",
                "qty": qty,
                "strike": order.strike,
                "expiry": order.expiry,
                "premium": round(credit, 4),
                "tag": order.tag,
            })
            return
        if order.action == "extend_put":
            leg = order.leg
            if leg is None or not leg.open or order.qty <= 0 or order.expiry <= t:
                return
            mark = _mark(leg, spot, t, cfg.rate, sigma)
            cash += order.qty * CONTRACT * max(0.0, mark - cfg.slippage)
            premium = _fast_price(spot, order.strike, (order.expiry - t) * DT, cfg.rate, sigma, "put") + cfg.slippage
            cash -= order.qty * CONTRACT * premium
            leg.qty -= order.qty
            if leg.qty <= 0:
                leg.open = False
            legs.append(Leg(
                "put", "long", order.strike, order.qty, order.expiry, premium, order.tag,
                tenor=order.tenor, roll_at=order.roll_at,
            ))
            day_trades.append({
                "rule": order.rule,
                "action": "extend_put",
                "qty": order.qty,
                "strike": order.strike,
                "expiry": order.expiry,
                "premium": round(premium, 4),
                "tag": order.tag,
                "reason": order.reason,
            })
            return
        if order.action == "add_put":
            if order.qty <= 0 or order.expiry <= t:
                return
            premium = _fast_price(spot, order.strike, (order.expiry - t) * DT, cfg.rate, sigma, "put") + cfg.slippage
            cash -= order.qty * CONTRACT * premium
            legs.append(Leg(
                "put", "long", order.strike, order.qty, order.expiry, premium, order.tag,
                tenor=order.tenor, roll_at=order.roll_at,
            ))
            day_trades.append({
                "rule": order.rule,
                "action": "add_put",
                "qty": order.qty,
                "strike": order.strike,
                "expiry": order.expiry,
                "premium": round(premium, 4),
                "tag": order.tag,
                "reason": order.reason,
            })
            return
        if order.action == "trim_put":
            leg = order.leg
            if leg is None or not leg.open or order.qty <= 0:
                return
            mark = _mark(leg, spot, t, cfg.rate, sigma)
            cash += order.qty * CONTRACT * max(0.0, mark - cfg.slippage)
            leg.qty -= order.qty
            if leg.qty <= 0:
                leg.open = False
            day_trades.append({
                "rule": order.rule,
                "action": "trim_put",
                "qty": order.qty,
                "strike": leg.strike,
                "premium": round(mark, 4),
                "tag": leg.tag,
                "reason": order.reason,
            })

    from tqqq_book.strategy import Position, rules_for, short_put_capacity
    rules = rules_for(cfg) if strategy == "wheel" else []

    def snap(at_spot: float, at_t: int, at_sigma: float) -> list[dict]:
        rows = [
            {
                "name": "share",
                "side": "long",
                "qty": shares,
                "cost": round(cfg.spot, 4),
                "mark": round(at_spot, 4),
                "mtm": round(shares * at_spot, 2),
            },
            {
                "name": "cash",
                "qty": 1,
                "cost": 0.0,
                "mark": round(cash, 2),
                "mtm": round(cash, 2),
            },
        ]
        for leg in legs:
            if not leg.open or leg.qty <= 0:
                continue
            mark = _mark(leg, at_spot, at_t, cfg.rate, at_sigma)
            sign = 1.0 if leg.side == "long" else -1.0
            rows.append({
                "name": leg.kind,
                "side": leg.side,
                "qty": leg.qty,
                "strike": leg.strike,
                "expiry": leg.expiry,
                "tag": leg.tag,
                "cost": round(leg.entry, 4),
                "mark": round(mark, 4),
                "mtm": round(sign * mark * leg.qty * CONTRACT, 2),
            })
        return rows

    for t, spot in enumerate(path):
        spot = float(spot)
        day_trades = []
        steps: list[dict] = []
        rule_log: list[dict] = []
        min_spot = min(min_spot, spot)
        market_peak = max(market_peak, spot)
        sigma = _iv_at(spot, cfg.spot, cfg.iv, cfg.iv_elasticity)
        before = snap(spot, t, sigma) if record_history else []
        settle(t, spot)
        check_inventory()
        if day_trades and record_history:
            steps.append({"rule": "settle", "trades": list(day_trades), "before": before, "after": snap(spot, t, sigma)})
        if t < last and rules:
            position = Position(
                day=t,
                trend=spot / float(path[max(0, t - cfg.call_lookback)]) - 1.0,
                drawdown=1.0 - spot / market_peak,
                initial_iv=cfg.iv,
                last=last,
                spot=spot,
                spot0=cfg.spot,
                sigma=sigma,
                shares=shares,
                max_shares=max_shares,
                legs=legs,
                assigned=wheel,
                rolled=rolled,
                mark=lambda leg, t=t, spot=spot, sigma=sigma: _mark(leg, spot, t, cfg.rate, sigma),
                price=lambda kind, strike, expiry, t=t, spot=spot, sigma=sigma: _fast_price(
                    spot, strike, max(0, expiry - t) * DT, cfg.rate, sigma, kind
                ),
            )
            for rule in rules:
                before = snap(spot, t, sigma) if record_history else []
                start = len(day_trades)
                for order in rule.apply(position):
                    execute(order)
                    check_inventory()
                acted = len(day_trades) > start
                if record_history:
                    rule_log.append({"name": rule.name, "acted": acted})
                if acted and record_history:
                    steps.append({
                        "rule": rule.name,
                        "trades": day_trades[start:],
                        "before": before,
                        "after": snap(spot, t, sigma),
                    })
            wheel = position.assigned
            rolled = position.rolled
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
            "position": snap(spot, t, sigma) if record_history else [],
            "steps": steps,
            "rules": rule_log,
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
        active = np.isfinite(spots[:, day]) & np.isfinite(navs[:, day])
        spot = spots[active, day]
        pnl = navs[active, day]
        trades = trade_counts[active, day]
        rows.append({
            "day": day,
            "active_paths": int(active.sum()),
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
        **({"configuration": chosen["configuration"]} if "configuration" in chosen else {}),
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


def run(cfg: BookConfig | None = None, progress: Callable[[dict], None] | None = None, *, persist: bool = True, checkpoint_dir: Path | None = None) -> dict:
    cfg = cfg or BookConfig()
    if cfg.paths < 1 or cfg.horizon_days < 1:
        raise ValueError("paths and horizon days must be at least 1")

    def notify(phase: str, percent: float, completed: int = 0, **extra) -> None:
        if progress:
            progress({"phase": phase, "percent": round(percent, 2), "completed": completed, "total": cfg.paths, **extra})

    from tqqq_book.checkpoint import PathCheckpoint, root as checkpoint_root
    checkpoint = PathCheckpoint(checkpoint_dir or checkpoint_root(), asdict(cfg))
    notify("Building opening book", 0)
    path_days = cfg.horizon_days
    book = build_book(cfg)
    notify("Generating price paths", 2)
    spots = _paths(cfg, path_days)
    rows = []
    navs = np.empty((cfg.paths, path_days + 1), dtype=float)
    trade_counts = np.zeros((cfg.paths, path_days + 1), dtype=int)
    paths = []
    for i in range(cfg.paths):
        cached = checkpoint.load(i)
        if cached is None:
            stats = _simulate(spots[i], book, cfg)
            days = stats.pop("days")
            nav = stats.pop("nav")
            cached = {"path": i, **stats, "days": days, "prices": spots[i].tolist(), "nav_values": nav}
            checkpoint.save(i, cached)
        else:
            days = cached["days"]
            nav = cached.get("nav_values", [day["pnl"] for day in days])
        navs[i] = nav
        trade_counts[i] = [len(day["trades"]) for day in days]
        paths.append(cached)
        rows.append({key: value for key, value in cached.items() if key not in ("days", "prices", "config", "nav_values")})
        notify("Simulating paths", 5 + 85 * (i + 1) / cfg.paths, i + 1)
    notify("Calculating summary", 92, cfg.paths)
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
    from tqqq_book.strategy import catalogue, describe, rules_for
    fired = {rule["name"]: 0 for rule in catalogue(cfg)}
    for path in paths:
        seen = set()
        for day in path.get("days") or []:
            for row in day.get("rules") or []:
                if row.get("acted"):
                    seen.add(row["name"])
        for name in seen:
            fired[name] = fired.get(name, 0) + 1
    rules = []
    for row in catalogue(cfg):
        row["paths"] = fired.get(row["name"], 0)
        rules.append(row)
    result = {
        "note": (
            "Opening premiums are this repo's analytic Black-Scholes at the opening spot, "
            "IV, and each leg's own DTE. They are not the IBKR fill. "
            "Puts are staggered. Each sleeve rolls back to its own DTE when the days left hit its roll point. "
            "Each day the wheel calls strategy.apply on the position. The rules are listed below."
        ),
        "config": {
            **asdict(cfg),
            "lots": cfg.lots,
            "shares": book["shares"],
            "spot": cfg.spot,
            "initial_dte": cfg.put_dte,
            "put_dte": cfg.put_dte,
            "put_sleeves": [
                {"weight": sleeve.weight, "dte": sleeve.dte, "roll_at": sleeve.roll_at, "tag": sleeve.tag}
                for sleeve in cfg.put_sleeves
            ],
            "horizon_days": cfg.horizon_days,
            "path_days": path_days,
            "put_days_left": max(0, cfg.put_dte - path_days),
            "iv": cfg.iv,
            "rate": cfg.rate,
            "tqqq_vol": cfg.tqqq_vol,
            "qqq_mu": cfg.qqq_mu,
            "strategy": cfg.strategy,
            "rules": list(cfg.rules),
            "paths": cfg.paths,
            "seed": cfg.seed,
            "tenors": [{"tag": t.tag, "days": t.days, "otm": t.otm, "contracts": _qty(t.contracts, cfg.lots), "contracts_at_100": t.contracts} for t in cfg.tenors],
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
        "rules": rules,
        "ribbon": _ribbon(spots, navs, trade_counts),
        "median_path": _median_path(paths),
    }
    mtm = navs + book["deployed"]
    for day, row in enumerate(result["ribbon"]):
        row["mtm_p05"] = float(np.quantile(mtm[:, day], 0.05))
        row["mtm_p50"] = float(np.quantile(mtm[:, day], 0.50))
        row["mtm_p95"] = float(np.quantile(mtm[:, day], 0.95))
    if not persist:
        return {**result, "_paths": paths}
    from tqqq_book.report import write_report
    from tqqq_book.store import save_run
    notify("Saving paths", 94, cfg.paths)
    result["saved"] = save_run(result, paths)
    notify("Writing report", 96, cfg.paths)
    html = write_report(result, Path(result["saved"]["dir"]) / "desk.html")
    notify("Finishing report", 98, cfg.paths)
    write_report(result, Path(result["saved"]["dir"]).parent / "tqqq-desk.html")
    result["saved"]["html"] = html
    notify("Completed", 100, cfg.paths, run=result["saved"]["id"])
    return result


def parse_ratio_grid(text: str) -> tuple:
    """'0:1,15:1.1,30:1.2,50:1.5' is drop percent, puts per 100 shares."""
    if not str(text).strip():
        return ((0.0, 1.0), (0.15, 1.1), (0.30, 1.2), (0.50, 1.5))
    rows = []
    for part in str(text).split(","):
        bits = [piece.strip() for piece in part.split(":") if piece.strip()]
        if len(bits) != 2:
            raise ValueError("Each ratio step is percent:ratio, for example 15:1.1")
        level, ratio = float(bits[0]), float(bits[1])
        if level < 0 or ratio <= 0:
            raise ValueError(f"Bad ratio step {part}")
        rows.append((level / 100.0, ratio))
    return tuple(sorted(rows))


def parse_put_sleeves(text: str) -> tuple[PutSleeve, ...]:
    """'10:60:30,40:90:45,50:180:90' is percent, DTE, roll-at DTE."""
    sleeves = []
    for part in str(text).split(","):
        bits = [piece.strip() for piece in part.split(":") if piece.strip()]
        if len(bits) != 3:
            raise ValueError("Each put sleeve is percent:dte:roll, for example 10:60:30")
        percent, dte, roll_at = float(bits[0]), int(bits[1]), int(bits[2])
        if percent <= 0 or dte < 1 or roll_at < 1 or roll_at >= dte:
            raise ValueError(f"Bad put sleeve {part}. Roll DTE has to be shorter than the sleeve.")
        sleeves.append(PutSleeve(percent / 100.0, dte, roll_at, f"d{dte}"))
    total = sum(sleeve.weight for sleeve in sleeves)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"Put sleeve percents must sum to 100, not {total * 100:.1f}")
    return tuple(sleeves)


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

    sleeves = raw.get("putSleeves", raw.get("put_sleeves", raw.get("puts")))
    if isinstance(sleeves, str) and sleeves.strip():
        put_sleeves = parse_put_sleeves(sleeves)
    else:
        put_sleeves = base.put_sleeves
    return BookConfig(
        lots=int(pick("lots", default=base.lots)),
        spot=float(pick("spot", default=base.spot)),
        qqq_mu=float(pick("qqqMu", "qqq_mu", "drift", default=base.qqq_mu)),
        tqqq_vol=float(pick("tqqqVol", "tqqq_vol", "vol", default=base.tqqq_vol)),
        iv=float(pick("iv", default=base.iv)),
        call_policy=str(pick("callPolicy", "call_policy", default=base.call_policy)),
        call_vol_sensitivity=float(pick("callVolSensitivity", "call_vol_sensitivity", default=base.call_vol_sensitivity)),
        call_trend_sensitivity=float(pick("callTrendSensitivity", "call_trend_sensitivity", default=base.call_trend_sensitivity)),
        call_drawdown_sensitivity=float(pick("callDrawdownSensitivity", "call_drawdown_sensitivity", default=base.call_drawdown_sensitivity)),
        call_otm_min=float(pick("callOtmMin", "call_otm_min", default=base.call_otm_min)),
        call_otm_max=float(pick("callOtmMax", "call_otm_max", default=base.call_otm_max)),
        call_lookback=int(pick("callLookback", "call_lookback", default=base.call_lookback)),
        iv_elasticity=float(pick("ivElasticity", "iv_elasticity", default=base.iv_elasticity)),
        slippage=float(pick("slippage", default=base.slippage)),
        lambda_j=float(pick("lambdaJ", "lambda_j", default=base.lambda_j)),
        mu_j=float(pick("muJ", "mu_j", default=base.mu_j)),
        sigma_j=float(pick("sigmaJ", "sigma_j", default=base.sigma_j)),
        rate=float(pick("rate", default=base.rate)),
        horizon_days=int(pick("horizonDays", "horizon_days", "horizon", default=base.horizon_days)),
        put_dte=int(pick("initialDte", "initial_dte", "putDte", "put_dte", "dte", default=base.put_dte)),
        paths=int(pick("paths", "nPaths", default=base.paths)),
        seed=int(pick("seed", default=base.seed)),
        strategy=str(pick("strategy", default=base.strategy)),
        rules=tuple(
            part.strip()
            for part in str(pick("rules", default="")).split(",")
            if part.strip()
        ),
        harvest_drop=float(pick("harvest_drop", "harvestDrop", default=base.harvest_drop)),
        harvest_keep=float(pick("harvest_keep", "harvestKeep", default=base.harvest_keep)),
        roll_fraction=float(pick("roll_fraction", "rollFraction", default=base.roll_fraction)),
        crash_line=float(pick("crash_line", "crashLine", default=base.crash_line)),
        crash_down=float(pick("crash_down", "crashDown", default=base.crash_down)),
        roll_up_gap=float(pick("roll_up_gap", "rollUpGap", default=base.roll_up_gap)),
        roll_down_gap=float(pick("roll_down_gap", "rollDownGap", default=base.roll_down_gap)),
        roll_down_floor=float(pick("roll_down_floor", "rollDownFloor", default=base.roll_down_floor)),
        put_ratio_grid=parse_ratio_grid(pick("put_ratio_grid", "putRatioGrid", "put_ratio", default="")),
        write_every=int(pick("write_every", "writeEvery", default=base.write_every)),
        write_floor=float(pick("write_floor", "writeFloor", default=base.write_floor)),
        stagger_min_spot_frac=float(pick("stagger_min_spot_frac", "min_spot_frac", "minSpotFrac", default=base.stagger_min_spot_frac)),
        reserve_shares=int(pick("reserve_shares", "reserve", default=base.reserve_shares)),
        reenter_otm=float(pick("reenter_otm", "reenterOtm", default=base.reenter_otm)),
        reenter_tenor=int(pick("reenter_tenor", "reenterTenor", default=base.reenter_tenor)),
        tenors=tenors,
        put_sleeves=put_sleeves,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Price and simulate the configurable TQQQ book.")
    parser.add_argument("--lots", type=int, default=100)
    parser.add_argument("--puts", default="10:60:30,40:90:45,50:180:90", help="Put sleeves as percent:dte:roll. They must sum to 100.")
    parser.add_argument("--put-dte", type=int, default=180, help="Label for the long hedge. The put sleeves set the actual DTEs.")
    parser.add_argument("--initial-dte", type=int, default=None, help="Same as --put-dte.")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--otm", default="0.03,0.05,0.08,0.10", help="1w,2w,3w,4w percent OTM as decimals")
    parser.add_argument("--contracts", default="12,12,12,10")
    parser.add_argument("--spot", type=float, default=59.22)
    parser.add_argument("--iv", type=float, default=0.71)
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--iv-elasticity", type=float, default=0.86)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--lambda-j", type=float, default=0.7)
    parser.add_argument("--mu-j", type=float, default=-0.07)
    parser.add_argument("--sigma-j", type=float, default=0.05)
    parser.add_argument("--call-policy", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--call-vol-sensitivity", type=float, default=1.0)
    parser.add_argument("--call-trend-sensitivity", type=float, default=0.5)
    parser.add_argument("--call-drawdown-sensitivity", type=float, default=0.5)
    parser.add_argument("--call-otm-min", type=float, default=0.01)
    parser.add_argument("--call-otm-max", type=float, default=0.50)
    parser.add_argument("--call-lookback", type=int, default=20)
    parser.add_argument("--vol", type=float, default=0.66)
    parser.add_argument("--drift", type=float, default=0.12)
    parser.add_argument("--paths", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strategy", default="wheel", choices=("wheel", "hold", "stock"))
    parser.add_argument("--rules", default="", help="Comma separated rule names to leave on. Empty means all on.")
    parser.add_argument("--harvest-drop", type=float, default=0.08)
    parser.add_argument("--harvest-keep", type=float, default=0.5)
    parser.add_argument("--roll-fraction", type=float, default=0.5)
    parser.add_argument("--crash-line", type=float, default=45.0)
    parser.add_argument("--crash-down", type=float, default=10.0)
    parser.add_argument("--roll-up-gap", type=float, default=20.0)
    parser.add_argument("--roll-down-gap", type=float, default=12.0)
    parser.add_argument("--roll-down-floor", type=float, default=40.0)
    parser.add_argument("--put-ratio", default="0:1,15:1.1,30:1.2,50:1.5", help="Drop percent:puts per 100 shares.")
    parser.add_argument("--write-every", type=int, default=5)
    parser.add_argument("--write-floor", type=float, default=52.0)
    parser.add_argument("--min-spot-frac", type=float, default=0.95)
    parser.add_argument("--reserve", type=int, default=1000)
    parser.add_argument("--reenter-otm", type=float, default=0.05)
    parser.add_argument("--reenter-tenor", type=int, default=10)
    parser.add_argument("--report", default="", help="Write an HTML desk report to this path.")
    parser.add_argument("--progress", action="store_true", help="Stream JSON progress events for the simulation queue.")
    parser.add_argument("--rebuild", action="store_true", help="Rewrite the latest desk.html from the saved run. Does not resimulate.")
    args = parser.parse_args()
    if args.rebuild:
        from tqqq_book.report import write_report
        from tqqq_book.store import folder_for, load_saved
        folder = folder_for("latest")
        saved = load_saved(folder)
        print(write_report(saved, folder / "desk.html"))
        print(write_report(saved, folder.parent / "tqqq-desk.html"))
        return
    put_dte = args.initial_dte if args.initial_dte is not None else args.put_dte
    def emit_progress(event: dict) -> None:
        print("TQQQ_PROGRESS " + json.dumps(event), flush=True)

    result = run(config_from_mapping(vars(args) | {
        "horizon_days": args.horizon,
        "put_dte": put_dte,
        "initial_dte": put_dte,
        "vol": args.vol,
        "drift": args.drift,
        "put_sleeves": args.puts,
    }), progress=emit_progress if args.progress else None)
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
