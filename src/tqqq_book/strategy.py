"""Rules for the TQQQ book.

Each rule looks at the position and returns orders. Nothing here prices a path
or moves cash. `apply` runs the rules in order and hands each order to the desk.
Change a rule to change the strategy. A rule does not sell the hedge because
the stock dipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CONTRACT = 100
REF_SPOT = 59.22
REF_LOTS = 100


@dataclass
class Position:
    day: int
    last: int
    spot: float
    spot0: float
    sigma: float
    shares: int
    legs: list
    assigned: list
    rolled: bool
    mark: Callable
    price: Callable


@dataclass(frozen=True)
class Order:
    rule: str
    action: str
    qty: int
    strike: float = 0.0
    expiry: int = 0
    tag: str = ""
    otm: float = 0.0
    tenor: int = 0
    roll_at: int = 0
    leg: object = None
    reason: str = ""


def apply(position: Position, rules: list, execute: Callable) -> list[Order]:
    """Walk the rules. Each one sees the position after the earlier orders."""
    done: list[Order] = []
    for rule in rules:
        orders = rule.apply(position)
        for order in orders:
            execute(order)
        done.extend(orders)
    return done


def _open_long_puts(legs: list) -> list:
    return [leg for leg in legs if leg.open and leg.side == "long" and leg.kind == "put" and leg.qty > 0]


def _protective_strike(legs: list) -> float:
    original = [leg.strike for leg in _open_long_puts(legs) if not str(leg.tag).startswith("roll")]
    if original:
        return min(original)
    strikes = [leg.strike for leg in _open_long_puts(legs)]
    return min(strikes) if strikes else 0.0


def _covered_calls(legs: list) -> int:
    return sum(leg.qty * CONTRACT for leg in legs if leg.open and leg.side == "short" and leg.kind == "call")


@dataclass(frozen=True)
class Harvest:
    """Buy back a short call that has already made its money, once the stock is down."""

    drop: float = 0.08
    keep: float = 0.5
    name: str = "harvest"

    def apply(self, position: Position) -> list[Order]:
        if self.drop <= 0 or position.spot > position.spot0 * (1.0 - self.drop):
            return []
        orders = []
        for leg in list(position.legs):
            if not (leg.open and leg.side == "short" and leg.kind == "call" and leg.entry > 0):
                continue
            if position.mark(leg) <= self.keep * leg.entry:
                orders.append(Order(self.name, "buy_call", leg.qty, leg.strike, leg.expiry, leg.tag, leg=leg))
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "If the stock is down this far, buy back a short call worth this fraction of its credit or less.",
            "drop": self.drop,
            "keep": self.keep,
        }


@dataclass(frozen=True)
class CrashRoll:
    """Once, below the crash line, roll part of each put down. Leave the rest on the shares."""

    fraction: float = 0.5
    spot: float = REF_SPOT
    line_at_ref: float = 45.0
    down: float = 10.0
    name: str = "crash_roll"

    @property
    def line(self) -> float:
        return self.line_at_ref * self.spot / REF_SPOT

    def apply(self, position: Position) -> list[Order]:
        if position.spot > self.line:
            return []
        if position.rolled or self.fraction <= 0:
            position.rolled = True
            return []
        orders = []
        for leg in list(position.legs):
            if not (leg.open and leg.side == "long" and leg.kind == "put" and leg.qty > 0):
                continue
            qty = int(leg.qty * self.fraction)
            if qty <= 0:
                continue
            strike = float(max(1, round(leg.strike - self.down)))
            orders.append(Order(
                self.name, "roll_put", qty, strike, leg.expiry, f"roll{int(strike)}",
                leg=leg, reason=f"from {leg.strike:.0f}",
            ))
        position.rolled = True
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "Once the stock is at or below the crash line, roll this fraction of each put down. The rest stay.",
            "fraction": self.fraction,
            "line": round(self.line, 2),
            "down": self.down,
        }


@dataclass(frozen=True)
class KeepTenor:
    """When a put has this many days left, sell it and buy the same strike back out at its own DTE."""

    sleeves: tuple = ()
    name: str = "keep_tenor"

    def apply(self, position: Position) -> list[Order]:
        orders = []
        for leg in list(position.legs):
            if not (leg.open and leg.side == "long" and leg.kind == "put" and leg.qty > 0):
                continue
            if leg.roll_at <= 0 or leg.tenor <= 0:
                continue
            left = leg.expiry - position.day
            if left > leg.roll_at or position.day >= position.last:
                continue
            orders.append(Order(
                self.name, "extend_put", leg.qty, leg.strike, position.day + leg.tenor, leg.tag,
                tenor=leg.tenor, roll_at=leg.roll_at, leg=leg,
                reason=f"{left}d left, back to {leg.tenor}d",
            ))
        return orders

    def describe(self) -> dict:
        schedule = ", ".join(
            f"{sleeve.weight:.0%} {sleeve.dte}d rolls at {sleeve.roll_at}"
            for sleeve in self.sleeves
        )
        return {
            "name": self.name,
            "says": "Roll a sleeve back to its own DTE, same strike, when the days left hit its roll point.",
            "sleeves": schedule,
        }


@dataclass(frozen=True)
class WriteCalls:
    """Sell the call ladder above the puts that still cover the shares. Leave a reserve uncovered."""

    tenors: tuple[tuple[int, float, int, str], ...]
    lots: int = REF_LOTS
    spot: float = REF_SPOT
    every: int = 5
    floor_at_ref: float = 52.0
    min_spot_frac: float = 0.95
    reserve_at_ref: int = 1000
    name: str = "write_calls"

    @property
    def floor(self) -> float:
        return self.floor_at_ref * self.spot / REF_SPOT

    @property
    def reserve(self) -> int:
        return int(round(self.reserve_at_ref * self.lots / REF_LOTS))

    def apply(self, position: Position) -> list[Order]:
        if position.day <= 0 or position.day % self.every != 0:
            return []
        if position.spot < min(self.floor, position.spot * self.min_spot_frac):
            return []
        room = max(0, int((position.shares - _covered_calls(position.legs) - self.reserve) // CONTRACT))
        if room <= 0:
            return []
        put_strike = _protective_strike(position.legs)
        orders = []
        for days, otm, contracts, tag in self.tenors:
            have = sum(
                leg.qty for leg in position.legs
                if leg.open and leg.side == "short" and leg.kind == "call" and leg.tag == tag
            )
            need = min(max(0, int(round(contracts * self.lots / REF_LOTS)) - have), room)
            if need <= 0 or position.day + days >= position.last + 1:
                continue
            strike = float(max(1, round(position.spot * (1.0 + otm))))
            if put_strike and strike < put_strike:
                continue
            orders.append(Order(self.name, "sell_call", need, strike, position.day + days, tag, otm=otm, tenor=days))
            room -= need
            if room <= 0:
                break
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "Every few days, sell the call ladder. Never below the puts still on the shares, and never on the reserve.",
            "every": self.every,
            "floor": round(self.floor, 2),
            "min_spot_frac": self.min_spot_frac,
            "reserve_shares": self.reserve,
        }


@dataclass(frozen=True)
class ReplaceAssigned:
    """After a call takes shares, sell a put under that call, only for stock the long puts still cover."""

    name: str = "replace_assigned"

    def apply(self, position: Position) -> list[Order]:
        if not position.assigned:
            return []
        protected = sum(leg.qty for leg in position.legs if leg.open and leg.kind == "put" and leg.side == "long")
        short_puts = sum(leg.qty for leg in position.legs if leg.open and leg.kind == "put" and leg.side == "short")
        room = protected - int(position.shares // CONTRACT) - short_puts
        orders = []
        still = []
        for qty, call_strike, otm, tenor_days in position.assigned:
            take = min(qty, max(0, room))
            if take <= 0:
                still.append((qty, call_strike, otm, tenor_days))
                continue
            mirror = otm if otm > 0 else 0.05
            strike = float(max(1, round(call_strike * (1.0 - mirror))))
            if strike >= call_strike:
                strike = max(1.0, call_strike - 1.0)
            expiry = min(position.last, position.day + max(5, tenor_days))
            if expiry <= position.day:
                still.append((qty, call_strike, otm, tenor_days))
                continue
            orders.append(Order(self.name, "sell_put", take, strike, expiry, "reenter", otm=mirror, tenor=tenor_days))
            room -= take
            if take < qty:
                still.append((qty - take, call_strike, otm, tenor_days))
        position.assigned = still
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "If a call took shares, sell a put under that call, and only while the long puts still cover more than the shares.",
        }


def rules_for(cfg) -> list:
    """The wheel is these four rules. Hold is no rules."""
    if getattr(cfg, "strategy", "wheel") != "wheel":
        return []
    tenors = tuple((tenor.days, tenor.otm, tenor.contracts, tenor.tag) for tenor in cfg.tenors)
    return [
        Harvest(drop=cfg.harvest_drop),
        CrashRoll(fraction=cfg.roll_fraction, spot=cfg.spot),
        KeepTenor(sleeves=tuple(cfg.put_sleeves)),
        WriteCalls(tenors=tenors, lots=cfg.lots, spot=cfg.spot, min_spot_frac=cfg.stagger_min_spot_frac),
        ReplaceAssigned(),
    ]


def describe(rules: list) -> list[dict]:
    return [rule.describe() for rule in rules]
