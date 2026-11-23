"""Rules for the TQQQ book.

Each rule looks at the position and returns orders. Nothing here prices a path
or moves cash. `apply` runs the rules in order and hands each order to the desk.
Change a rule to change the strategy. A rule does not sell the hedge because
the stock dipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from tqqq_book.policy import CallPolicy
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
    trend: float = 0.0
    drawdown: float = 0.0
    initial_iv: float = 0.71
    max_shares: int = REF_LOTS * CONTRACT


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


def short_put_capacity(shares: int, legs: list, max_shares: int) -> int:
    """Reserve inventory for full assignment of every open short put."""
    short_puts = sum(leg.qty for leg in legs if leg.open and leg.side == "short" and leg.kind == "put")
    return max(0, (max_shares - shares) // CONTRACT - short_puts)


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
class RollUp:
    """Trail the long puts up. Every $20 the stock rises, lift the put by $20."""

    gap: float = 20.0
    name: str = "roll_up"

    def apply(self, position: Position) -> list[Order]:
        orders = []
        for leg in list(position.legs):
            if not (leg.open and leg.side == "long" and leg.kind == "put" and leg.qty > 0):
                continue
            if position.spot < leg.strike + self.gap:
                continue
            target = float(int(position.spot // self.gap) * self.gap)
            if target <= leg.strike:
                continue
            orders.append(Order(
                self.name, "roll_put", leg.qty, target, leg.expiry, leg.tag,
                tenor=leg.tenor, roll_at=leg.roll_at, leg=leg,
                reason=f"from {leg.strike:.0f}, spot {position.spot:.0f}",
            ))
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "When the stock is $20 above a long put, roll that put up to the $20 below the stock. Locks a higher floor. Costs the roll.",
            "gap": self.gap,
        }


@dataclass(frozen=True)
class RollDown:
    """Trail the long puts down. Every $12 the stock falls, lower the put, but not under the floor."""

    gap: float = 12.0
    floor: float = 40.0
    name: str = "roll_down"

    def apply(self, position: Position) -> list[Order]:
        orders = []
        for leg in list(position.legs):
            if not (leg.open and leg.side == "long" and leg.kind == "put" and leg.qty > 0):
                continue
            if position.spot > leg.strike - self.gap:
                continue
            target = float(int(position.spot // self.gap) * self.gap)
            if target < self.floor or target >= leg.strike:
                continue
            orders.append(Order(
                self.name, "roll_put", leg.qty, target, leg.expiry, leg.tag,
                tenor=leg.tenor, roll_at=leg.roll_at, leg=leg,
                reason=f"from {leg.strike:.0f}, spot {position.spot:.0f}, floor {self.floor:.0f}",
            ))
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "When the stock is this far under a long put, roll that put down to the rung under the stock. Never below the min strike.",
            "gap": self.gap,
            "floor": self.floor,
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


def _ratio_for(grid: tuple, drop: float) -> float:
    ratio = grid[0][1] if grid else 1.0
    for level, value in grid:
        if drop + 1e-9 >= level:
            ratio = value
    return ratio


@dataclass(frozen=True)
class PutRatio:
    """Hold more puts than shares once the stock is down from the opening price."""

    grid: tuple = ((0.0, 1.0), (0.15, 1.1), (0.30, 1.2), (0.50, 1.5))
    name: str = "put_ratio"

    def apply(self, position: Position) -> list[Order]:
        if position.shares <= 0 or position.spot0 <= 0 or not self.grid:
            return []
        drop = max(0.0, (position.spot0 - position.spot) / position.spot0)
        want = int(round(position.shares / CONTRACT * _ratio_for(self.grid, drop)))
        puts = _open_long_puts(position.legs)
        have = sum(leg.qty for leg in puts)
        if want == have or not puts:
            return []
        if want > have:
            return self._add(puts, want - have, drop)
        return self._trim(puts, have - want, drop)

    def _add(self, puts: list, need: int, drop: float) -> list[Order]:
        weights = [max(1, leg.qty) for leg in puts]
        total = sum(weights)
        raw = [need * weight / total for weight in weights]
        floors = [int(value) for value in raw]
        left = need - sum(floors)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
        for index in order[:left]:
            floors[index] += 1
        orders = []
        for leg, qty in zip(puts, floors):
            if qty <= 0:
                continue
            orders.append(Order(
                self.name, "add_put", qty, leg.strike, leg.expiry, leg.tag,
                tenor=leg.tenor, roll_at=leg.roll_at,
                reason=f"drop {drop:.0%}, add to {leg.strike:.0f}",
            ))
        return orders

    def _trim(self, puts: list, extra: int, drop: float) -> list[Order]:
        orders = []
        for leg in sorted(puts, key=lambda item: item.strike, reverse=True):
            if extra <= 0:
                break
            qty = min(extra, leg.qty)
            orders.append(Order(
                self.name, "trim_put", qty, leg.strike, leg.expiry, leg.tag, leg=leg,
                reason=f"drop {drop:.0%}, trim {leg.strike:.0f}",
            ))
            extra -= qty
        return orders

    def describe(self) -> dict:
        grid = ",".join(f"{int(level * 100)}:{ratio:g}" for level, ratio in self.grid)
        return {
            "name": self.name,
            "says": "From the opening price, hold this many puts per 100 shares. The ratio steps up as the stock falls, and back as it recovers.",
            "grid": grid,
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
    policy: CallPolicy = field(default_factory=CallPolicy)
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
            target_otm = self.policy.otm(otm, position.sigma, position.initial_iv, position.trend, position.drawdown)
            strike = float(max(1, round(position.spot * (1.0 + target_otm))))
            if put_strike and strike < put_strike:
                continue
            orders.append(Order(self.name, "sell_call", need, strike, position.day + days, tag, otm=target_otm, tenor=days, reason=f"{self.policy.mode}: IV {position.sigma:.3f}, trend {position.trend:.3f}, drawdown {position.drawdown:.3f}, OTM {target_otm:.3f}"))
            room -= need
            if room <= 0:
                break
        return orders

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "Every few days, sell the call ladder off the current spot. Never below the puts still on the shares, and never on the reserve.",
            "call_policy": self.policy.mode,
            "every": self.every,
            "floor": round(self.floor, 2),
            "min_spot_frac": self.min_spot_frac,
            "reserve_shares": self.reserve,
        }


@dataclass(frozen=True)
class ReplaceAssigned:
    """Replace called-away shares, within both the hedge and initial inventory."""

    otm: float = 0.05
    tenor: int = 10
    name: str = "replace_assigned"

    def apply(self, position: Position) -> list[Order]:
        protected = sum(leg.qty for leg in position.legs if leg.open and leg.kind == "put" and leg.side == "long")
        short_puts = sum(leg.qty for leg in position.legs if leg.open and leg.kind == "put" and leg.side == "short")
        hedge_room = protected - (position.shares + CONTRACT - 1) // CONTRACT - short_puts
        room = min(hedge_room, short_put_capacity(position.shares, position.legs, position.max_shares))
        position.assigned = []
        if room <= 0:
            return []
        floor = _protective_strike(position.legs)
        strike = float(max(1, round(position.spot * (1.0 - self.otm))))
        if floor:
            strike = min(strike, floor)
        expiry = min(position.last, position.day + self.tenor)
        if expiry <= position.day:
            return []
        return [Order(
            self.name, "sell_put", room, strike, expiry, "reenter",
            otm=self.otm, tenor=self.tenor,
            reason=f"{room} lots under spot",
        )]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "says": "Replace called-away shares with short puts under spot and no higher than the long put. Shares plus all short-put assignment obligations cannot exceed the initial lots × 100; extra long puts do not increase that limit.",
            "otm": self.otm,
            "tenor": self.tenor,
        }


def all_rules(cfg) -> list:
    tenors = tuple((tenor.days, tenor.otm, tenor.contracts, tenor.tag) for tenor in cfg.tenors)
    return [
        Harvest(drop=cfg.harvest_drop, keep=getattr(cfg, "harvest_keep", 0.5)),
        CrashRoll(
            fraction=cfg.roll_fraction,
            spot=cfg.spot,
            line_at_ref=getattr(cfg, "crash_line", 45.0),
            down=getattr(cfg, "crash_down", 10.0),
        ),
        KeepTenor(sleeves=tuple(cfg.put_sleeves)),
        RollUp(gap=getattr(cfg, "roll_up_gap", 20.0)),
        RollDown(gap=getattr(cfg, "roll_down_gap", 12.0), floor=getattr(cfg, "roll_down_floor", 40.0)),
        PutRatio(grid=getattr(cfg, "put_ratio_grid", ((0.0, 1.0), (0.15, 1.1), (0.30, 1.2), (0.50, 1.5)))),
        WriteCalls(
            tenors=tenors, lots=cfg.lots, spot=cfg.spot, policy=CallPolicy.from_config(cfg),
            every=getattr(cfg, "write_every", 5),
            floor_at_ref=getattr(cfg, "write_floor", 52.0),
            min_spot_frac=cfg.stagger_min_spot_frac,
            reserve_at_ref=getattr(cfg, "reserve_shares", 1000),
        ),
        ReplaceAssigned(
            otm=getattr(cfg, "reenter_otm", 0.05),
            tenor=getattr(cfg, "reenter_tenor", 10),
        ),
    ]


def rules_for(cfg) -> list:
    """The wheel is these rules. A rule that is switched off is not here, so it cannot act."""
    if getattr(cfg, "strategy", "wheel") != "wheel":
        return []
    chosen = set(getattr(cfg, "rules", ()) or ())
    rules = all_rules(cfg)
    if not chosen:
        return rules
    return [rule for rule in rules if rule.name in chosen]


def catalogue(cfg) -> list[dict]:
    """Every rule, marked on or off. An empty choice means all on."""
    chosen = set(getattr(cfg, "rules", ()) or ())
    rows = []
    for rule in all_rules(cfg):
        row = rule.describe()
        row["on"] = getattr(cfg, "strategy", "wheel") == "wheel" and ((not chosen) or rule.name in chosen)
        rows.append(row)
    return rows


def describe(rules: list) -> list[dict]:
    return [rule.describe() for rule in rules]
