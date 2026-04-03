from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

from models.schwartz_two_factor import SchwartzTwoFactorModel
from products.storage import Storage
from products.storage_helpers import StorageConfig


def d(year: int, month: int, day: int) -> date:
    return date(year, month, day)


@dataclass(frozen=True)
class StorageScenario:
    name: str
    asset_id: str
    start_date: date
    end_date: date
    initial_amount: float
    volume_constraints: tuple[tuple[date, date, float, float], ...]
    injection_rates: tuple[tuple[date, date, float, float], ...]
    withdrawal_rates: tuple[tuple[date, date, float, float], ...]
    injection_cost: float
    withdrawal_cost: float
    penalty: float
    num_states: int
    rollout_interval_days: float = 1.0


@dataclass(frozen=True)
class StorageS2FScenario:
    name: str
    storage: StorageScenario
    curve_points: tuple[tuple[date, float], ...]
    rate_annual: float = 0.0
    short_term_mean_reversion_annual: float = 8.0
    short_term_vol_annual: float = 0.00001
    long_term_drift_annual: float = 0.0
    long_term_vol_annual: float = 0.00005
    rho: float = 0.2


def day_number(base_date: date, current_date: date) -> float:
    return float((current_date - base_date).days)


def offset_to_date(base_date: date, offset: float) -> date:
    return base_date + timedelta(days=int(round(offset)))


def curve_date(start_date: date, end_date: date, fraction: float) -> date:
    days = (end_date - start_date).days
    return start_date + timedelta(days=int(round(days * fraction)))


def build_storage_from_scenario(scenario: StorageScenario) -> Storage:
    storage_config = StorageConfig()

    for start_date, end_date, vmin, vmax in scenario.volume_constraints:
        storage_config.add_volume_constraint(
            day_number(scenario.start_date, start_date),
            day_number(scenario.start_date, end_date),
            vmin,
            vmax,
            scenario.penalty,
        )

    for start_date, end_date, point, rate in scenario.injection_rates:
        storage_config.add_injection_flexibility(
            day_number(scenario.start_date, start_date),
            day_number(scenario.start_date, end_date),
            point,
            rate,
        )

    for start_date, end_date, point, rate in scenario.withdrawal_rates:
        storage_config.add_withdrawal_flexibility(
            day_number(scenario.start_date, start_date),
            day_number(scenario.start_date, end_date),
            point,
            rate,
        )

    storage_config.add_variable_injection_cost(0.0, scenario.injection_cost)
    storage_config.add_variable_withdrawal_cost(0.0, scenario.withdrawal_cost)

    return Storage(
        asset_id=scenario.asset_id,
        start_date=0.0,
        end_date=day_number(scenario.start_date, scenario.end_date),
        initial_amount=scenario.initial_amount,
        storage_config=storage_config,
        num_states=scenario.num_states,
        rollout_interval=scenario.rollout_interval_days,
    )


def build_storage_for_s2f_scenario(scenario: StorageS2FScenario) -> Storage:
    return build_storage_from_scenario(scenario.storage)


def build_model_for_s2f_scenario(scenario: StorageS2FScenario) -> SchwartzTwoFactorModel:
    curve_times = [
        day_number(scenario.storage.start_date, curve_date)
        for curve_date, _ in scenario.curve_points
    ]
    curve_values = [value for _, value in scenario.curve_points]

    return SchwartzTwoFactorModel(
        calibration_date=0.0,
        curve_times=curve_times,
        curve_values=curve_values,
        rate=scenario.rate_annual / 365.0,
        short_term_mean_reversion=scenario.short_term_mean_reversion_annual / 365.0,
        short_term_vol=scenario.short_term_vol_annual / math.sqrt(365.0),
        long_term_drift=scenario.long_term_drift_annual / 365.0,
        long_term_vol=scenario.long_term_vol_annual / math.sqrt(365.0),
        rho=scenario.rho,
        asset_id=scenario.storage.asset_id,
    )


STORAGE1 = StorageScenario(
    name="storage1",
    asset_id="thegasprice",
    start_date=d(2024, 12, 1),
    end_date=d(2025, 1, 31),
    initial_amount=0.0,
    volume_constraints=((d(2024, 12, 1), d(2025, 2, 1), 0.0, 90.0),),
    injection_rates=((d(2024, 12, 1), d(2025, 2, 1), 0.0, 90.0),),
    withdrawal_rates=((d(2024, 12, 1), d(2025, 2, 1), 0.0, 90.0),),
    injection_cost=0.2,
    withdrawal_cost=0.0,
    penalty=0.0,
    num_states=10,
)

STORAGE2 = StorageScenario(
    name="storage2",
    asset_id="thegasprice",
    start_date=d(2025, 1, 1),
    end_date=d(2026, 3, 31),
    initial_amount=0.0,
    volume_constraints=(
        (d(2025, 1, 1), d(2025, 7, 1), 0.0, 200000.0),
        (d(2025, 7, 1), d(2025, 10, 1), 50000.0, 260000.0),
        (d(2025, 10, 1), d(2026, 1, 1), 180000.0, 280000.0),
        (d(2026, 1, 1), d(2026, 3, 1), 40000.0, 260000.0),
        (d(2026, 3, 1), d(2026, 4, 1), 0.0, 260000.0),
    ),
    injection_rates=(
        (d(2025, 1, 1), d(2025, 10, 1), 0.0, 3400.0),
        (d(2025, 1, 1), d(2025, 10, 1), 60000.0, 2920.0),
        (d(2025, 1, 1), d(2025, 10, 1), 150000.0, 2200.0),
        (d(2025, 1, 1), d(2025, 10, 1), 225000.0, 1480.0),
        (d(2025, 10, 1), d(2026, 4, 1), 0.0, 5800.0),
        (d(2025, 10, 1), d(2026, 4, 1), 60000.0, 4840.0),
        (d(2025, 10, 1), d(2026, 4, 1), 150000.0, 3400.0),
        (d(2025, 10, 1), d(2026, 4, 1), 225000.0, 1960.0),
    ),
    withdrawal_rates=(
        (d(2025, 1, 1), d(2025, 10, 1), 0.0, 1720.0),
        (d(2025, 1, 1), d(2025, 10, 1), 60000.0, 2800.0),
        (d(2025, 1, 1), d(2025, 10, 1), 150000.0, 3880.0),
        (d(2025, 1, 1), d(2025, 10, 1), 225000.0, 4600.0),
        (d(2025, 10, 1), d(2026, 4, 1), 0.0, 2200.0),
        (d(2025, 10, 1), d(2026, 4, 1), 60000.0, 4000.0),
        (d(2025, 10, 1), d(2026, 4, 1), 150000.0, 5800.0),
        (d(2025, 10, 1), d(2026, 4, 1), 225000.0, 7000.0),
    ),
    injection_cost=0.35,
    withdrawal_cost=0.12,
    penalty=0.0,
    num_states=10,
)


STORAGE_S2F_SCENARIOS = (
    StorageS2FScenario(
        name=STORAGE1.name,
        storage=STORAGE1,
        curve_points=(
            (STORAGE1.start_date, 100.0),
            (curve_date(STORAGE1.start_date, STORAGE1.end_date, 0.25), 100.0),
            (curve_date(STORAGE1.start_date, STORAGE1.end_date, 0.55), 110.0),
            (STORAGE1.end_date, 112.0),
        ),
    ),
    StorageS2FScenario(
        name=STORAGE2.name,
        storage=STORAGE2,
        curve_points=(
            (STORAGE2.start_date, 90.0),
            (d(2025, 4, 1), 94.0),
            (d(2025, 7, 1), 88.0),
            (d(2025, 10, 1), 96.0),
            (d(2026, 1, 1), 104.0),
            (STORAGE2.end_date, 98.0),
        ),
        short_term_mean_reversion_annual=1.5,
        short_term_vol_annual=0.18,
        long_term_vol_annual=0.08,
    ),
)
