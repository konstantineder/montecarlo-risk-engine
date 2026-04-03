import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from common.enums import SimulationScheme
from common.packages import device
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.pv_metric import PVMetric, Metric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes_multi import BlackScholesMulti
from products.asian_option import AsianAveragingType, AsianOption
from products.barrier_option import BarrierOption, BarrierOptionType
from products.basket_option import BasketOption, BasketOptionType
from products.bermudan_option import AmericanOption
from products.binary_option import BinaryOption
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType
from products.flexicall import FlexiCall
from products.storage import Storage
from products.storage_helpers import StorageConfig


def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_correlation_matrix(num_assets: int, rho: float) -> np.ndarray:
    correlation_matrix = np.full((num_assets, num_assets), rho, dtype=float)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix


def build_storage_product(
    asset_id: str,
    maturity: float,
    capacity: float,
    initial_amount: float,
    injection_cost: float,
    withdrawal_cost: float,
    num_states: int,
    rollout_interval: float,
) -> Storage:
    storage_config = StorageConfig()
    ramp_up_end = 0.35 * maturity
    plateau_end = 0.70 * maturity
    storage_config.add_volume_constraint(0.0, ramp_up_end, 0.0, 0.55 * capacity, 0.0)
    storage_config.add_volume_constraint(ramp_up_end, plateau_end, 0.10 * capacity, 0.85 * capacity, 0.0)
    storage_config.add_volume_constraint(plateau_end, maturity, 0.0, capacity, 0.0)

    storage_config.add_injection_flexibility(0.0, ramp_up_end, 0.0, 0.30 * capacity)
    storage_config.add_injection_flexibility(0.0, ramp_up_end, 0.60 * capacity, 0.18 * capacity)
    storage_config.add_injection_flexibility(ramp_up_end, maturity, 0.0, 0.22 * capacity)
    storage_config.add_injection_flexibility(ramp_up_end, maturity, 0.60 * capacity, 0.12 * capacity)

    storage_config.add_withdrawal_flexibility(0.0, plateau_end, 0.0, 0.16 * capacity)
    storage_config.add_withdrawal_flexibility(0.0, plateau_end, 0.60 * capacity, 0.24 * capacity)
    storage_config.add_withdrawal_flexibility(plateau_end, maturity, 0.0, 0.24 * capacity)
    storage_config.add_withdrawal_flexibility(plateau_end, maturity, 0.60 * capacity, 0.32 * capacity)

    storage_config.add_variable_injection_cost(0.0, injection_cost)
    storage_config.add_variable_injection_cost(plateau_end, injection_cost * 1.10)
    storage_config.add_variable_withdrawal_cost(0.0, withdrawal_cost)
    storage_config.add_variable_withdrawal_cost(plateau_end, withdrawal_cost * 1.10)

    return Storage(
        asset_id=asset_id,
        start_date=0.0,
        end_date=maturity,
        initial_amount=initial_amount,
        storage_config=storage_config,
        num_states=num_states,
        rollout_interval=rollout_interval,
    )


def build_mixed_book(
    asset_ids: list[str],
    num_european: int,
    num_binary: int,
    num_basket: int,
    num_asian: int,
    num_barrier: int,
    num_american: int,
    num_flexicall: int,
    num_storage: int,
):
    products = []

    european_maturities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    european_strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    for idx in range(num_european):
        asset_id = asset_ids[idx % len(asset_ids)]
        option_type = OptionType.CALL if idx % 2 == 0 else OptionType.PUT
        products.append(
            EuropeanOption(
                underlying=Equity(asset_id),
                exercise_date=european_maturities[idx % len(european_maturities)],
                strike=european_strikes[idx % len(european_strikes)],
                option_type=option_type,
                asset_id=asset_id,
            )
        )

    binary_maturities = [0.5, 1.0, 1.5, 2.0]
    binary_strikes = [90.0, 100.0, 110.0]
    for idx in range(num_binary):
        asset_id = asset_ids[idx % len(asset_ids)]
        product = BinaryOption(
            maturity=binary_maturities[idx % len(binary_maturities)],
            strike=binary_strikes[idx % len(binary_strikes)],
            payment_amount=8.0 + 2.0 * (idx % 4),
            option_type=OptionType.CALL if idx % 2 == 0 else OptionType.PUT,
            asset_id=asset_id,
        )
        product.name = f"binary_{idx}"
        products.append(product)

    basket_maturities = [0.75, 1.25, 2.0, 2.5]
    basket_weights = [
        [0.5, 0.3, 0.2, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.35, 0.15, 0.10],
    ]
    for idx in range(num_basket):
        active_asset_count = 2 + (idx % min(3, len(asset_ids) - 1 if len(asset_ids) > 1 else 1))
        basket_asset_ids = asset_ids[:active_asset_count]
        base_weights = basket_weights[idx % len(basket_weights)][:active_asset_count]
        norm = sum(base_weights)
        product = BasketOption(
            maturity=basket_maturities[idx % len(basket_maturities)],
            asset_ids=basket_asset_ids,
            weights=[weight / norm for weight in base_weights],
            strike=95.0 + 5.0 * (idx % 5),
            option_type=OptionType.CALL if idx % 2 == 0 else OptionType.PUT,
            basket_option_type=(
                BasketOptionType.ARITHMETIC
                if idx % 3 != 0
                else BasketOptionType.GEOMETRIC
            ),
            use_variation_reduction=False,
        )
        product.name = f"basket_{idx}"
        products.append(product)

    asian_maturities = [0.5, 0.75, 1.0, 1.5, 2.0]
    asian_observation_counts = [8, 12, 18, 24]
    for idx in range(num_asian):
        asset_id = asset_ids[idx % len(asset_ids)]
        product = AsianOption(
            startdate=0.0,
            maturity=asian_maturities[idx % len(asian_maturities)],
            strike=88.0 + 6.0 * (idx % 6),
            num_observation_timepoints=asian_observation_counts[idx % len(asian_observation_counts)],
            option_type=OptionType.CALL if idx % 2 == 0 else OptionType.PUT,
            averaging_type=(
                AsianAveragingType.ARITHMETIC
                if idx % 3 != 0
                else AsianAveragingType.GEOMETRIC
            ),
            asset_id=asset_id,
        )
        product.name = f"asian_{idx}"
        products.append(product)

    barrier_maturities = [0.5, 0.75, 1.25, 1.75, 2.5, 3.0]
    barrier_levels = [118.0, 125.0, 132.0, 140.0]
    observation_counts = [8, 12, 18, 24, 36]
    for idx in range(num_barrier):
        asset_id = asset_ids[idx % len(asset_ids)]
        product = BarrierOption(
            startdate=0.0,
            maturity=barrier_maturities[idx % len(barrier_maturities)],
            strike=85.0 + 7.5 * (idx % 6),
            num_observation_timepoints=observation_counts[idx % len(observation_counts)],
            option_type=OptionType.CALL if idx % 3 != 0 else OptionType.PUT,
            barrier1=barrier_levels[idx % len(barrier_levels)] + 2.0 * (idx % 2),
            barrier_option_type1=BarrierOptionType.UPANDOUT,
            asset_id=asset_id,
        )
        product.name = f"barrier_{idx}"
        products.append(product)

    american_maturities = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    exercise_dates = [8, 12, 18, 24, 36, 48]
    american_strikes = [80.0, 92.5, 100.0, 107.5, 120.0]
    for idx in range(num_american):
        asset_id = asset_ids[idx % len(asset_ids)]
        product = AmericanOption(
            underlying=Equity(asset_id),
            maturity=american_maturities[idx % len(american_maturities)],
            num_exercise_dates=exercise_dates[idx % len(exercise_dates)],
            strike=american_strikes[idx % len(american_strikes)],
            option_type=OptionType.PUT if idx % 2 == 0 else OptionType.CALL,
            asset_id=asset_id,
        )
        product.name = f"american_{idx}"
        products.append(product)

    flex_maturities = [1.0, 1.5, 2.0, 2.5]
    flex_schedule_lengths = [3, 4, 5]
    for idx in range(num_flexicall):
        asset_id = asset_ids[idx % len(asset_ids)]
        maturity = flex_maturities[idx % len(flex_maturities)]
        schedule_length = flex_schedule_lengths[idx % len(flex_schedule_lengths)]
        exercise_dates = np.linspace(maturity / schedule_length, maturity, schedule_length)
        underlyings = []
        for exercise_idx, exercise_date in enumerate(exercise_dates):
            underlyings.append(
                EuropeanOption(
                    underlying=Equity(asset_id),
                    exercise_date=float(exercise_date),
                    strike=90.0 + 6.0 * ((idx + exercise_idx) % 6),
                    option_type=OptionType.CALL,
                    asset_id=asset_id,
                )
            )
        product = FlexiCall(
            underlyings=underlyings,
            num_exercise_rights=min(1 + (idx % 3), schedule_length - 1),
            asset_id=asset_id,
        )
        product.name = f"flexicall_{idx}"
        products.append(product)

    storage_maturities = [1.0, 1.5, 2.0, 2.5]
    capacities = [18.0, 26.0, 34.0, 42.0]
    rollout_intervals = [0.05, 0.10, 0.125]
    for idx in range(num_storage):
        asset_id = asset_ids[idx % len(asset_ids)]
        product = build_storage_product(
            asset_id=asset_id,
            maturity=storage_maturities[idx % len(storage_maturities)],
            capacity=capacities[idx % len(capacities)],
            initial_amount=2.0 + 0.5 * (idx % 5),
            injection_cost=0.10 + 0.02 * (idx % 4),
            withdrawal_cost=0.08 + 0.015 * (idx % 4),
            num_states=6 + (idx % 5),
            rollout_interval=rollout_intervals[idx % len(rollout_intervals)],
        )
        product.name = f"storage_{idx}"
        products.append(product)

    profile = {
        "european_maturities": len(european_maturities),
        "binary_maturities": len(binary_maturities),
        "basket_maturities": len(basket_maturities),
        "asian_maturities": len(asian_maturities),
        "barrier_maturities": len(barrier_maturities),
        "american_maturities": len(american_maturities),
        "flex_schedule_lengths": ",".join(str(x) for x in flex_schedule_lengths),
        "storage_maturities": len(storage_maturities),
    }
    return products, profile


def run_benchmark(
    num_assets: int = 4,
    num_european: int = 39_400,
    num_binary: int = 1_000,
    num_basket: int = 1_000,
    num_asian: int = 2_000,
    num_barrier: int = 4_000,
    num_american: int = 1_800,
    num_flexicall: int = 700,
    num_storage: int = 100,
    num_paths_main: int = 1_000,
    num_paths_pre: int = 1_000,
    num_steps: int = 1,
):
    asset_ids = [f"asset_{idx}" for idx in range(num_assets)]
    correlation_matrix = build_correlation_matrix(num_assets=num_assets, rho=0.35)
    model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.03,
        asset_ids=asset_ids,
        spots=[95.0 + 7.5 * idx for idx in range(num_assets)],
        volatilities=[0.18 + 0.03 * idx for idx in range(num_assets)],
        correlation_matrix=correlation_matrix,
    )
    products, profile = build_mixed_book(
        asset_ids=asset_ids,
        num_european=num_european,
        num_binary=num_binary,
        num_basket=num_basket,
        num_asian=num_asian,
        num_barrier=num_barrier,
        num_american=num_american,
        num_flexicall=num_flexicall,
        num_storage=num_storage,
    )
    netting_set = NettingSet(name="mixed_state_dependent_book", products=products)
    pv_metric = PVMetric()
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=num_paths_main,
        num_paths_presim=num_paths_pre,
        num_steps=num_steps,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    synchronize_device()
    started = time.perf_counter()
    results = controller.run_simulation()
    synchronize_device()
    elapsed = time.perf_counter() - started

    pv = float(results.get_results(netting_set.get_name(), pv_metric.get_name(), evaluation_idx=0))
    mc_error = float(results.get_mc_error(netting_set.get_name(), pv_metric.get_name(), evaluation_idx=0))
    num_products = len(products)

    summary = {
        "device": str(device),
        "num_assets": num_assets,
        "num_products": num_products,
        "num_european": num_european,
        "num_binary": num_binary,
        "num_basket": num_basket,
        "num_asian": num_asian,
        "num_barrier": num_barrier,
        "num_american": num_american,
        "num_flexicall": num_flexicall,
        "num_storage": num_storage,
        "num_paths_main": num_paths_main,
        "num_paths_pre": num_paths_pre,
        "num_steps": num_steps,
        "timeline_size": int(controller.simulation_timeline.numel()),
        "pv": pv,
        "mc_error": mc_error,
        "total_seconds": elapsed,
        "products_per_second": num_products / elapsed,
        **profile,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_benchmark()

    lines = [
        "Mixed State-Dependent Netting Set Benchmark",
        f"device: {summary['device']}",
        f"num_assets: {summary['num_assets']}",
        f"num_products: {summary['num_products']}",
        f"num_european: {summary['num_european']}",
        f"num_binary: {summary['num_binary']}",
        f"num_basket: {summary['num_basket']}",
        f"num_asian: {summary['num_asian']}",
        f"num_barrier: {summary['num_barrier']}",
        f"num_american: {summary['num_american']}",
        f"num_flexicall: {summary['num_flexicall']}",
        f"num_storage: {summary['num_storage']}",
        f"num_paths_main: {summary['num_paths_main']}",
        f"num_paths_pre: {summary['num_paths_pre']}",
        f"num_steps: {summary['num_steps']}",
        f"timeline_size: {summary['timeline_size']}",
        f"distinct_european_maturities: {summary['european_maturities']}",
        f"distinct_binary_maturities: {summary['binary_maturities']}",
        f"distinct_basket_maturities: {summary['basket_maturities']}",
        f"distinct_asian_maturities: {summary['asian_maturities']}",
        f"distinct_barrier_maturities: {summary['barrier_maturities']}",
        f"distinct_american_maturities: {summary['american_maturities']}",
        f"flexicall_schedule_lengths: {summary['flex_schedule_lengths']}",
        f"distinct_storage_maturities: {summary['storage_maturities']}",
        "",
        "Controller run:",
        f"pv: {summary['pv']:.6f}",
        f"mc_error: {summary['mc_error']:.6f}",
        f"total_seconds: {summary['total_seconds']:.6f}",
        f"products_per_second: {summary['products_per_second']:.2f}",
        "phase_timings: see controller logger output",
    ]
    output = "\n".join(lines)
    print(output)

    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perf_mixed_state_dependent_netting_set.txt")
    with open(out_path, "w", encoding="ascii") as fh:
        fh.write(output + "\n")
    print(f"Wrote benchmark summary to {out_path}")
