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
from metrics.metric import Metric
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes_multi import BlackScholesMulti
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType


def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_correlation_matrix(num_assets: int, rho: float) -> np.ndarray:
    correlation_matrix = np.full((num_assets, num_assets), rho, dtype=float)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix


def build_european_option_book(
    num_options: int,
    asset_ids: list[str],
    maturities: list[float],
    strikes: list[float],
):
    products = []
    for idx in range(num_options):
        asset_id = asset_ids[idx % len(asset_ids)]
        maturity = maturities[idx % len(maturities)]
        strike = strikes[idx % len(strikes)]
        products.append(
            EuropeanOption(
                underlying=Equity(asset_id),
                exercise_date=maturity,
                strike=strike,
                option_type=OptionType.CALL,
                asset_id=asset_id,
            )
        )
    return products


def run_benchmark(
    num_options: int = 100_000,
    num_assets: int = 8,
    num_paths: int = 100_000,
    num_steps: int = 1,
):
    asset_ids = [f"asset_{idx}" for idx in range(num_assets)]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    spots = [100.0 + 2.0 * idx for idx in range(num_assets)]
    volatilities = [0.20 + 0.01 * (idx % 4) for idx in range(num_assets)]
    correlation_matrix = build_correlation_matrix(num_assets=num_assets, rho=0.35)

    model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.03,
        asset_ids=asset_ids,
        spots=spots,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
    )
    products = build_european_option_book(
        num_options=num_options,
        asset_ids=asset_ids,
        maturities=maturities,
        strikes=strikes,
    )
    netting_set = NettingSet(name="bs_multi_european_option_book", products=products)
    pv_metric = PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=num_paths,
        num_paths_presim=0,
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

    summary = {
        "device": str(device),
        "num_options": num_options,
        "num_assets": num_assets,
        "num_paths": num_paths,
        "num_steps": num_steps,
        "timeline_size": int(controller.simulation_timeline.numel()),
        "evaluation_type": pv_metric.evaluation_type.name.lower(),
        "pv": pv,
        "mc_error": mc_error,
        "total_seconds": elapsed,
        "options_per_second": num_options / elapsed,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_benchmark()

    lines = [
        "Black-Scholes Multi European Option Book Benchmark",
        f"device: {summary['device']}",
        f"num_options: {summary['num_options']}",
        f"num_assets: {summary['num_assets']}",
        f"num_paths: {summary['num_paths']}",
        f"num_steps: {summary['num_steps']}",
        f"timeline_size: {summary['timeline_size']}",
        f"evaluation_type: {summary['evaluation_type']}",
        "",
        "Controller run:",
        f"pv: {summary['pv']:.6f}",
        f"mc_error: {summary['mc_error']:.6f}",
        f"total_seconds: {summary['total_seconds']:.6f}",
        f"options_per_second: {summary['options_per_second']:.2f}",
        "phase_timings: see controller logger output",
    ]
    output = "\n".join(lines)
    print(output)

    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perf_bs_multi_european_option_book.txt")
    with open(out_path, "w", encoding="ascii") as fh:
        fh.write(output + "\n")
    print(f"Wrote benchmark summary to {out_path}")
