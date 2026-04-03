from context import *
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pv_tests")))

from common.enums import SimulationScheme
from common.packages import FLOAT, device
from controller.controller import SimulationController
from products.netting_set import NettingSet
from helpers.cs_helper import CSHelper
from metrics.cva_metric import CVAMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes_multi import BlackScholesMulti
from models.cirpp import CIRPPModel
from models.model_config import ModelConfig
from pv_performance_large_netting_set import ( 
    build_correlation_matrix,
    build_mixed_book,
    synchronize_device,
)


COUNTERPARTY_ID = "mixed_book_counterparty"
DEFAULT_HAZARD_RATES: dict[float, float] = {
    0.5: 0.006402303360855854,
    1.0: 0.01553038972325307,
    2.0: 0.009729741230773657,
    3.0: 0.015552544648116201,
    4.0: 0.021196186202801115,
    5.0: 0.02284319986706472,
    7.0: 0.010111423894480876,
    10.0: 0.00613267811172937,
    15.0: 0.0036969930706003337,
    20.0: 0.003791311459217732,
}


def build_default_probability_profile(
    hazard_rates: dict[float, float],
    exposure_timeline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cs_helper = CSHelper()
    tenors = torch.tensor(list(hazard_rates.keys()), dtype=FLOAT, device=device)
    hazards = torch.tensor(list(hazard_rates.values()), dtype=FLOAT, device=device)

    cumulative_default_probability = np.array(
        [
            float(
                cs_helper.probability_of_default(
                    hazards=hazards,
                    tenors=tenors,
                    date=torch.tensor(t, dtype=FLOAT, device=device),
                )
            )
            for t in exposure_timeline
        ]
    )
    default_increment = np.diff(cumulative_default_probability, prepend=0.0)
    return cumulative_default_probability, default_increment


def run_cva_benchmark(
    num_assets: int = 4,
    num_european: int = 3940,
    num_binary: int = 100,
    num_basket: int = 100,
    num_asian: int = 200,
    num_barrier: int = 400,
    num_american: int = 180,
    num_flexicall: int = 70,
    num_storage: int = 10,
    num_paths_main: int = 1000,
    num_paths_pre: int = 1000,
    num_steps: int = 1,
    num_exposure_points: int = 80,
    deterministic_credit: bool = False,
    market_credit_correlation: float = 0.0,
    recovery_rate: float = 0.4,
):
    asset_ids = [f"asset_{idx}" for idx in range(num_assets)]
    market_correlation_matrix = build_correlation_matrix(num_assets=num_assets, rho=0.35)
    market_model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.03,
        asset_ids=asset_ids,
        spots=[95.0 + 7.5 * idx for idx in range(num_assets)],
        volatilities=[0.18 + 0.03 * idx for idx in range(num_assets)],
        correlation_matrix=market_correlation_matrix,
    )
    credit_model = CIRPPModel(
        calibration_date=0.0,
        asset_id=COUNTERPARTY_ID,
        hazard_rates=DEFAULT_HAZARD_RATES,
        kappa=0.10,
        theta=0.01,
        volatility=0.02,
        y0=0.0001,
        deterministic=deterministic_credit,
    )
    model = ModelConfig(
        models=[market_model, credit_model],
        inter_asset_correlation_matrix=[
            np.full((num_assets, 1), market_credit_correlation, dtype=float),
        ],
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
    exposure_horizon = max(float(product.modeling_timeline[-1].item()) for product in products)
    exposure_timeline = np.linspace(0.0, exposure_horizon, num_exposure_points)
    mpor_days = 10
    mpor_years = mpor_days / 252
    netting_set = NettingSet(
        name="mixed_state_dependent_book_cva",
        products=products,
        counterparty_id=COUNTERPARTY_ID,
        margin_period_of_risk=mpor_years,
    )
    cva_metric = CVAMetric(counterparty_id=COUNTERPARTY_ID, recovery_rate=recovery_rate)
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[cva_metric],
            exposure_timeline=exposure_timeline,
        ),
        num_paths_mainsim=num_paths_main,
        num_paths_presim=num_paths_pre,
        num_steps=num_steps,
        simulation_scheme=SimulationScheme.EULER,
        differentiate=False,
    )

    synchronize_device()
    started = time.perf_counter()
    results = controller.run_simulation()
    synchronize_device()
    elapsed = time.perf_counter() - started

    cva = float(results.get_results(netting_set.get_name(), cva_metric.get_name(), evaluation_idx=0))
    mc_error = float(results.get_mc_error(netting_set.get_name(), cva_metric.get_name(), evaluation_idx=0))
    cumulative_default_probability, default_increment = build_default_probability_profile(
        DEFAULT_HAZARD_RATES,
        exposure_timeline,
    )

    summary = {
        "device": str(device),
        "num_assets": num_assets,
        "num_products": len(products),
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
        "num_exposure_points": num_exposure_points,
        "exposure_horizon": exposure_horizon,
        "timeline_size": int(controller.simulation_timeline.numel()),
        "deterministic_credit": deterministic_credit,
        "market_credit_correlation": market_credit_correlation,
        "recovery_rate": recovery_rate,
        "total_seconds": elapsed,
        "products_per_second": len(products) / elapsed,
        "cva": cva,
        "mc_error": mc_error,
        "exposure_timeline": exposure_timeline,
        "cumulative_default_probability": cumulative_default_probability,
        "default_increment": default_increment,
        **profile,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_cva_benchmark(deterministic_credit=False, market_credit_correlation=0.2)

    lines = [
        "Mixed State-Dependent Netting Set CVA Benchmark",
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
        f"num_exposure_points: {summary['num_exposure_points']}",
        f"exposure_horizon: {summary['exposure_horizon']}",
        f"timeline_size: {summary['timeline_size']}",
        f"deterministic_credit: {summary['deterministic_credit']}",
        f"market_credit_correlation: {summary['market_credit_correlation']}",
        f"recovery_rate: {summary['recovery_rate']}",
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
        f"cva: {summary['cva']:.6f}",
        f"mc_error: {summary['mc_error']:.6f}",
        f"total_seconds: {summary['total_seconds']:.6f}",
        f"products_per_second: {summary['products_per_second']:.2f}",
        "phase_timings: see controller logger output",
    ]
    output = "\n".join(lines)
    print(output)

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(
        summary["exposure_timeline"],
        summary["default_increment"],
        width=summary["exposure_horizon"] / max(1, summary["num_exposure_points"]) * 0.9,
        color="darkred",
        alpha=0.6,
        label="Default increment",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Default increment")
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(
        summary["exposure_timeline"],
        summary["cumulative_default_probability"],
        color="navy",
        linewidth=2.0,
        label="Cumulative default probability",
    )
    ax2.set_ylabel("Cumulative default probability")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    plt.title(
        "Mixed State-Dependent Netting Set CVA Benchmark\n"
        f"CVA={summary['cva']:.4f}, MC error={summary['mc_error']:.4f}"
    )
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "perf_mixed_state_dependent_netting_set_cva.png")
    plt.savefig(plot_path)

    summary_path = os.path.join(out_dir, "perf_mixed_state_dependent_netting_set_cva.txt")
    with open(summary_path, "w", encoding="ascii") as fh:
        fh.write(output + "\n")

    print(f"Plot saved to {plot_path}")
    print(f"Wrote benchmark summary to {summary_path}")
