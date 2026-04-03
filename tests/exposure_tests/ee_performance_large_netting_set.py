import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pv_tests")))

from common.enums import SimulationScheme
from common.packages import device
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.epe_metric import EPEMetric
from metrics.pfe_metric import PFEMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes_multi import BlackScholesMulti
from pv_performance_large_netting_set import (  # noqa: E402
    build_correlation_matrix,
    build_mixed_book,
    synchronize_device,
)


def run_exposure_benchmark(
    num_assets: int = 4,
    num_european: int = 3940,
    num_binary: int = 100,
    num_basket: int = 100,
    num_asian: int = 200,
    num_barrier: int = 400,
    num_american: int = 180,
    num_flexicall: int = 70,
    num_storage: int = 10,
    num_paths_main: int = 1_000,
    num_paths_pre: int = 1_000,
    num_steps: int = 1,
    num_exposure_points: int = 80,
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
    exposure_horizon = max(float(product.modeling_timeline[-1].item()) for product in products)
    exposure_timeline = np.linspace(0.0, exposure_horizon, num_exposure_points)

    netting_set = NettingSet(name="mixed_state_dependent_book", products=products)
    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.95)
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[epe_metric, pfe_metric],
            exposure_timeline=exposure_timeline,
        ),
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

    epe = results.get_results(netting_set.get_name(), epe_metric.get_name())
    pfe = results.get_results(netting_set.get_name(), pfe_metric.get_name())

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
        "total_seconds": elapsed,
        "products_per_second": len(products) / elapsed,
        "peak_epe": float(np.max(epe)),
        "peak_pfe": float(np.max(pfe)),
        "final_epe": float(epe[-1]),
        "final_pfe": float(pfe[-1]),
        "exposure_timeline": exposure_timeline,
        "epe": epe,
        "pfe": pfe,
        **profile,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_exposure_benchmark()

    lines = [
        "Mixed State-Dependent Netting Set Exposure Benchmark",
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
        f"peak_epe: {summary['peak_epe']:.6f}",
        f"peak_pfe: {summary['peak_pfe']:.6f}",
        f"final_epe: {summary['final_epe']:.6f}",
        f"final_pfe: {summary['final_pfe']:.6f}",
        f"total_seconds: {summary['total_seconds']:.6f}",
        f"products_per_second: {summary['products_per_second']:.2f}",
        "phase_timings: see controller logger output",
    ]
    output = "\n".join(lines)
    print(output)

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(11, 6))
    plt.plot(
        summary["exposure_timeline"],
        summary["epe"],
        label="Expected Exposure (EPE)",
        color="darkred",
    )
    plt.plot(
        summary["exposure_timeline"],
        summary["pfe"],
        label="Potential Future Exposure (PFE 95%)",
        color="navy",
        linestyle="--",
    )
    plt.xlabel("Time")
    plt.ylabel("Exposure")
    plt.title("Mixed State-Dependent Netting Set Exposure")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "perf_mixed_state_dependent_netting_set_exposure.png")
    plt.savefig(plot_path)

    summary_path = os.path.join(out_dir, "perf_mixed_state_dependent_netting_set_exposure.txt")
    with open(summary_path, "w", encoding="ascii") as fh:
        fh.write(output + "\n")

    print(f"Plot saved to {plot_path}")
    print(f"Wrote benchmark summary to {summary_path}")
