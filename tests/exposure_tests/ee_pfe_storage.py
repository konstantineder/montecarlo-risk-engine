from context import *

from common.enums import SimulationScheme
from common.packages import device
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.epe_metric import EPEMetric
from metrics.pfe_metric import PFEMetric
from metrics.risk_metrics import RiskMetrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage_s2f_cases import (  # noqa: E402
    STORAGE_S2F_SCENARIOS,
    build_model_for_s2f_scenario,
    build_storage_for_s2f_scenario,
)


if __name__ == "__main__":
    print(f"Using device: {device}")

    scenario = STORAGE_S2F_SCENARIOS[5]
    model = build_model_for_s2f_scenario(scenario)
    product = build_storage_for_s2f_scenario(scenario)
    netting_set = NettingSet(name=scenario.name, products=[product])

    exposure_timeline = np.linspace(0.0, float(product.end_date), 60)
    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.95)
    risk_metrics = RiskMetrics(
        metrics=[epe_metric, pfe_metric],
        exposure_timeline=exposure_timeline,
    )

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=risk_metrics,
        num_paths_mainsim=5000,
        num_paths_presim=15000,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    sim_results = controller.run_simulation()
    ees = sim_results.get_results(netting_set.get_name(), epe_metric.get_name())
    pfes = sim_results.get_results(netting_set.get_name(), pfe_metric.get_name())

    plt.figure(figsize=(10, 6))
    plt.plot(exposure_timeline, ees, label="Expected Exposure (EPE)", color="darkred")
    plt.plot(
        exposure_timeline,
        pfes,
        label="Potential Future Exposure (PFE 95%)",
        color="navy",
        linestyle="--",
    )
    plt.xlabel("Time")
    plt.ylabel("Exposure")
    plt.title("Storage Exposure vs Time (S2F)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_storage_s2f.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
