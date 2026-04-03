from context import *

import os
import sys
from common.packages import device
from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage_s2f_cases import (
    STORAGE_S2F_SCENARIOS,
    build_model_for_s2f_scenario,
    build_storage_for_s2f_scenario,
)


if __name__ == "__main__":
    print(f"Using device: {device}")

    scenario = STORAGE_S2F_SCENARIOS[0]
    model = build_model_for_s2f_scenario(scenario)
    product = build_storage_for_s2f_scenario(scenario)
    netting_set = NettingSet(name=scenario.name, products=[product])
    pv_metric = PVMetric()
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=10000,
        num_paths_presim=25000,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    pv = controller.run_simulation().get_results(netting_set.get_name(), pv_metric.get_name(), evaluation_idx=0)
    print(f"Storage PV (S2F): {pv:.6f}")

    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "pv_storage_s2f.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"Storage PV (S2F): {pv:.6f}\n")

    print(f"Result saved to {out_path}")
