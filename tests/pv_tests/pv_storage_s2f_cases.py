from context import *

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from maths.regression import PolyomialRegression
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from storage_s2f_cases import (
    STORAGE_S2F_SCENARIOS,
    build_model_for_s2f_scenario,
    build_storage_for_s2f_scenario,
)


if __name__ == "__main__":
    print("Storage S2F scenario PVs:")
    for scenario in STORAGE_S2F_SCENARIOS:
        product = build_storage_for_s2f_scenario(scenario)
        model = build_model_for_s2f_scenario(scenario)
        netting_set = NettingSet(name=product.get_name(), products=[product])

        pv_metric = PVMetric()
        controller = SimulationController(
            netting_sets=[netting_set],
            model=model,
            risk_metrics=RiskMetrics(metrics=[pv_metric]),
            num_paths_mainsim=10000,
            num_paths_presim=10000,
            num_steps=1,
            simulation_scheme=SimulationScheme.ANALYTICAL,
            differentiate=False,
            regression_function=PolyomialRegression(degree=3),
        )

        pv = controller.run_simulation().get_results(netting_set.get_name(), pv_metric.get_name(), evaluation_idx=0)
        print(f"- {scenario.name}: {pv:.12f}")
