from context import *

import os
import sys

import pytest

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from maths.regression import PolyomialRegression
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage_s2f_cases import (  # noqa: E402
    STORAGE_S2F_SCENARIOS,
    build_model_for_s2f_scenario,
    build_storage_for_s2f_scenario,
)

EXPECTED_PVS = {
    "storage1": 1055.330006881181,
    "storage2": 3769746.378205333,
}


@pytest.mark.parametrize(
    "scenario",
    STORAGE_S2F_SCENARIOS,
    ids=[scenario.name for scenario in STORAGE_S2F_SCENARIOS],
)
def test_storage_s2f_pv_regression(scenario):
    product = build_storage_for_s2f_scenario(scenario)
    model = build_model_for_s2f_scenario(scenario)

    pv_metric = PVMetric()
    controller = SimulationController(
        netting_sets=[NettingSet(name=product.get_name(), products=[product])],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=2000,
        num_paths_presim=4000,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
        regression_function=PolyomialRegression(degree=3),
    )

    pv = controller.run_simulation().get_results(product.get_name(), pv_metric.get_name(), evaluation_idx=0)
    assert pv == pytest.approx(EXPECTED_PVS[scenario.name], abs=1e-6)
