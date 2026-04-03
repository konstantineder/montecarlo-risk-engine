from context import *

import pytest
import math

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.metric import Metric
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes import BlackScholesModel
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType


def test_simulation_results_named_access_for_duplicate_products():
    model = BlackScholesModel(0.0, 100.0, 0.05, 0.3)
    product_1 = EuropeanOption(Equity("id_1"), 2.0, 100.0, OptionType.CALL)
    product_2 = EuropeanOption(Equity("id_2"), 2.0, 120.0, OptionType.CALL)

    controller = SimulationController(
        netting_sets=[
            NettingSet(name=product_1.get_name(), products=[product_1]),
            NettingSet(name=product_2.get_name(), products=[product_2]),
        ],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)]
        ),
        num_paths_mainsim=1,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=True,
    )

    results = controller.run_simulation()

    assert results.get_product_names() == ["EuropeanOption", "EuropeanOption#2"]
    assert results.get_metric_names() == ["pv"]

    pv_1_named = float(results.get_results("EuropeanOption", "pv", evaluation_idx=0))
    pv_2_named = float(results.get_results("EuropeanOption#2", "pv", evaluation_idx=0))

    assert pv_1_named != pv_2_named

    vega_named = float(results.get_derivatives("EuropeanOption", "pv", "volatility", evaluation_idx=0))
    assert math.isfinite(vega_named)


def test_simulation_results_accepts_netting_set_and_metric_keywords():
    model = BlackScholesModel(0.0, 100.0, 0.05, 0.3)
    product = EuropeanOption(Equity("id_1"), 2.0, 100.0, OptionType.CALL)

    controller = SimulationController(
        netting_sets=[NettingSet(name=product.get_name(), products=[product])],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)]
        ),
        num_paths_mainsim=1,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=True,
    )

    results = controller.run_simulation()

    pv_new = float(
        results.get_results(
            netting_set=product.get_name(),
            metric="pv",
            evaluation_idx=0,
        )
    )
    pv_old = float(
        results.get_results(
            prod_idx=product.get_name(),
            metric_idx="pv",
            evaluation_index=0,
        )
    )
    vega = float(
        results.get_derivatives(
            netting_set=product.get_name(),
            metric="pv",
            param="volatility",
            evaluation_idx=0,
        )
    )

    assert pv_new == pv_old
    assert math.isfinite(vega)
