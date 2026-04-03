from context import *

import pytest
import torch

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.metric import Metric
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes import BlackScholesModel
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType


PARAM_ORDER = ("spot", "volatility", "rate")
SECOND_DERIVATIVE_PAIRS = [
    ("spot", "spot"),
    ("spot", "volatility"),
    ("spot", "rate"),
    ("volatility", "volatility"),
    ("volatility", "rate"),
    ("rate", "rate"),
]


def analytical_hessian(
    spot: float,
    rate: float,
    sigma: float,
    maturity: float,
    strike: float,
):
    strike_tensor = torch.tensor(strike, dtype=torch.float64)
    maturity_tensor = torch.tensor(maturity, dtype=torch.float64)
    params = torch.tensor([spot, sigma, rate], dtype=torch.float64, requires_grad=True)

    def pv_fn(x: torch.Tensor) -> torch.Tensor:
        spot_x = x[0]
        sigma_x = x[1]
        rate_x = x[2]
        sqrt_t = torch.sqrt(maturity_tensor)
        d1 = (
            torch.log(spot_x / strike_tensor)
            + (rate_x + 0.5 * sigma_x ** 2) * maturity_tensor
        ) / (sigma_x * sqrt_t)
        d2 = d1 - sigma_x * sqrt_t
        norm = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        return spot_x * norm.cdf(d1) - strike_tensor * torch.exp(-rate_x * maturity_tensor) * norm.cdf(d2)

    hessian = torch.autograd.functional.hessian(pv_fn, params)
    return {
        row_name: {
            col_name: float(hessian[row_idx, col_idx])
            for col_idx, col_name in enumerate(PARAM_ORDER)
        }
        for row_idx, row_name in enumerate(PARAM_ORDER)
    }


def test_controller_analytical_pv_second_derivatives_match_formula():
    model = BlackScholesModel(0.0, 100.0, 0.05, 0.3)
    product = EuropeanOption(Equity("id"), 2.0, 100.0, OptionType.CALL)

    controller = SimulationController(
        netting_sets=[NettingSet(name=product.get_name(), products=[product])],
        model=model,
        risk_metrics=RiskMetrics(metrics=[PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)]),
        num_paths_mainsim=1,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=True,
    )
    controller.compute_higher_derivatives()

    results = controller.run_simulation()
    assert results.get_product_names() == ["EuropeanOption"]
    assert results.get_metric_names() == ["pv"]
    assert results.get_model_param_names() == ["spot", "volatility", "rate"]

    controller_hessian = results.get_second_derivatives(
        "EuropeanOption",
        "pv",
        evaluation_idx=0,
    )

    expected_hessian = analytical_hessian(100.0, 0.05, 0.3, 2.0, 100.0)

    for param_1, param_2 in SECOND_DERIVATIVE_PAIRS:
        expected = expected_hessian[param_1][param_2]
        actual = float(controller_hessian[param_1][param_2])
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)

    assert float(controller_hessian["spot"]["spot"]) == pytest.approx(
        float(product.compute_dDeltadSpot_analytically(model)),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(controller_hessian["volatility"]["volatility"]) == pytest.approx(
        float(product.compute_dVegadSigma_analytically(model)),
        rel=1e-9,
        abs=1e-9,
    )
