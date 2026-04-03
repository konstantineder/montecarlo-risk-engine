from context import *

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import torch

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes import BlackScholesModel
from models.black_scholes_multi import BlackScholesMulti
from models.vasicek import VasicekModel
from products.asian_option import AsianAveragingType, AsianOption
from products.barrier_option import BarrierOption, BarrierOptionType
from products.basket_option import BasketOption, BasketOptionType
from products.bermudan_option import AmericanOption
from products.binary_option import BinaryOption
from products.bond import Bond
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType
from products.flexicall import FlexiCall
from products.storage import Storage
from products.storage_helpers import StorageConfig
from products.swap import IRSType, InterestRateSwap


@dataclass(frozen=True)
class SingleProductCase:
    name: str
    build_model: Callable[[], object]
    build_product: Callable[[], object]
    num_paths_main: int = 256
    num_paths_pre: int = 256
    pv_tol: float = 1e-8
    deriv_tol: float = 1e-7


def _build_storage_product() -> Storage:
    storage_config = StorageConfig()
    storage_config.add_volume_constraint(0.0, 2.0, 0.0, 6.0, 0.0)
    storage_config.add_injection_flexibility(0.0, 2.0, 0.0, 2.0)
    storage_config.add_withdrawal_flexibility(0.0, 2.0, 0.0, 2.0)
    storage_config.add_variable_injection_cost(0.0, 0.1)
    storage_config.add_variable_withdrawal_cost(0.0, 0.1)
    return Storage(
        asset_id="gas",
        start_date=0.0,
        end_date=2.0,
        initial_amount=1.0,
        storage_config=storage_config,
        num_states=4,
    )


SINGLE_PRODUCT_CASES = [
    SingleProductCase(
        name="european",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: EuropeanOption(Equity("asset"), 1.0, 100.0, OptionType.CALL, asset_id="asset"),
        num_paths_pre=0,
        pv_tol=1e-10,
        deriv_tol=1e-10,
    ),
    SingleProductCase(
        name="binary",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: BinaryOption(1.0, 100.0, 10.0, OptionType.CALL, asset_id="asset"),
        num_paths_pre=0,
        pv_tol=1e-10,
        deriv_tol=1e-10,
    ),
    SingleProductCase(
        name="basket",
        build_model=lambda: BlackScholesMulti(
            calibration_date=0.0,
            rate=0.03,
            asset_ids=["asset_1", "asset_2"],
            spots=[100.0, 105.0],
            volatilities=[0.20, 0.24],
            correlation_matrix=np.array([[1.0, 0.35], [0.35, 1.0]]),
        ),
        build_product=lambda: BasketOption(
            maturity=1.0,
            asset_ids=["asset_1", "asset_2"],
            weights=[0.55, 0.45],
            strike=100.0,
            option_type=OptionType.CALL,
            basket_option_type=BasketOptionType.ARITHMETIC,
            use_variation_reduction=False,
        ),
        num_paths_pre=0,
        pv_tol=1e-9,
        deriv_tol=5e-7,
    ),
    SingleProductCase(
        name="barrier",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: BarrierOption(
            startdate=0.0,
            maturity=1.0,
            strike=100.0,
            num_observation_timepoints=10,
            option_type=OptionType.CALL,
            barrier1=130.0,
            barrier_option_type1=BarrierOptionType.UPANDOUT,
            asset_id="asset",
        ),
        num_paths_pre=256,
        pv_tol=1e-8,
        deriv_tol=5e-7,
    ),
    SingleProductCase(
        name="asian",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: AsianOption(
            startdate=0.0,
            maturity=1.0,
            strike=100.0,
            num_observation_timepoints=10,
            option_type=OptionType.CALL,
            averaging_type=AsianAveragingType.ARITHMETIC,
            asset_id="asset",
        ),
        num_paths_pre=256,
        pv_tol=1e-8,
        deriv_tol=5e-7,
    ),
    SingleProductCase(
        name="american",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: AmericanOption(
            underlying=Equity("asset"),
            maturity=1.0,
            num_exercise_dates=8,
            strike=100.0,
            option_type=OptionType.PUT,
            asset_id="asset",
        ),
        pv_tol=1e-8,
        deriv_tol=1e-6,
    ),
    SingleProductCase(
        name="flexicall",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="asset"),
        build_product=lambda: FlexiCall(
            underlyings=[
                EuropeanOption(Equity("asset"), 0.5, 95.0, OptionType.CALL, asset_id="asset"),
                EuropeanOption(Equity("asset"), 1.0, 100.0, OptionType.CALL, asset_id="asset"),
                EuropeanOption(Equity("asset"), 1.5, 105.0, OptionType.CALL, asset_id="asset"),
            ],
            num_exercise_rights=2,
            asset_id="asset",
        ),
        pv_tol=1e-8,
        deriv_tol=1e-6,
    ),
    SingleProductCase(
        name="storage",
        build_model=lambda: BlackScholesModel(0.0, 100.0, 0.03, 0.2, asset_id="gas"),
        build_product=_build_storage_product,
        pv_tol=1e-8,
        deriv_tol=1e-6,
    ),
    SingleProductCase(
        name="bond",
        build_model=lambda: VasicekModel(0.0, 0.02, 0.03, 1.2, 0.01, asset_id="rate"),
        build_product=lambda: Bond(
            startdate=0.0,
            maturity=1.0,
            notional=1.0,
            tenor=0.5,
            pays_notional=True,
            fixed_rate=0.02,
            asset_id="rate",
        ),
        num_paths_pre=0,
        pv_tol=1e-10,
        deriv_tol=1e-10,
    ),
    SingleProductCase(
        name="swap",
        build_model=lambda: VasicekModel(0.0, 0.02, 0.03, 1.2, 0.01, asset_id="rate"),
        build_product=lambda: InterestRateSwap(
            startdate=0.0,
            enddate=1.0,
            notional=1.0,
            fixed_rate=0.02,
            tenor_fixed=0.5,
            tenor_float=0.5,
            irs_type=IRSType.PAYER,
            asset_id="rate",
        ),
        num_paths_pre=0,
        pv_tol=1e-10,
        deriv_tol=1e-10,
    ),
]


def _run_case(case: SingleProductCase):
    product = case.build_product()
    product.name = case.name
    model = case.build_model()

    controller = SimulationController(
        netting_sets=[NettingSet(name=case.name, products=[product])],
        model=model,
        risk_metrics=RiskMetrics(metrics=[PVMetric()]),
        num_paths_mainsim=case.num_paths_main,
        num_paths_presim=case.num_paths_pre,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=True,
    )
    torch.manual_seed(1234)
    np.random.seed(1234)
    return controller.run_simulation()


@pytest.mark.parametrize("case", SINGLE_PRODUCT_CASES, ids=[case.name for case in SINGLE_PRODUCT_CASES])
def test_single_product_pv_and_derivatives_are_finite(case: SingleProductCase):
    results = _run_case(case)

    pv = float(results.get_results(case.name, "pv", evaluation_idx=0))
    mc_error = float(results.get_mc_error(case.name, "pv", evaluation_idx=0))
    derivatives = results.get_derivatives(case.name, "pv", evaluation_idx=0)

    assert np.isfinite(pv)
    assert np.isfinite(mc_error)
    assert derivatives.keys()
    for value in derivatives.values():
        if value is None:
            continue
        assert np.isfinite(float(value))
