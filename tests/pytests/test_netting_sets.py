from context import *

import numpy as np
import pytest
import torch

from common.packages import FLOAT, device
from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.epe_metric import EPEMetric
from metrics.metric import Metric
from metrics.pfe_metric import PFEMetric
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes import BlackScholesModel
from models.black_scholes_multi import BlackScholesMulti
from models.vasicek import VasicekModel
from products.barrier_option import BarrierOption, BarrierOptionType
from products.bond import Bond
from products.bermudan_option import AmericanOption
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType
from products.flexicall import FlexiCall


def test_netting_set_analytical_pv_sums_products():
    model = BlackScholesModel(0.0, 100.0, 0.05, 0.2)
    product_1 = EuropeanOption(Equity("eq"), 1.0, 90.0, OptionType.CALL)
    product_2 = EuropeanOption(Equity("eq"), 2.0, 110.0, OptionType.CALL)
    netting_set = NettingSet(name="equity_ns", products=[product_1, product_2])
    pv_metric = PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=1,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    results = controller.run_simulation()
    expected = float(
        product_1.compute_pv_analytically(model).squeeze()
        + product_2.compute_pv_analytically(model).squeeze()
    )

    assert results.get_netting_set_names() == ["equity_ns"]
    assert float(results.get_results("equity_ns", "pv", evaluation_idx=0)) == pytest.approx(expected)


def test_netting_set_analytical_multi_asset_pv_sums_products():
    model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.05,
        asset_ids=["eq1", "eq2"],
        spots=[100.0, 120.0],
        volatilities=[0.2, 0.3],
        correlation_matrix=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float),
    )
    product_1 = EuropeanOption(
        Equity("eq1"),
        1.0,
        90.0,
        OptionType.CALL,
        asset_id="eq1",
    )
    product_2 = EuropeanOption(
        Equity("eq2"),
        2.0,
        110.0,
        OptionType.CALL,
        asset_id="eq2",
    )
    netting_set = NettingSet(name="multi_equity_ns", products=[product_1, product_2])
    pv_metric = PVMetric(evaluation_type=Metric.EvaluationType.ANALYTICAL)

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=1,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    results = controller.run_simulation()
    expected = float(
        product_1.compute_pv_analytically(model).squeeze()
        + product_2.compute_pv_analytically(model).squeeze()
    )

    assert float(results.get_results("multi_equity_ns", "pv", evaluation_idx=0)) == pytest.approx(expected)


def test_bs_european_exposure_does_not_require_regression():
    model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.03,
        asset_ids=["eq1", "eq2"],
        spots=[100.0, 110.0],
        volatilities=[0.2, 0.25],
        correlation_matrix=np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    )
    product = EuropeanOption(Equity("eq1"), 1.0, 100.0, OptionType.CALL, asset_id="eq1")
    netting_set = NettingSet(name="european_bs_ns", products=[product])
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[EPEMetric(), PFEMetric(0.95)],
            exposure_timeline=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
        num_paths_mainsim=512,
        num_paths_presim=512,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    assert controller._product_requires_regression(product) is False
    assert controller.requires_regression is False


def test_bs_european_discounted_epe_matches_initial_pv_before_maturity():
    model = BlackScholesMulti(
        calibration_date=0.0,
        rate=0.03,
        asset_ids=["eq1", "eq2"],
        spots=[100.0, 110.0],
        volatilities=[0.2, 0.25],
        correlation_matrix=np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    )
    product = EuropeanOption(Equity("eq1"), 1.0, 100.0, OptionType.CALL, asset_id="eq1")
    netting_set = NettingSet(name="european_epe_ns", products=[product])
    epe_metric = EPEMetric()
    exposure_timeline = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(
            metrics=[epe_metric],
            exposure_timeline=exposure_timeline,
        ),
        num_paths_mainsim=4096,
        num_paths_presim=4096,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    torch.manual_seed(1234)
    results = controller.run_simulation()

    epe = results.get_results("european_epe_ns", epe_metric.get_name())
    initial_pv = float(product.compute_pv_analytically(model).squeeze())

    assert np.allclose(epe[:-1], initial_pv, atol=0.35, rtol=0.0)
    assert epe[-1] == pytest.approx(0.0, abs=1e-6)


def test_netting_set_threshold_reduces_exposure_metrics():
    model = VasicekModel(
        calibration_date=0.0,
        rate=0.0,
        mean=0.0,
        mean_reversion_speed=1.0,
        volatility=1e-8,
        asset_id="bond",
    )
    bond = Bond(
        startdate=0.0,
        maturity=1.0,
        notional=1.0,
        tenor=1.0,
        pays_notional=True,
        fixed_rate=0.0,
        asset_id="bond",
    )
    netting_set = NettingSet(name="bond_ns", products=[bond], threshold=0.25)
    exposure_timeline = np.array([0.0, 0.5])
    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.95)

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[epe_metric, pfe_metric], exposure_timeline=exposure_timeline),
        num_paths_mainsim=512,
        num_paths_presim=512,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    results = controller.run_simulation()
    epe = results.get_results("bond_ns", epe_metric.get_name())
    pfe = results.get_results("bond_ns", pfe_metric.get_name())

    assert np.allclose(epe, np.array([0.75, 0.75]), atol=1e-4, rtol=0.0)
    assert np.allclose(pfe, np.array([0.75, 0.75]), atol=1e-4, rtol=0.0)


def test_collateral_profile_uses_exact_delayed_exposure_times():
    product = EuropeanOption(Equity("eq"), 1.0, 100.0, OptionType.CALL)
    netting_set = NettingSet(
        name="collateral_ns",
        products=[product],
        margin_period_of_risk=0.5,
    )
    exposure_timeline = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=FLOAT, device=device)
    netted_exposures = torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [10.0, 20.0],
            [15.0, 30.0],
            [20.0, 40.0],
        ],
        dtype=FLOAT,
        device=device,
    )
    metric_exposure_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
    delayed_exposure_indices = torch.tensor([-1, 1, 3], dtype=torch.long, device=device)

    collateral_profile = netting_set.compute_collateral_profile(
        netted_exposures=netted_exposures,
        exposure_timeline=exposure_timeline,
        metric_exposure_indices=metric_exposure_indices,
        delayed_exposure_indices=delayed_exposure_indices,
    )
    unsecured_exposures = netting_set.compute_unsecured_exposure_profiles(
        netted_exposures=netted_exposures,
        exposure_timeline=exposure_timeline,
        metric_exposure_indices=metric_exposure_indices,
        delayed_exposure_indices=delayed_exposure_indices,
    )

    expected_collateral = torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [15.0, 30.0],
        ],
        dtype=FLOAT,
        device=device,
    )
    expected_unsecured = torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [5.0, 10.0],
        ],
        dtype=FLOAT,
        device=device,
    )

    assert torch.allclose(collateral_profile, expected_collateral)
    assert torch.allclose(unsecured_exposures, expected_unsecured)


def test_collateralized_netting_set_uses_delayed_netted_pv():
    model = VasicekModel(
        calibration_date=0.0,
        rate=0.0,
        mean=0.0,
        mean_reversion_speed=1.0,
        volatility=1e-8,
        asset_id="bond",
    )
    bond = Bond(
        startdate=0.0,
        maturity=2.0,
        notional=1.0,
        tenor=1.0,
        pays_notional=True,
        fixed_rate=0.0,
        asset_id="bond",
    )
    netting_set = NettingSet(
        name="collateralized_bond_ns",
        products=[bond],
        margin_period_of_risk=0.25,
    )
    exposure_timeline = np.array([0.0, 0.5, 1.0])
    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.95)

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[epe_metric, pfe_metric], exposure_timeline=exposure_timeline),
        num_paths_mainsim=512,
        num_paths_presim=512,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    results = controller.run_simulation()
    epe = results.get_results("collateralized_bond_ns", epe_metric.get_name())
    pfe = results.get_results("collateralized_bond_ns", pfe_metric.get_name())

    assert np.allclose(epe, np.array([1.0, 0.0, 0.0]), atol=1e-4, rtol=0.0)
    assert np.allclose(pfe, np.array([1.0, 0.0, 0.0]), atol=1e-4, rtol=0.0)


def test_numerical_netting_set_pv_matches_sum_of_components_for_bs_multi():
    asset_ids = ["asset_1", "asset_2"]
    correlation_matrix = np.array([[1.0, 0.4], [0.4, 1.0]])

    def build_model():
        return BlackScholesMulti(
            calibration_date=0.0,
            rate=0.03,
            asset_ids=asset_ids,
            spots=[100.0, 105.0],
            volatilities=[0.20, 0.25],
            correlation_matrix=correlation_matrix,
        )

    def build_products():
        return (
            EuropeanOption(Equity("asset_1"), 1.0, 95.0, OptionType.CALL, asset_id="asset_1"),
            EuropeanOption(Equity("asset_2"), 1.5, 110.0, OptionType.CALL, asset_id="asset_2"),
        )

    product_1, product_2 = build_products()

    pv_metric = PVMetric()
    risk_metrics = RiskMetrics(metrics=[pv_metric])

    standalone_controller = SimulationController(
        netting_sets=[
            NettingSet(name="ns_1", products=[product_1]),
            NettingSet(name="ns_2", products=[product_2]),
        ],
        model=build_model(),
        risk_metrics=risk_metrics,
        num_paths_mainsim=4096,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    combined_product_1, combined_product_2 = build_products()
    combined_controller = SimulationController(
        netting_sets=[NettingSet(name="combined_ns", products=[combined_product_1, combined_product_2])],
        model=build_model(),
        risk_metrics=risk_metrics,
        num_paths_mainsim=4096,
        num_paths_presim=0,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    standalone_results = standalone_controller.run_simulation()
    combined_results = combined_controller.run_simulation()

    expected = float(standalone_results.get_results("ns_1", "pv", evaluation_idx=0)) + float(
        standalone_results.get_results("ns_2", "pv", evaluation_idx=0)
    )
    actual = float(combined_results.get_results("combined_ns", "pv", evaluation_idx=0))

    assert actual == pytest.approx(expected, abs=1e-10, rel=0.0)


def test_family_executors_match_legacy_path_for_mixed_book():
    asset_ids = ["asset_1", "asset_2"]
    correlation_matrix = np.array([[1.0, 0.35], [0.35, 1.0]])

    def build_model():
        return BlackScholesMulti(
            calibration_date=0.0,
            rate=0.03,
            asset_ids=asset_ids,
            spots=[100.0, 105.0],
            volatilities=[0.20, 0.24],
            correlation_matrix=correlation_matrix,
        )

    def build_products():
        european_1 = EuropeanOption(Equity("asset_1"), 1.0, 95.0, OptionType.CALL, asset_id="asset_1")
        european_2 = EuropeanOption(Equity("asset_2"), 1.5, 110.0, OptionType.PUT, asset_id="asset_2")
        american_1 = AmericanOption(
            underlying=Equity("asset_1"),
            maturity=1.0,
            num_exercise_dates=8,
            strike=100.0,
            option_type=OptionType.PUT,
            asset_id="asset_1",
        )
        american_2 = AmericanOption(
            underlying=Equity("asset_2"),
            maturity=1.5,
            num_exercise_dates=12,
            strike=102.5,
            option_type=OptionType.CALL,
            asset_id="asset_2",
        )
        flexicall = FlexiCall(
            underlyings=[
                EuropeanOption(Equity("asset_1"), 0.5, 95.0, OptionType.CALL, asset_id="asset_1"),
                EuropeanOption(Equity("asset_1"), 1.0, 100.0, OptionType.CALL, asset_id="asset_1"),
                EuropeanOption(Equity("asset_1"), 1.5, 105.0, OptionType.CALL, asset_id="asset_1"),
            ],
            num_exercise_rights=2,
            asset_id="asset_1",
        )
        barrier = BarrierOption(
            startdate=0.0,
            maturity=1.25,
            strike=100.0,
            num_observation_timepoints=12,
            option_type=OptionType.CALL,
            barrier1=130.0,
            barrier_option_type1=BarrierOptionType.UPANDOUT,
            asset_id="asset_2",
        )
        return [european_1, european_2, american_1, american_2, flexicall, barrier]

    controller = SimulationController(
        netting_sets=[NettingSet(name="mixed_ns", products=build_products())],
        model=build_model(),
        risk_metrics=RiskMetrics(metrics=[PVMetric()]),
        num_paths_mainsim=1024,
        num_paths_presim=1024,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )
    torch.manual_seed(1234)
    results = controller.run_simulation()

    pv = float(results.get_results("mixed_ns", "pv", evaluation_idx=0))
    mc_error = float(results.get_mc_error("mixed_ns", "pv", evaluation_idx=0))

    assert np.isfinite(pv)
    assert np.isfinite(mc_error)


def test_family_executors_match_legacy_path_for_mixed_book_exposure_metrics():
    asset_ids = ["asset_1", "asset_2"]
    correlation_matrix = np.array([[1.0, 0.35], [0.35, 1.0]])
    exposure_timeline = np.linspace(0.0, 1.5, 6)

    def build_model():
        return BlackScholesMulti(
            calibration_date=0.0,
            rate=0.03,
            asset_ids=asset_ids,
            spots=[100.0, 105.0],
            volatilities=[0.20, 0.24],
            correlation_matrix=correlation_matrix,
        )

    def build_products():
        european_1 = EuropeanOption(Equity("asset_1"), 1.0, 95.0, OptionType.CALL, asset_id="asset_1")
        european_2 = EuropeanOption(Equity("asset_2"), 1.5, 110.0, OptionType.PUT, asset_id="asset_2")
        american_1 = AmericanOption(
            underlying=Equity("asset_1"),
            maturity=1.0,
            num_exercise_dates=8,
            strike=100.0,
            option_type=OptionType.PUT,
            asset_id="asset_1",
        )
        american_2 = AmericanOption(
            underlying=Equity("asset_2"),
            maturity=1.5,
            num_exercise_dates=12,
            strike=102.5,
            option_type=OptionType.CALL,
            asset_id="asset_2",
        )
        flexicall = FlexiCall(
            underlyings=[
                EuropeanOption(Equity("asset_1"), 0.5, 95.0, OptionType.CALL, asset_id="asset_1"),
                EuropeanOption(Equity("asset_1"), 1.0, 100.0, OptionType.CALL, asset_id="asset_1"),
                EuropeanOption(Equity("asset_1"), 1.5, 105.0, OptionType.CALL, asset_id="asset_1"),
            ],
            num_exercise_rights=2,
            asset_id="asset_1",
        )
        barrier = BarrierOption(
            startdate=0.0,
            maturity=1.25,
            strike=100.0,
            num_observation_timepoints=12,
            option_type=OptionType.CALL,
            barrier1=130.0,
            barrier_option_type1=BarrierOptionType.UPANDOUT,
            asset_id="asset_2",
        )
        return [european_1, european_2, american_1, american_2, flexicall, barrier]

    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.95)
    controller = SimulationController(
        netting_sets=[NettingSet(name="mixed_ns", products=build_products())],
        model=build_model(),
        risk_metrics=RiskMetrics(
            metrics=[epe_metric, pfe_metric],
            exposure_timeline=exposure_timeline,
        ),
        num_paths_mainsim=512,
        num_paths_presim=512,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )
    torch.manual_seed(1234)
    results = controller.run_simulation()

    epe = results.get_results("mixed_ns", epe_metric.get_name())
    pfe = results.get_results("mixed_ns", pfe_metric.get_name())

    assert epe.shape[0] == len(exposure_timeline)
    assert pfe.shape[0] == len(exposure_timeline)
    assert np.all(np.isfinite(epe))
    assert np.all(np.isfinite(pfe))
