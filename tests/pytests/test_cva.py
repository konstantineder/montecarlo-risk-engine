from context import *
import pytest
import numpy as np

from common.packages import *
from common.enums import SimulationScheme
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from models.cirpp import CIRPPModel
from models.model_config import ModelConfig
from metrics.cva_metric import CVAMetric 
from metrics.risk_metrics import RiskMetrics
from products.bond import Bond
from products.swap import InterestRateSwap, IRSType

@pytest.fixture
def hazards() -> dict[float, float]:
    """Hazard rates for General Motors Co from CDS data bootstrap."""
    hazards_rates: dict[float, float] = {
        0.5: 0.006402303360855854,
        1.0: 0.01553038972325307,
        2.0: 0.009729741230773657,
        3.0: 0.015552544648116201,
        4.0: 0.021196186202801115,
        5.0: 0.02284319986706472,
        7.0: 0.010111423894480876,
        10.0: 0.00613267811172937,
        15.0: 0.0036969930706003337,
        20.0: 0.003791311459217732
    }   
    return hazards_rates


def test_cva_corporate_bond(hazards):
    """Test CVA calculation for a zero-coupon corporate bond.
    
    Interest rates are modeled using a Vasicek model, and default intensity
    is modeled using a CIR++ model calibrated to market hazard rates.
    Both factors are assumed uncorrelated (no WWR), hence the CVA should match the
    analytical expected loss calculation.
    """
    
    interest_rate_model = VasicekModel(
        calibration_date=0.,
        rate=0.03,
        mean=0.05,
        mean_reversion_speed=1,
        volatility=0.2,
        asset_id="bond"
    )
    counterparty_id = "General Motors Co"
    intensity_model = CIRPPModel(
        calibration_date=0.,
        y0=0.0001,
        theta=0.01,
        kappa=0.1,
        volatility=0.02,
        hazard_rates=hazards,
        asset_id=counterparty_id
    )
    models = [interest_rate_model, intensity_model]
    inter_correlation_matrices: list[np.ndarray] = []
    inter_correlation_matrix = np.array([0.0])
    inter_correlation_matrices.append(inter_correlation_matrix)

    model_config = ModelConfig(
        models=models,
        inter_asset_correlation_matrix=inter_correlation_matrix,
    )

    maturity = 2.0
    zero_bond = Bond(
        startdate=0.0,
        maturity=maturity,
        notional=1,
        tenor=maturity,
        pays_notional=True, 
        fixed_rate=0.0,
        asset_id="bond"
    )
    portfolio=[zero_bond]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, maturity,100)
    cva_metric = CVAMetric(counterparty_id=counterparty_id, recovery_rate=0.4)
    risk_metrics = RiskMetrics(metrics=[cva_metric], exposure_timeline=exposure_timeline)

    num_paths_mainsim=100000
    num_paths_presim=100000
    num_steps=10
    sc=SimulationController(
        portfolio=portfolio, 
        model=model_config, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.EULER, 
        differentiate=False,
    )

    sim_results=sc.run_simulation()

    cva_bond=sim_results.get_results(0,0)[0]
    
    pv_bond = interest_rate_model.compute_bond_price(0.0, maturity, 0.03)
    survival_prob = intensity_model.survival_probability(0.0, maturity, 0.0001)
    expected_loss = (1 - 0.4) * (1 - survival_prob) * pv_bond
    
    assert abs(cva_bond.item() - expected_loss.item()) < 2e-6
    
def test_cva_wwr_payer_swap(hazards):
    """Test CVA calculation for a payer swap.
    
    Interest rates are modeled using a Vasicek model, and default intensity
    is modeled using a CIR++ model calibrated to market hazard rates.
    Both factors are assumed to be highly positively correlated (WWR).
    Hence the CVA should be larger than in the uncorrelated case (within MC error).
    """
    
    interest_rate_model = VasicekModel(
        calibration_date=0.,
        rate=0.03,
        mean=0.05,
        mean_reversion_speed=0.02,
        volatility=0.2,
        asset_id="irs"
    )
    counterparty_id = "General Motors Co"
    intensity_model = CIRPPModel(
        calibration_date=0.,
        y0=0.0001,
        theta=0.01,
        kappa=0.1,
        volatility=0.02,
        hazard_rates=hazards,
        asset_id=counterparty_id
    )
    models = [interest_rate_model, intensity_model]
    inter_correlation_matrices: list[np.ndarray] = []
    inter_correlation_matrix = np.array([0.99999])
    inter_correlation_matrices.append(inter_correlation_matrix)

    model_config = ModelConfig(
        models=models,
        inter_asset_correlation_matrix=inter_correlation_matrix,
    )

    maturity = 10.0
    irs = InterestRateSwap(
        startdate=0.0,
        enddate=maturity,
        notional=1.0,
        fixed_rate=0.03,
        tenor_fixed=0.25,
        tenor_float=0.25, 
        irs_type=IRSType.PAYER,
        asset_id="irs"
    )
    portfolio=[irs]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, maturity,100)
    cva_metric = CVAMetric(counterparty_id=counterparty_id,recovery_rate=0.4)

    metrics=[cva_metric]
    risk_metrics=RiskMetrics(metrics=metrics, exposure_timeline=exposure_timeline)

    num_paths_mainsim=100000
    num_paths_presim=100000
    num_steps=10
    sc=SimulationController(portfolio, model_config, risk_metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False)
    sim_results=sc.run_simulation()

    cva_irs=sim_results.get_results(0,0)[0]
    cva_irs_error=sim_results.get_mc_error(0,0)[0]

    cva_uncorr=1.114576156484541  # from test without WWR
    cva_uncorr_error=0.0024446898428056294  # MC error from test without WWR

    diff = cva_irs - cva_uncorr
    se_diff = (cva_irs_error**2 + cva_uncorr_error**2) ** 0.5
    assert (diff > 3 * se_diff)  # CVA with WWR should be larger than without WWR
