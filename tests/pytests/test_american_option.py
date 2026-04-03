from context import *

from common.packages import *
import numpy as np
from controller.controller import SimulationController
from products.netting_set import NettingSet
from models.black_scholes import BlackScholesModel
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from products.bermudan_option import AmericanOption, OptionType
from products.equity import Equity 
from engine.engine import SimulationScheme

threshold = 1e-8

def test_american_option_pv():
    """Test the PV calculation of an American option."""

    # Setup model and product
    model = BlackScholesModel(calibration_date=0.0, spot=100, rate=0.05, sigma=0.5)
    num_exercise_dates=1000
    maturity = 3.0
    strike = 100.0

    underlying = Equity('id')
    product = AmericanOption(
        underlying=underlying,
        maturity=maturity, 
        num_exercise_dates=num_exercise_dates, 
        strike=strike, 
        option_type=OptionType.CALL
    )

    netting_sets = [NettingSet(name=product.get_name(), products=[product])]

    # Metric timeline for exposure
    pv_metric = PVMetric()

    risk_metrics = RiskMetrics(
        metrics=[pv_metric],
    )

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    sc = SimulationController(
        netting_sets=netting_sets,
        model=model, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.ANALYTICAL, 
        differentiate=False,
    )

    sim_results=sc.run_simulation()

    pv=sim_results.get_results(product.get_name(), pv_metric.get_name(), evaluation_idx=0)
    
    assert abs(pv - 34.323036543142706) < threshold  # Expected PV value
    
