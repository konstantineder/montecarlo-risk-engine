from context import *
import pytest
import torch
import numpy as np
from common.packages import device
from common.enums import SimulationScheme
from controller.controller import SimulationController
from models.black_scholes_multi import BlackScholesMulti
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from products.basket_option import BasketOption, OptionType,BasketOptionType
from engine.engine import SimulationScheme


def test_pv_basket_option():
    # # --- GPU device setup ---
    print(f"Using device: {device}")

    num_assets=4

    correlation_matrix = np.array([
        [1.0, 0.5, 0.5, 0.5],
        [0.5, 1.0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5],
        [0.5, 0.5, 0.5, 1.0]
    ])

    #sigmas, corr = compute_sigmas_and_correlation_from_cholesky(L)
    asset_ids = ["asset1", "asset2", "asset3", "asset4"]
    spots=[100.0,100.0,100.0,100.0]
    sigmas=[0.4,0.4,0.4,0.4]
    rate=0.0
    model=BlackScholesMulti(0.0,rate,asset_ids,spots,sigmas,correlation_matrix)
    weights=[0.25,0.25,0.25,0.25]
    basket=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.ARITHMETIC,True)
    basket_geo=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.GEOMETRIC)

    portfolio=[basket,basket_geo]

    metrics = [PVMetric()]
    risk_metrics=RiskMetrics(metrics=metrics)

    num_paths = 1000000
    steps = 1

    sc=SimulationController(
        portfolio=portfolio, 
        model=model, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths, 
        num_paths_presim=0, 
        num_steps=steps, 
        simulation_scheme=SimulationScheme.ANALYTICAL, 
        differentiate=False
    )
    
    sim_results=sc.run_simulation()
    price_basket=sim_results.get_results(0,0)[0]
    price_geo=sim_results.get_results(1,0)[0]
    analytical_price=basket_geo.compute_pv_analytically(model).item()
    
    precision = 0.02
    assert abs(price_basket - 12.60) < precision
    assert abs(price_geo - analytical_price) < precision

    



    