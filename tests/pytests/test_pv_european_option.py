from context import *
import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from common.enums import SimulationScheme
from controller.controller import SimulationController
from models.black_scholes import BlackScholesModel
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from products.european_option import EuropeanOption, OptionType
from products.equity import Equity
from engine.engine import SimulationScheme


# Numerical Differentiation
def partial_diff(function, args, arg_num, shift=1e-4):
    # Modify the argument at position `arg_num` by `shift`
    args[arg_num] = args[arg_num] + shift
    fplus = function(args) 
    args[arg_num] = args[arg_num] - 2 * shift
    fminus = function(args)  
    derivative = (fplus - fminus) / (2 * shift)
    return derivative

# # --- Relative error function ---
def rel_err(x, y, eps=1e-4):
    denom = abs(x) + abs(y)
    if denom < eps:
        return 0.0
    return 2 * abs(x - y) / denom

def compute_prices_for_grid(param_grid, num_paths, steps):
    results = []

    for T, S0, sigma, rate, strike in param_grid:
        model = BlackScholesModel(0, S0, rate, sigma)

        underlying=Equity()
        product = EuropeanOption(underlying=underlying,exercise_date=T,strike=strike,option_type=OptionType.CALL)

        portfolio = [product]
        metrics=[PVMetric()]
        risk_metrics=RiskMetrics(metrics=metrics)

        price_analytical = product.compute_pv_analytically(model)

        sc=SimulationController(portfolio, model, risk_metrics, num_paths, 0, steps, SimulationScheme.ANALYTICAL, True)

        sim_results=sc.run_simulation()
        price_sim=sim_results.get_results(0,0)
        greeks=sim_results.get_derivatives(0,0)[0]
        error_sim = rel_err(price_sim[0], float(price_analytical[0]))

        results.append({
            "spot": S0,
            "vola": sigma,
            "rate": rate,
            "time to maturity": T,
            "price": price_analytical,
            "price (sim)": price_sim[0],
            "rel. error (sim)": error_sim,
            "Delta": greeks[0],
            "Vega": greeks[1],
            "Rho": greeks[2],
        })

    return pd.DataFrame(results)

def test_pv_european_option():
    threshold=1e-3
    # # --- GPU device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define parameter grid
    S0_vals = [120]
    sigma_vals = [0.2]
    r_vals = [0.05]
    strikes = [100]
    T_vals = [2.0]

    # Cartesian product of all combinations
    # defining the parameter grid
    param_grid = list(cartesian_product(T_vals,S0_vals, sigma_vals, r_vals, strikes))

    num_paths = 1000000
    steps = 1
    
    # Simulate option prices and store in data frame.
    # Since only spot price and time to maturity are varied
    # the rate and volatility will be filtered out
    df_results_spot_maturity=compute_prices_for_grid(param_grid,num_paths,steps).drop(columns=["rate", "vola"])
    
    errors = df_results_spot_maturity["rel. error (sim)"].astype(float).to_numpy()

    errors_filtered = errors[errors >= threshold]
    print(errors_filtered)
    assert errors_filtered.shape[0] == 0




    