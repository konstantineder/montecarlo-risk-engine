from context import *

from common.packages import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from controller.controller import SimulationController
from models.black_scholes import *
from metrics.eepe_metric import *
from metrics.risk_metrics import RiskMetrics
from products.european_option import EuropeanOption, OptionType
from products.equity import Equity
from engine.engine import SimulationScheme


if __name__ == "__main__":
    # # --- COU/GPU device setup ---
    print(f"Using device: {device}")


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

            underlying = Equity("id")
            product = EuropeanOption(underlying=underlying,exercise_date=T,strike=strike,option_type=OptionType.CALL)

            portfolio = [product]
            metrics=[EEPEMetric()]

            exposure_timeline = np.linspace(0, T,10)
            risk_metrics=RiskMetrics(metrics=metrics, exposure_timeline=exposure_timeline)

            sc=SimulationController(portfolio, model, risk_metrics, num_paths, 1000, steps, SimulationScheme.ANALYTICAL, True)

            sim_results=sc.run_simulation()
            price_sim=sim_results.get_results(0,0)[0]
            greeks=sim_results.get_derivatives(0,0)[0]

            results.append({
                "spot": S0,
                "vola": sigma,
                "rate": rate,
                "time to maturity": T,
                "price (sim)": price_sim,
                "Delta": greeks[0],
                "Vega": greeks[1],
                "Rho": greeks[2],
            })

        return pd.DataFrame(results)
    
    # Define parameter grid
    S0_vals = np.linspace(60, 130, 20)
    sigma_vals = [0.2]
    r_vals = [0.05]
    strikes = [100]
    T_vals = np.linspace(0.25, 2.0, 10)

    # Cartesian product of all combinations
    # defining the parameter grid
    param_grid = list(cartesian_product(T_vals,S0_vals, sigma_vals, r_vals, strikes))

    num_paths = 10000
    steps = 1
    
    # Simulate option prices and store in data frame.
    # Since only spot price and time to maturity are varied
    # the rate and volatility will be filtered out
    df_results_spot_maturity=compute_prices_for_grid(param_grid,num_paths,steps).drop(columns=["rate", "vola"])
    
    def compute_pv_analytically_wrapper(args):
        spot, rate, vola = args
        model_deriv = BlackScholesModel(0, spot, rate, vola)

        underlying = Equity("id")
        product_deriv = EuropeanOption(underlying=underlying,exercise_date=2.0,strike=100,option_type=OptionType.CALL)

        portfolio = [product_deriv]
        metrics=[EEPEMetric()]
        
        exposure_timeline = np.linspace(0.0, 2.0,10)
        risk_metrics=RiskMetrics(metrics=metrics, exposure_timeline=exposure_timeline)
    

        sc=SimulationController(portfolio, model_deriv, risk_metrics, num_paths, 1000, steps, SimulationScheme.ANALYTICAL, False)
        sim_results=sc.run_simulation()
        eepe=sim_results.get_results(0,0)[0]
        
        return eepe

    # Define X, Y and Z data
    X = df_results_spot_maturity["time to maturity"].astype(float).to_numpy()
    Y = df_results_spot_maturity["spot"].astype(float).to_numpy()
    Zs = {
        "Simulation": df_results_spot_maturity["price (sim)"].astype(float).to_numpy(),
    }


    # Set fixed parameters for Delta and Vega plotting
    T_fixed = 2.0
    sigma_fixed = 0.2

    df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["time to maturity"], T_fixed)]

    # Create plots
    fig = plt.figure(figsize=(14, 12))

    # Plot for Analytical Price and Simulation
    for i, (title, Z) in enumerate(Zs.items(), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("Time to Maturity (T)")
        ax.set_ylabel("Spot Price (S₀)")
        ax.set_zlabel(title)
        ax.set_title(title)

    deriv_param_grid = list(cartesian_product(S0_vals, r_vals, sigma_vals))

    analytic_delta = np.array([partial_diff(compute_pv_analytically_wrapper, [spot, rate, sigma], 0) 
                           for spot, rate, sigma in deriv_param_grid])

    analytic_vega = np.array([partial_diff(compute_pv_analytically_wrapper, [spot, rate, sigma], 2) 
                          for spot, rate, sigma in deriv_param_grid])
    
    analytic_rho = np.array([partial_diff(compute_pv_analytically_wrapper, [spot, rate, sigma], 1) 
                        for spot, rate, sigma in deriv_param_grid])

    # Plot Delta, Vega and Rho below, comparing analytical derivatives of exposures against AAD
    df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["time to maturity"], T_fixed)]

    # Delta Plot
    ax_delta = fig.add_subplot(2, 2, 2)
    ax_delta.plot(S0_vals, analytic_delta, marker='o', linestyle='-', label='Analytic Delta')
    ax_delta.plot(df_delta_vega["spot"], df_delta_vega["Delta"], marker='x', linestyle='--', label='Simulated Delta')
    ax_delta.set_title(f"Delta vs Spot (T={T_fixed}, σ={sigma_fixed})")
    ax_delta.set_xlabel("Spot Price (S₀)")
    ax_delta.set_ylabel("Delta")
    ax_delta.grid(True)
    ax_delta.legend()

    # Vega Plot
    ax_vega = fig.add_subplot(2, 2, 3)
    ax_vega.plot(S0_vals, analytic_vega, marker='o', linestyle='-', label='Analytic Vega')
    ax_vega.plot(df_delta_vega["spot"], df_delta_vega["Vega"], marker='x', linestyle='--', label='Simulated Vega')
    ax_vega.set_title(f"Vega vs Spot (T={T_fixed}, σ={sigma_fixed})")
    ax_vega.set_xlabel("Spot Price (S₀)")
    ax_vega.set_ylabel("Vega")
    ax_vega.grid(True)
    ax_vega.legend()

    # Rho Plot
    ax_vega = fig.add_subplot(2, 2, 4)
    ax_vega.plot(S0_vals, analytic_rho, marker='o', linestyle='-', label='Analytic Rho')
    ax_vega.plot(df_delta_vega["spot"], df_delta_vega["Rho"], marker='x', linestyle='--', label='Simulated Rho')
    ax_vega.set_title(f"Rho vs Spot (T={T_fixed}, σ={sigma_fixed})")
    ax_vega.set_xlabel("Spot Price (S₀)")
    ax_vega.set_ylabel("Rho")
    ax_vega.grid(True)
    ax_vega.legend()

    plt.tight_layout()
    
    # Build the output directory relative to the repo root
    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "eepe_simulation.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")



    