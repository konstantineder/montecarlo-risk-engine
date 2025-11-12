from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from controller.controller import SimulationController
from models.black_scholes_multi import BlackScholesMulti
from metrics.pv_metric import PVMetric
from products.basket_option import BasketOption, OptionType,BasketOptionType


if __name__ == "__main__":
    # # --- GPU device setup ---
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


    def compute_prices_for_grid(param_grid,asset_ids,weights,correlation_matrix, num_paths, steps):
        results = []

        for T, S0, sigma, rate, strike in param_grid:
            spots=[S0,S0,S0,S0]
            sigmas=[sigma,sigma,sigma,sigma]
            model = BlackScholesMulti(0.0,rate,asset_ids,spots,sigmas,correlation_matrix)

            bo_artm = BasketOption(T,asset_ids,weights,strike,OptionType.CALL,BasketOptionType.ARITHMETIC,True)
            bo_geo = BasketOption(T,asset_ids,weights,strike,OptionType.CALL,BasketOptionType.GEOMETRIC)

            portfolio = [bo_artm,bo_geo]
            metrics=[PVMetric()]

            sc=SimulationController(portfolio, model, metrics, num_paths, 0, steps, SimulationScheme.ANALYTICAL, True)

            sim_results=sc.run_simulation()
            price_artm=sim_results.get_results(0,0)
            price_geo=sim_results.get_results(1,0)
            greeks_geo=sim_results.get_derivatives(1,0)[0]

            results.append({
                "spot": S0,
                "vola": sigma,
                "rate": rate,
                "time to maturity": T,
                "price (artm)": price_artm[0],
                "price (geo)": price_geo[0],
                "Delta": greeks_geo[0],
                "Vega": greeks_geo[4],
                "Rho": greeks_geo[8],
            })

        return pd.DataFrame(results)
    
    # Define parameter grid
    S0_vals = np.linspace(10, 300, 50)
    sigma_vals = [0.4]
    r_vals = [0.0]
    strikes = [100]
    T_vals = np.linspace(0.25, 1.0, 20)

    asset_ids = ["asset1", "asset2", "asset3", "asset4"]

    correlation_matrix = np.array([
        [1.0, 0.5, 0.5, 0.5],
        [0.5, 1.0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5],
        [0.5, 0.5, 0.5, 1.0]
    ])

    #sigmas, corr = compute_sigmas_and_correlation_from_cholesky(L)
    spots=[100.0,100.0,100.0,100.0]
    sigmas=[0.4,0.4,0.4,0.4]
    rate=0.0
    model=BlackScholesMulti(0.0,rate,asset_ids,spots,sigmas,correlation_matrix)
    weights=[0.25,0.25,0.25,0.25]
    basket=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.ARITHMETIC,True)
    basket_geo=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.GEOMETRIC)

    portfolio=[basket,basket_geo]

    metrics = [PVMetric()]

    # Cartesian product of all combinations
    # defining the parameter grid
    param_grid = list(cartesian_product(T_vals,S0_vals, sigma_vals, r_vals, strikes))
    
    # Simulate option prices and store in data frame.
    # Since only spot price and time to maturity are varied
    # the rate and volatility will be filtered out
    num_paths = 100000
    steps = 1
    df_results_spot_maturity=compute_prices_for_grid(param_grid,asset_ids,weights,correlation_matrix,num_paths,steps).drop(columns=["rate", "vola"])
    
    def compute_pv_analytically_wrapper(args):
        spot, rate, vola = args
        spots_deriv=[spot,spot,spot,spot]
        sigmas_deriv=[vola,vola,vola,vola]
        model_deriv = BlackScholesMulti(0.0,rate,asset_ids,spots_deriv,sigmas_deriv,correlation_matrix)
        #product_deriv = BarrierOption(100, 120,BarrierOptionType.UPANDOUT,0.0,2.0,OptionType.CALL,True,10)
        product_deriv = BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.GEOMETRIC)
        #product_deriv=BinaryOption(2.0,100,10,OptionType.CALL)
        
        return 0.25*float(product_deriv.compute_pv_analytically(model_deriv))

    # Define X, Y and Z data
    X = df_results_spot_maturity["time to maturity"].astype(float).to_numpy()
    Y = df_results_spot_maturity["spot"].astype(float).to_numpy()
    Zs = {
        "Simulation (Arithmetic BasketOption)": df_results_spot_maturity["price (artm)"].astype(float).to_numpy(),
        "Simulation (Geometric BasketOption)": df_results_spot_maturity["price (geo)"].astype(float).to_numpy(),
    }


    # Set fixed parameters for Delta and Vega plotting
    T_fixed = 1.0

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

    # Plot Delta and Vega below, comparing analytical and simulated values
    df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["time to maturity"], T_fixed)]

    # Delta Plot
    ax_delta = fig.add_subplot(2, 2, 3)
    ax_delta.plot(S0_vals, analytic_delta, marker='o', linestyle='-', label='Analytic Delta')
    ax_delta.plot(df_delta_vega["spot"], df_delta_vega["Delta"], marker='x', linestyle='--', label='Simulated Delta')
    ax_delta.set_title(f"Delta vs Spot (T={T_fixed})")
    ax_delta.set_xlabel("Spot Price (S₀)")
    ax_delta.set_ylabel("Delta")
    ax_delta.grid(True)
    ax_delta.legend()

    # Vega Plot
    ax_vega = fig.add_subplot(2, 2, 4)
    ax_vega.plot(S0_vals, analytic_vega, marker='o', linestyle='-', label='Analytic Vega')
    ax_vega.plot(df_delta_vega["spot"], df_delta_vega["Vega"], marker='x', linestyle='--', label='Simulated Vega')
    ax_vega.set_title(f"Vega vs Spot (T={T_fixed})")
    ax_vega.set_xlabel("Spot Price (S₀)")
    ax_vega.set_ylabel("Vega")
    ax_vega.grid(True)
    ax_vega.legend()

    plt.tight_layout()
    
    # Build the output directory relative to the repo root
    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "pv_basket_option.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")



    