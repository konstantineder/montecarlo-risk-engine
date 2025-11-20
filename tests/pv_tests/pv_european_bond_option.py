from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from products.european_option import EuropeanOption, OptionType
from products.bond import Bond
from maths.regression import PolyomialRegression


if __name__ == "__main__":
    # # --- GPU device setup ---
    print(f"Using device: {device}")

    # Numerical Differentiation
    def partial_diff(function, args, arg_num, shift=1e-4):
        # Modify the argument at position `arg_num` by `shift`
        args[arg_num] = args[arg_num] + shift
        fplus = function(args)  # Call function with the modified argument
        args[arg_num] = args[arg_num] - 2 * shift
        fminus = function(args)  # Call function with the modified argument
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
            model = VasicekModel(calibration_date=0.,rate=rate,mean=0.05,mean_reversion_speed=0.02,volatility=sigma)
            #product = BinaryOption(T,strike,10,OptionType.CALL)
            underlying=Bond(startdate=0.0,maturity=2.0,notional=1.0,tenor=2.0, pays_notional=True, fixed_rate=0.0)
            product = EuropeanOption(underlying=underlying,exercise_date=T,strike=strike,option_type=OptionType.CALL)
            #portfolio=[BarrierOption(strike, 120,BarrierOptionType.UPANDOUT,0,T,OptionType.CALL,True,10)]
            portfolio = [product]
            risk_metrics=RiskMetrics(metrics=[PVMetric()])
            # Compute analytical price (if available)
            price_analytical = product.compute_pv_bond_option_analytically(model)
            RegressionFunction=PolyomialRegression(degree=3)

            sc=SimulationController(portfolio, model, risk_metrics, num_paths, 0, steps, SimulationScheme.EULER, True,RegressionFunction)

            sim_results=sc.run_simulation()
            price_sim=sim_results.get_results(0,0)
            greeks=sim_results.get_derivatives(0,0)[0]
            error_sim = rel_err(price_sim[0], float(price_analytical[0]))

            results.append({
                "vola": sigma,
                "rate": rate,
                "time to maturity": T,
                "price": price_analytical,
                "price (sim)": price_sim[0],
                "rel. error (sim)": error_sim,
                "Rho": greeks[0],
                "Vega": greeks[1],
            })

        return pd.DataFrame(results)
    
    # Define parameter grid
    S0_vals = [0.]
    sigma_vals = np.linspace(0.01,0.5,10)
    r_vals = [0.05]
    strikes = [0.5]
    T_vals = np.linspace(0.2, 2, 10)

    # Cartesian product of all combinations
    # defining the parameter grid
    param_grid = list(cartesian_product(T_vals,S0_vals, sigma_vals, r_vals, strikes))

    num_paths = 100000
    steps = 100
    
    # Simulate option prices and store in data frame.
    # Since only spot price and time to maturity are varied
    # the rate and volatility will be filtered out
    df_results_spot_maturity=compute_prices_for_grid(param_grid,num_paths,steps)
    
    T_fixed = 1.8
    def compute_pv_analytically_wrapper(args):
        rate, vola = args
        model_deriv = VasicekModel(calibration_date=0.,rate=rate,mean=0.05,mean_reversion_speed=0.02,volatility=vola)
        #product = BinaryOption(T,strike,10,OptionType.CALL)
        
        #product_deriv = BarrierOption(100, 120,BarrierOptionType.UPANDOUT,0.0,2.0,OptionType.CALL,True,10)
        underlying=Bond(startdate=0.0,maturity=2.0,notional=1.0,tenor=2.0, pays_notional=True, fixed_rate=0.0)
        product_deriv = EuropeanOption(underlying=underlying,exercise_date=T_fixed,strike=0.5,option_type=OptionType.CALL)
        #product_deriv=BinaryOption(2.0,100,10,OptionType.CALL)
        
        return float(product_deriv.compute_pv_bond_option_analytically(model_deriv))

    # Define X, Y and Z data
    X = df_results_spot_maturity["time to maturity"].astype(float).to_numpy()
    Y = df_results_spot_maturity["vola"].astype(float).to_numpy()
    Zs = {
        "Analytical Price": df_results_spot_maturity["price"].astype(float).to_numpy(),
        "Simulation": df_results_spot_maturity["price (sim)"].astype(float).to_numpy(),
    }
    T = df_results_spot_maturity["rel. error (sim)"].astype(float).to_numpy()


    # Set fixed parameters for Delta and Vega plotting

    df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["time to maturity"], T_fixed)]

    # Create plots
    fig = plt.figure(figsize=(14, 12))

    # Plot for Analytical Price and Simulation
    for i, (title, Z) in enumerate(Zs.items(), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("Time to Maturity (T)")
        ax.set_ylabel("Vola (sigma)")
        ax.set_zlabel(title)
        ax.set_title(title)

    deriv_param_grid = list(cartesian_product(r_vals, sigma_vals))

    analytic_delta = np.array([partial_diff(compute_pv_analytically_wrapper, [rate, sigma], 0) 
                           for rate, sigma in deriv_param_grid])

    analytic_vega = np.array([partial_diff(compute_pv_analytically_wrapper, [rate, sigma], 1) 
                          for rate, sigma in deriv_param_grid])

    # Plot Delta and Vega below, comparing analytical and simulated values
    df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["time to maturity"], T_fixed)]

    # Delta Plot
    ax_delta = fig.add_subplot(2, 2, 3)
    ax_delta.plot(sigma_vals, analytic_delta, marker='o', linestyle='-', label='Analytic Delta')
    ax_delta.plot(df_delta_vega["vola"], df_delta_vega["Rho"], marker='x', linestyle='--', label='Simulated Delta')
    ax_delta.set_title(f"Rho vs Vola (T={T_fixed}")
    ax_delta.set_xlabel("Vola")
    ax_delta.set_ylabel("Rho")
    ax_delta.grid(True)
    ax_delta.legend()

    # Vega Plot
    ax_vega = fig.add_subplot(2, 2, 4)
    ax_vega.plot(sigma_vals, analytic_vega, marker='o', linestyle='-', label='Analytic Vega')
    ax_vega.plot(df_delta_vega["vola"], df_delta_vega["Vega"], marker='x', linestyle='--', label='Simulated Vega')
    ax_vega.set_title(f"Vega vs Vola (T={T_fixed}")
    ax_vega.set_xlabel("Vola")
    ax_vega.set_ylabel("Vega")
    ax_vega.grid(True)
    ax_vega.legend()

    plt.tight_layout()
    
    # Build the output directory relative to the repo root
    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "pv_european_bond_option.png")
    plt.show()
    print(f"Plot saved to {out_path}")



    