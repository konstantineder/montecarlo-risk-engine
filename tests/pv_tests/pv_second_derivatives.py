from context import *

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from itertools import product as cartesian_product

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


def controller_hessian(
    spot: float,
    rate: float,
    sigma: float,
    maturity: float,
    strike: float,
):
    model = BlackScholesModel(0.0, spot, rate, sigma)
    product = EuropeanOption(Equity("id"), maturity, strike, OptionType.CALL)
    netting_set = NettingSet(name=product.get_name(), products=[product])

    controller = SimulationController(
        netting_sets=[netting_set],
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
    return results.get_second_derivatives(
        netting_set.get_name(),
        "pv",
        evaluation_idx=0,
    )


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


def compute_prices_for_grid(param_grid):
    results = []

    for maturity, spot, sigma, rate, strike in param_grid:
        formula_hessian = analytical_hessian(spot, rate, sigma, maturity, strike)
        simulated_hessian = controller_hessian(spot, rate, sigma, maturity, strike)
        results.append(
            {
                "spot": spot,
                "vola": sigma,
                "rate": rate,
                "time to maturity": maturity,
                "Gamma (Formula)": formula_hessian["spot"]["spot"],
                "Gamma (Controller)": float(simulated_hessian["spot"]["spot"]),
                "Vanna (Formula)": formula_hessian["spot"]["volatility"],
                "Vanna (Controller)": float(simulated_hessian["spot"]["volatility"]),
                "Vomma (Formula)": formula_hessian["volatility"]["volatility"],
                "Vomma (Controller)": float(simulated_hessian["volatility"]["volatility"]),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    spot_values = np.linspace(10.0, 300.0, 30)
    sigma_values = np.linspace(0.3, 1.0, 30)
    rate_values = [0.05]
    strike_values = [100.0]
    maturity_values = [2.0]

    param_grid = list(cartesian_product(maturity_values, spot_values, sigma_values, rate_values, strike_values))
    df_results = compute_prices_for_grid(param_grid).drop(columns=["rate", "time to maturity"])

    x = df_results["spot"].astype(float).to_numpy()
    y = df_results["vola"].astype(float).to_numpy()
    z_surfaces = [
        ("Gamma (Formula)", df_results["Gamma (Formula)"].astype(float).to_numpy()),
        ("Gamma (Controller)", df_results["Gamma (Controller)"].astype(float).to_numpy()),
        ("Vanna (Formula)", df_results["Vanna (Formula)"].astype(float).to_numpy()),
        ("Vanna (Controller)", df_results["Vanna (Controller)"].astype(float).to_numpy()),
        ("Vomma (Formula)", df_results["Vomma (Formula)"].astype(float).to_numpy()),
        ("Vomma (Controller)", df_results["Vomma (Controller)"].astype(float).to_numpy()),
    ]

    fig = plt.figure(figsize=(18, 16))

    for idx, (title, z_values) in enumerate(z_surfaces, 1):
        ax = fig.add_subplot(3, 2, idx, projection="3d")
        ax.plot_trisurf(x, y, z_values, cmap="viridis", edgecolor="none")
        ax.set_xlabel("Spot (S0)")
        ax.set_ylabel("Vola (sigma)")
        ax.set_zlabel(title)
        ax.set_title(title)

    plt.tight_layout()
    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "second_derivatives_bs.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
