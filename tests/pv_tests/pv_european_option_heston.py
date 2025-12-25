from context import *

from common.packages import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product

from controller.controller import SimulationController
from models.heston import HestonModel
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from products.european_option import EuropeanOption, OptionType
from products.equity import Equity
from common.enums import SimulationScheme


# -----------------------------
# Helpers
# -----------------------------
def partial_diff(function, args, arg_num, shift=1e-4):
    args = list(args)
    args[arg_num] = args[arg_num] + shift
    fplus = function(args)
    args[arg_num] = args[arg_num] - 2 * shift
    fminus = function(args)
    return (fplus - fminus) / (2 * shift)

def rel_err(x, y, eps=1e-2):
    denom = abs(x) + abs(y)
    if denom < eps:
        return 0.0
    return 2 * abs(x - y) / denom

def _f(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.asarray(x).reshape(-1)[0])
    try:
        return float(x)
    except Exception:
        return float(x.item())

def run_point(init_params, S0, T, r, K, paths, steps, scheme, differentiate=False):
    kappa, theta, sigma, rho, v0 = init_params
    model = HestonModel(0, S0, r, sigma, kappa=kappa, theta=theta, v0=v0, rho=rho)

    underlying = Equity()
    product = EuropeanOption(underlying=underlying, exercise_date=T, strike=K, option_type=OptionType.CALL)

    portfolio = [product]
    risk_metrics = RiskMetrics(metrics=[PVMetric()])

    p_an = _f(product.compute_pv_analytically_heston(model))

    sc = SimulationController(
        portfolio=portfolio,
        model=model,
        risk_metrics=risk_metrics,
        num_paths_mainsim=paths,
        num_paths_presim=0,
        num_steps=steps,
        simulation_scheme=scheme,
        differentiate=differentiate,
    )
    res = sc.run_simulation()
    p_sim = _f(res.get_results(0, 0)[0])
    mc = _f(res.get_mc_error(0, 0)[0])

    out = {"price_an": p_an, "price_sim": p_sim, "mc_error": mc}
    if differentiate:
        greeks = res.get_derivatives(0, 0)[0]
        out.update({"Delta": _f(greeks[0]), "Vega": _f(greeks[1]), "Rho": _f(greeks[2])})
    return out

def run_grid(init_params, grid, paths, steps, scheme):
    rows = []
    for T, S0, r, K in grid:
        out = run_point(init_params, S0, T, r, K, paths, steps, scheme, differentiate=False)
        diff = out["price_sim"] - out["price_an"]
        rows.append({
            "T": float(T), "S0": float(S0), "r": float(r), "K": float(K),
            "price_an": out["price_an"],
            "price_sim": out["price_sim"],
            "diff": diff,
            "rel": rel_err(out["price_sim"], out["price_an"]),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main (single figure with subplots)
# -----------------------------
if __name__ == "__main__":
    print(f"Using device: {device}")

    init_params = (0.01713417, 2.0, 0.45545583, -0.78975708, 0.0286834)  # (kappa, theta, sigma, rho, v0)

    r_fixed = 0.0
    K_fixed = 720.0

    # small surface grid (keep it fast)
    S0_surface = np.linspace(100, 800, 15)
    T_surface = np.linspace(0.1, 1.0, 15)
    grid_surface = list(cartesian_product(T_surface, S0_surface, [r_fixed], [K_fixed]))

    paths_surface = 100000
    steps_surface = 5

    df_surface = run_grid(init_params, grid_surface, paths_surface, steps_surface, SimulationScheme.QE)

    # slice greeks
    T_greek = 1.0
    S0_greek = np.linspace(100, 800, 25)
    paths_greek = 500000
    steps_greek = 10

    # analytic PV wrapper for FD greeks
    def pv_an(args):
        S0, r, sigma = args
        kappa, theta, _, rho, v0 = init_params
        model = HestonModel(0, S0, r, sigma, kappa=kappa, theta=theta, v0=v0, rho=rho)
        underlying = Equity()
        product = EuropeanOption(underlying=underlying, exercise_date=T_greek, strike=K_fixed, option_type=OptionType.CALL)
        return float(product.compute_pv_analytically_heston(model))

    analytic_delta = np.array([partial_diff(pv_an, [s, r_fixed, init_params[2]], 0) for s in S0_greek])
    analytic_vega  = np.array([partial_diff(pv_an, [s, r_fixed, init_params[2]], 2) for s in S0_greek])

    sim_delta = []
    sim_vega = []
    for s in S0_greek:
        out = run_point(init_params, s, T_greek, r_fixed, K_fixed, paths_greek, steps_greek, SimulationScheme.QE, differentiate=True)
        sim_delta.append(out["Delta"])
        sim_vega.append(out["Vega"])
    sim_delta = np.array(sim_delta)
    sim_vega = np.array(sim_vega)

    # convergence: price vs steps @ fixed point
    S0_conv = 800.0
    T_conv = 1.0
    paths_conv = 200000
    steps_list = np.linspace(2, 40, 10, dtype=int)

    rows_steps = []
    for scheme in [SimulationScheme.EULER, SimulationScheme.QE]:
        for st in steps_list:
            out = run_point(init_params, S0_conv, T_conv, r_fixed, K_fixed, paths_conv, st, scheme, differentiate=False)
            rows_steps.append({"scheme": scheme.name, "x": st, **out})
    df_steps = pd.DataFrame(rows_steps)

    # convergence: price vs paths @ fixed point
    # steps_conv = 10
    # paths_list = [2000, 5000, 10000, 20000, 50000, 100000, 200000]

    # rows_paths = []
    # for scheme in [SimulationScheme.EULER, SimulationScheme.QE]:
    #     for p in paths_list:
    #         out = run_point(init_params, S0_conv, T_conv, r_fixed, K_fixed, p, steps_conv, scheme, differentiate=False)
    #         rows_paths.append({"scheme": scheme.name, "x": p, **out})
    # df_paths = pd.DataFrame(rows_paths)

    # -----------------------------
    # ONE figure with subplots
    # -----------------------------
    fig = plt.figure(figsize=(18, 12))

    # (1) 3D analytic surface
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot_trisurf(df_surface["T"], df_surface["S0"], df_surface["price_an"], cmap="viridis", edgecolor="none")
    ax1.set_title("Analytical Price Surface")
    ax1.set_xlabel("T")
    ax1.set_ylabel("S0")
    ax1.set_zlabel("PV")

    # (2) 3D QE sim surface
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.plot_trisurf(df_surface["T"], df_surface["S0"], df_surface["price_sim"], cmap="viridis", edgecolor="none")
    ax2.set_title(f"QE Simulation Surface (paths={paths_surface}, steps={steps_surface})")
    ax2.set_xlabel("T")
    ax2.set_ylabel("S0")
    ax2.set_zlabel("PV")

    # (3) histogram of relative error
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(df_surface["rel"].to_numpy(), bins=40)
    ax3.set_title("Histogram: relative error (QE surface)")
    ax3.set_xlabel("rel err")
    ax3.set_ylabel("count")
    ax3.grid(True)

    # (4) Delta vs spot
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(S0_greek, analytic_delta, marker="o", linestyle="-", label="Analytic Delta")
    ax4.plot(S0_greek, sim_delta, marker="x", linestyle="--", label="QE AAD Delta")
    ax4.set_title(f"Delta vs Spot (T={T_greek}, K={K_fixed})")
    ax4.set_xlabel("S0")
    ax4.set_ylabel("Delta")
    ax4.grid(True)
    ax4.legend()

    # (5) Vega vs spot
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(S0_greek, analytic_vega, marker="o", linestyle="-", label="Analytic Vega")
    ax5.plot(S0_greek, sim_vega, marker="x", linestyle="--", label="QE AAD Vega")
    ax5.set_title(f"Vega vs Spot (T={T_greek}, K={K_fixed})")
    ax5.set_xlabel("S0")
    ax5.set_ylabel("Vega")
    ax5.grid(True)
    ax5.legend()

    # (6) Price convergence (two curves in one panel): steps and paths combined
    ax6 = fig.add_subplot(2, 3, 6)
    # steps on left axis
    p_an = float(df_steps["price_an"].iloc[0])
    ax6.axhline(p_an, linestyle="--", linewidth=2, label="Analytic")

    for scheme_name in df_steps["scheme"].unique():
        d = df_steps[df_steps["scheme"] == scheme_name].sort_values("x")
        ax6.errorbar(d["x"], d["price_sim"], yerr=d["mc_error"], marker="o", capsize=3, label=f"{scheme_name}")
    ax6.set_xscale("log")
    ax6.set_title(f"Price vs Steps (S0={S0_conv}, T={T_conv}, paths={paths_conv}\n(see legend))")
    ax6.set_xlabel("Steps (log)")
    ax6.set_ylabel("PV")
    ax6.grid(True)
    ax6.legend(fontsize=8)

    plt.tight_layout()
    
    # Build the output directory relative to the repo root
    out_dir = os.path.join("tests", "plots", "pv_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pv_european_bond_option_heston.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

    # Separate compact figure for paths (still “single plot” is impossible to mix axes cleanly)
    # If you insist on ONE figure only, comment this out and keep only ax6.
    # fig2 = plt.figure(figsize=(10, 5))
    # ax = fig2.add_subplot(1, 1, 1)
    # p_an2 = float(df_paths["price_an"].iloc[0])
    # ax.axhline(p_an2, linestyle="--", linewidth=2, label="Analytic")
    # for scheme_name in df_paths["scheme"].unique():
    #     d = df_paths[df_paths["scheme"] == scheme_name].sort_values("x")
    #     ax.errorbar(d["x"], d["price_sim"], yerr=d["mc_error"], marker="o", capsize=3, label=f"{scheme_name} vs paths")
    # ax.set_xscale("log")
    # ax.set_title(f"Price vs Paths @ S0={S0_conv}, T={T_conv}, steps={steps_conv}")
    # ax.set_xlabel("Paths (log)")
    # ax.set_ylabel("PV")
    # ax.grid(True)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()