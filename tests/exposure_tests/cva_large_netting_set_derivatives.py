from context import *

import os
from itertools import product as cartesian_product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.cva_metric import CVAMetric
from metrics.risk_metrics import RiskMetrics
from models.black_scholes import BlackScholesModel
from models.cirpp import CIRPPModel
from models.model_config import ModelConfig
from models.vasicek import VasicekModel
from products.bond import Bond
from products.equity import Equity
from products.european_option import EuropeanOption, OptionType
from products.swap import InterestRateSwap, IRSType


COUNTERPARTY_ID = "large_counterparty"
EQUITY_ASSET_ID = "equity"
RATES_ASSET_ID = "rates"

SPOT_PARAM_NAME = f"{EQUITY_ASSET_ID}.spot"
RATE_PARAM_NAMES = (
    f"{EQUITY_ASSET_ID}.rate",
    f"{RATES_ASSET_ID}.rate",
)

DEFAULT_HAZARD_RATES: dict[float, float] = {
    0.5: 0.006402303360855854,
    1.0: 0.01553038972325307,
    2.0: 0.009729741230773657,
    3.0: 0.015552544648116201,
    4.0: 0.021196186202801115,
    5.0: 0.02284319986706472,
    7.0: 0.010111423894480876,
    10.0: 0.00613267811172937,
    15.0: 0.0036969930706003337,
    20.0: 0.003791311459217732,
}

DEFAULT_SPOT_GRID = np.linspace(80.0, 130.0, 5)
DEFAULT_RATE_GRID = np.linspace(0.01, 0.05, 5)
DEFAULT_SPOT_LINE = np.linspace(80.0, 130.0, 11)
DEFAULT_RATE_LINE = np.linspace(0.01, 0.05, 11)
DEFAULT_EXPOSURE_TIMELINE = np.linspace(0.0, 6.0, 60)
DEFAULT_SPOT_BUMP = 1.0
DEFAULT_RATE_BUMP = 0.0025


def build_large_option_and_linear_netting_set(
    *,
    counterparty_id: str = COUNTERPARTY_ID,
    num_europeans: int = 20,
    num_bonds: int = 10,
    num_swaps: int = 150,
) -> NettingSet:
    products = []

    european_maturities = np.linspace(0.5, 3.0, 8)
    european_strike_scales = np.linspace(0.85, 1.15, 10)
    for idx in range(num_europeans):
        maturity = float(european_maturities[idx % len(european_maturities)])
        strike = 100.0 * float(european_strike_scales[idx % len(european_strike_scales)])
        option = EuropeanOption(
            Equity(EQUITY_ASSET_ID),
            maturity,
            strike,
            OptionType.CALL,
            asset_id=EQUITY_ASSET_ID,
        )
        option.name = f"large_european_call_{idx}"
        products.append(option)

    bond_maturities = np.linspace(2.0, 6.0, 8)
    bond_coupon_rates = np.linspace(0.018, 0.030, 5)
    for idx in range(num_bonds):
        maturity = float(bond_maturities[idx % len(bond_maturities)])
        fixed_rate = float(bond_coupon_rates[idx % len(bond_coupon_rates)])
        bond = Bond(
            startdate=0.0,
            maturity=maturity,
            notional=2.0,
            tenor=0.5,
            pays_notional=True,
            fixed_rate=fixed_rate,
            asset_id=RATES_ASSET_ID,
        )
        bond.name = f"large_bond_{idx}"
        products.append(bond)

    swap_maturities = np.linspace(2.0, 6.0, 8)
    swap_fixed_rates = np.linspace(0.019, 0.031, 6)
    for idx in range(num_swaps):
        maturity = float(swap_maturities[idx % len(swap_maturities)])
        fixed_rate = float(swap_fixed_rates[idx % len(swap_fixed_rates)])
        swap = InterestRateSwap(
            startdate=0.0,
            enddate=maturity,
            notional=25.0,
            fixed_rate=fixed_rate,
            tenor_fixed=0.5,
            tenor_float=0.25,
            irs_type=IRSType.PAYER,
            asset_id=RATES_ASSET_ID,
        )
        swap.name = f"large_swap_{idx}"
        products.append(swap)

    return NettingSet(
        name="large_cva_ns",
        products=products,
        counterparty_id=counterparty_id,
    )


def build_market_and_credit_model(
    *,
    spot: float,
    rate_level: float,
    sigma: float = 0.22,
    hazard_rates: dict[float, float] | None = None,
    counterparty_id: str = COUNTERPARTY_ID,
    deterministic_credit: bool = True,
) -> ModelConfig:
    if hazard_rates is None:
        hazard_rates = DEFAULT_HAZARD_RATES

    equity_model = BlackScholesModel(
        calibration_date=0.0,
        spot=spot,
        rate=rate_level,
        sigma=sigma,
        asset_id=EQUITY_ASSET_ID,
    )
    rates_model = VasicekModel(
        calibration_date=0.0,
        rate=rate_level,
        mean=0.03,
        mean_reversion_speed=1.0,
        volatility=0.01,
        asset_id=RATES_ASSET_ID,
    )
    credit_model = CIRPPModel(
        calibration_date=0.0,
        asset_id=counterparty_id,
        hazard_rates=hazard_rates,
        kappa=0.10,
        theta=0.01,
        volatility=0.02,
        y0=0.0001,
        deterministic=deterministic_credit,
    )
    return ModelConfig(
        models=[equity_model, rates_model, credit_model],
        inter_asset_correlation_matrix=[
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        ],
    )


def compute_cva_value(
    *,
    spot: float,
    rate_level: float,
    exposure_timeline: np.ndarray | None = None,
    sigma: float = 0.22,
    num_europeans: int = 20,
    num_bonds: int = 10,
    num_swaps: int = 150,
    num_paths_mainsim: int = 512,
    num_paths_presim: int = 512,
    num_steps: int = 4,
    differentiate: bool = False,
    deterministic_credit: bool = True,
) -> dict[str, float]:
    if exposure_timeline is None:
        exposure_timeline = DEFAULT_EXPOSURE_TIMELINE

    netting_set = build_large_option_and_linear_netting_set(
        num_europeans=num_europeans,
        num_bonds=num_bonds,
        num_swaps=num_swaps,
    )
    model = build_market_and_credit_model(
        spot=spot,
        rate_level=rate_level,
        sigma=sigma,
        counterparty_id=netting_set.counterparty_id,
        deterministic_credit=deterministic_credit,
    )
    cva_metric = CVAMetric(
        counterparty_id=netting_set.counterparty_id,
        recovery_rate=0.4,
    )
    risk_metrics = RiskMetrics(
        metrics=[cva_metric],
        exposure_timeline=exposure_timeline,
    )

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=risk_metrics,
        num_paths_mainsim=num_paths_mainsim,
        num_paths_presim=num_paths_presim,
        num_steps=num_steps,
        simulation_scheme=SimulationScheme.EULER,
        differentiate=differentiate,
    )

    results = controller.run_simulation()
    values = {
        "cva": float(
            results.get_results(
                netting_set.get_name(),
                cva_metric.get_name(),
                evaluation_idx=0,
            )
        ),
        "mc_error": float(
            results.get_mc_error(
                netting_set.get_name(),
                cva_metric.get_name(),
                evaluation_idx=0,
            )
        ),
    }
    if differentiate:
        derivatives = results.get_derivatives(
            netting_set.get_name(),
            cva_metric.get_name(),
            evaluation_idx=0,
        )
        values["dcva_dspot"] = float(derivatives[SPOT_PARAM_NAME])
        values["dcva_drate"] = float(sum(derivatives[param_name] for param_name in RATE_PARAM_NAMES))
    return values


def compute_cva_with_aad(
    *,
    spot: float,
    rate_level: float,
    **kwargs,
) -> dict[str, float]:
    return compute_cva_value(
        spot=spot,
        rate_level=rate_level,
        differentiate=True,
        **kwargs,
    )


def compute_cva_with_finite_differences(
    *,
    spot: float,
    rate_level: float,
    spot_bump: float = DEFAULT_SPOT_BUMP,
    rate_bump: float = DEFAULT_RATE_BUMP,
    **kwargs,
) -> dict[str, float]:
    base = compute_cva_value(
        spot=spot,
        rate_level=rate_level,
        **kwargs,
    )
    spot_up = compute_cva_value(
        spot=spot + spot_bump,
        rate_level=rate_level,
        **kwargs,
    )
    rate_up = compute_cva_value(
        spot=spot,
        rate_level=rate_level + rate_bump,
        **kwargs,
    )
    return {
        "cva": base["cva"],
        "mc_error": base["mc_error"],
        "dcva_dspot": (spot_up["cva"] - base["cva"]) / spot_bump,
        "dcva_drate": (rate_up["cva"] - base["cva"]) / rate_bump,
    }


def compute_cva_and_derivatives(
    *,
    spot: float,
    rate_level: float,
    derivative_method: str = "aad",
    **kwargs,
) -> dict[str, float]:
    if derivative_method == "aad":
        return compute_cva_with_aad(
            spot=spot,
            rate_level=rate_level,
            **kwargs,
        )
    if derivative_method == "finite_difference":
        return compute_cva_with_finite_differences(
            spot=spot,
            rate_level=rate_level,
            **kwargs,
        )
    raise ValueError(
        f"Unknown derivative_method '{derivative_method}'. Use 'aad' or 'finite_difference'."
    )


def compute_cva_surface(
    *,
    spot_values: np.ndarray,
    rate_values: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for spot, rate_level in cartesian_product(spot_values, rate_values):
        values = compute_cva_value(
            spot=float(spot),
            rate_level=float(rate_level),
            **kwargs,
        )
        rows.append(
            {
                "spot": float(spot),
                "rate": float(rate_level),
                **values,
            }
        )
    return pd.DataFrame(rows)


def _plot_surface(ax, x, y, z, title, zlabel):
    ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Spot")
    ax.set_ylabel("Initial rate")
    ax.set_zlabel(zlabel)
    ax.set_title(title)


def compute_spot_sensitivity_curve(
    *,
    spot_values: np.ndarray,
    fixed_rate: float,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for spot in spot_values:
        aad = compute_cva_and_derivatives(
            spot=float(spot),
            rate_level=fixed_rate,
            derivative_method="aad",
            **kwargs,
        )
        fd = compute_cva_and_derivatives(
            spot=float(spot),
            rate_level=fixed_rate,
            derivative_method="finite_difference",
            **kwargs,
        )
        rows.append(
            {
                "spot": float(spot),
                "rate": float(fixed_rate),
                "cva": aad["cva"],
                "mc_error": aad["mc_error"],
                "dcva_dspot_aad": aad["dcva_dspot"],
                "dcva_dspot_fd": fd["dcva_dspot"],
                "dcva_dspot_abs_diff": abs(aad["dcva_dspot"] - fd["dcva_dspot"]),
            }
        )
    return pd.DataFrame(rows)


def compute_rate_sensitivity_curve(
    *,
    rate_values: np.ndarray,
    fixed_spot: float,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for rate_level in rate_values:
        aad = compute_cva_and_derivatives(
            spot=fixed_spot,
            rate_level=float(rate_level),
            derivative_method="aad",
            **kwargs,
        )
        fd = compute_cva_and_derivatives(
            spot=fixed_spot,
            rate_level=float(rate_level),
            derivative_method="finite_difference",
            **kwargs,
        )
        rows.append(
            {
                "spot": float(fixed_spot),
                "rate": float(rate_level),
                "cva": aad["cva"],
                "mc_error": aad["mc_error"],
                "dcva_drate_aad": aad["dcva_drate"],
                "dcva_drate_fd": fd["dcva_drate"],
                "dcva_drate_abs_diff": abs(aad["dcva_drate"] - fd["dcva_drate"]),
            }
        )
    return pd.DataFrame(rows)


def plot_cva_surface_and_sensitivities(
    surface_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    *,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(18, 5.5))

    ax_surface = fig.add_subplot(1, 3, 1, projection="3d")
    _plot_surface(
        ax_surface,
        surface_df["spot"].to_numpy(dtype=float),
        surface_df["rate"].to_numpy(dtype=float),
        surface_df["cva"].to_numpy(dtype=float),
        "CVA",
        "CVA",
    )

    ax_spot = fig.add_subplot(1, 3, 2)
    ax_spot.plot(spot_df["spot"], spot_df["dcva_dspot_aad"], marker="o", linestyle="-", label="AAD")
    ax_spot.plot(spot_df["spot"], spot_df["dcva_dspot_fd"], marker="x", linestyle="--", label="Finite difference")
    ax_spot.set_title("dCVA/dSpot vs Spot")
    ax_spot.set_xlabel("Spot")
    ax_spot.set_ylabel("dCVA/dSpot")
    ax_spot.grid(True)
    ax_spot.legend()

    ax_rate = fig.add_subplot(1, 3, 3)
    ax_rate.plot(rate_df["rate"], rate_df["dcva_drate_aad"], marker="o", linestyle="-", label="AAD")
    ax_rate.plot(rate_df["rate"], rate_df["dcva_drate_fd"], marker="x", linestyle="--", label="Finite difference")
    ax_rate.set_title("dCVA/dRate vs Initial Rate")
    ax_rate.set_xlabel("Initial rate")
    ax_rate.set_ylabel("dCVA/dRate")
    ax_rate.grid(True)
    ax_rate.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    common_kwargs = {
        "num_europeans": 20,
        "num_bonds": 10,
        "num_swaps": 150,
        "num_paths_mainsim": 512,
        "num_paths_presim": 512,
        "num_steps": 4,
        "exposure_timeline": DEFAULT_EXPOSURE_TIMELINE,
        "deterministic_credit": True,
    }
    surface_df = compute_cva_surface(
        spot_values=DEFAULT_SPOT_GRID,
        rate_values=DEFAULT_RATE_GRID,
        **common_kwargs,
    )

    aad_fd_common_kwargs = {
        **common_kwargs,
        "num_paths_mainsim": 256,
        "num_paths_presim": 256,
    }
    spot_df = compute_spot_sensitivity_curve(
        spot_values=DEFAULT_SPOT_LINE[::2],
        fixed_rate=0.03,
        **aad_fd_common_kwargs,
    )
    rate_df = compute_rate_sensitivity_curve(
        rate_values=DEFAULT_RATE_LINE[::2],
        fixed_spot=100.0,
        **aad_fd_common_kwargs,
    )

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    surface_plot_path = os.path.join(out_dir, "cva_large_netting_set_surface.png")
    surface_csv_path = os.path.join(out_dir, "cva_large_netting_set_surface.csv")
    surface_summary_path = os.path.join(out_dir, "cva_large_netting_set_surface.txt")
    aad_fd_csv_path = os.path.join(out_dir, "cva_large_netting_set_aad_vs_fd.csv")

    plot_cva_surface_and_sensitivities(
        surface_df,
        spot_df,
        rate_df,
        out_path=surface_plot_path,
    )
    surface_df.to_csv(surface_csv_path, index=False)

    surface_center_row = surface_df.iloc[len(surface_df) // 2]
    surface_summary_lines = [
        "Large Netting Set CVA Surface Benchmark",
        f"num_grid_points: {len(DEFAULT_SPOT_GRID)} x {len(DEFAULT_RATE_GRID)}",
        f"num_products: {common_kwargs['num_europeans'] + common_kwargs['num_bonds'] + common_kwargs['num_swaps']}",
        "product_mix: 20 European calls, 10 bonds, 150 payer swaps",
        f"num_paths_main: {common_kwargs['num_paths_mainsim']}",
        f"num_paths_pre: {common_kwargs['num_paths_presim']}",
        f"num_exposure_points: {len(DEFAULT_EXPOSURE_TIMELINE)}",
        "simulation_scheme: EULER",
        f"deterministic_credit: {common_kwargs['deterministic_credit']}",
        f"mean_cva: {surface_df['cva'].mean():.6f}",
        f"min_cva: {surface_df['cva'].min():.6f}",
        f"max_cva: {surface_df['cva'].max():.6f}",
        "center_grid_point:",
        f"  spot={surface_center_row['spot']:.6f}",
        f"  rate={surface_center_row['rate']:.6f}",
        f"  cva={surface_center_row['cva']:.6f}",
        "sensitivity_curves:",
        f"  num_spot_points={len(spot_df)}",
        f"  num_rate_points={len(rate_df)}",
        f"  max_abs_diff_dspot={spot_df['dcva_dspot_abs_diff'].max():.6f}",
        f"  max_abs_diff_drate={rate_df['dcva_drate_abs_diff'].max():.6f}",
    ]

    aad_fd_df = pd.concat(
        [
            spot_df.assign(grid_type="spot"),
            rate_df.assign(grid_type="rate"),
        ],
        ignore_index=True,
        sort=False,
    )
    aad_fd_df.to_csv(aad_fd_csv_path, index=False)
    with open(surface_summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(surface_summary_lines))

    print(f"Plot saved to {surface_plot_path}")
    print(f"Surface data saved to {surface_csv_path}")
    print(f"Summary written to {surface_summary_path}")
    print(f"Comparison data saved to {aad_fd_csv_path}")
