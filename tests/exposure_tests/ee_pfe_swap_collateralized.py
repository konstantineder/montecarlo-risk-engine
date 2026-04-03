from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import matplotlib.pyplot as plt
import os
from controller.controller import SimulationController
from products.netting_set import NettingSet
from models.vasicek import VasicekModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from metrics.ene_metric import ENEMetric
from metrics.risk_metrics import RiskMetrics
from products.swap import InterestRateSwap, IRSType


def build_irs(name: str) -> InterestRateSwap:
    irs = InterestRateSwap(
        startdate=0.0,
        enddate=2.0,
        notional=1.0,
        fixed_rate=0.03,
        tenor_fixed=0.5,
        tenor_float=0.5,
        irs_type=IRSType.PAYER,
    )
    irs.name = name
    return irs


def build_exposure_timeline(
    irs: InterestRateSwap,
    horizon: float,
    margin_period_of_risk: float,
    num_dense_points: int = 500,
) -> np.ndarray:
    dense_grid = np.linspace(0.0, horizon, num_dense_points)
    coupon_dates = np.array(irs.product_timeline.detach().cpu().tolist(), dtype=float)
    shifted_coupon_dates = coupon_dates + margin_period_of_risk
    exposure_timeline = np.unique(
        np.concatenate(
            [
                dense_grid,
                coupon_dates,
                shifted_coupon_dates,
                np.array([0.0, horizon], dtype=float),
            ]
        )
    )
    return exposure_timeline[(exposure_timeline >= 0.0) & (exposure_timeline <= horizon)]


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = VasicekModel(calibration_date=0.,rate=0.03,mean=0.05,mean_reversion_speed=0.02,volatility=0.02)

    uncollateralized_irs = build_irs("irs_uncollateralized")
    collateralized_irs = build_irs("irs_collateralized")

    uncollateralized_netting_set = NettingSet(
        name="irs_uncollateralized",
        products=[uncollateralized_irs],
    )
    
    mpor_days = 10
    mpor_years = mpor_days / 252
    collateralized_netting_set = NettingSet(
        name="irs_collateralized",
        products=[collateralized_irs],
        margin_period_of_risk=mpor_years,
    )

    # Metric timeline for exposure metrics
    exposure_horizon = 3.0
    exposure_timeline = np.linspace(0, 3., 200)
    epe_metric = EPEMetric()
    ene_metric = ENEMetric()
    pfe_metric = PFEMetric(0.9)

    risk_metrics = RiskMetrics(
        metrics=[epe_metric, ene_metric, pfe_metric],
        exposure_timeline=exposure_timeline
    )

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    
    sc = SimulationController(
        netting_sets=[
            uncollateralized_netting_set,
            collateralized_netting_set,
        ],
        model=model, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.EULER, 
        differentiate=False,
    )

    sim_results=sc.run_simulation()

    ees_irs1=sim_results.get_results(uncollateralized_netting_set.get_name(), epe_metric.get_name())
    enes_irs1=sim_results.get_results(uncollateralized_netting_set.get_name(), ene_metric.get_name())
    pfes_irs1=sim_results.get_results(uncollateralized_netting_set.get_name(), pfe_metric.get_name())
    ees_irs2=sim_results.get_results(collateralized_netting_set.get_name(), epe_metric.get_name())
    enes_irs2=sim_results.get_results(collateralized_netting_set.get_name(), ene_metric.get_name())
    pfes_irs2=sim_results.get_results(collateralized_netting_set.get_name(), pfe_metric.get_name())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for IRS1
    ax1.plot(exposure_timeline, ees_irs1, label='EPE (uncollateralized)', color='red')
    ax1.plot(exposure_timeline, enes_irs1, label='ENE (uncollateralized)', color='orange')
    ax1.plot(exposure_timeline, pfes_irs1, label='PFE (uncollateralized)', color='blue', linestyle='--')
    ax1.set_title('Uncollateralized Swap Exposure')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Exposure')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot for comparison between IRS1 and IRS2 EE
    ax2.plot(exposure_timeline, ees_irs1, label='EPE (uncollateralized)', color='red')
    ax2.plot(exposure_timeline, ees_irs2, label=f'EPE (collateralized, MPoR={mpor_days}d)', color='green')
    ax2.plot(exposure_timeline, pfes_irs1, label='PFE (uncollateralized)', color='blue', linestyle='--')
    ax2.plot(exposure_timeline, pfes_irs2, label=f'PFE (collateralized, MPoR={mpor_days}d)', color='navy', linestyle='--')
    ax2.set_title('Collateralized vs Uncollateralized Exposure')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Exposure')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_swap_collateralized.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
