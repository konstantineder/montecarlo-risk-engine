from context import *

from common.packages import *
import numpy as np
import matplotlib.pyplot as plt
import os
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from metrics.ene_metric import ENEMetric
from products.bond import Bond
from engine.engine import SimulationScheme


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = VasicekModel(calibration_date=0.,rate=0.03,mean=0.05,mean_reversion_speed=0.02,volatility=0.2)
    maturity = 2.0

    frn = Bond(startdate=0.0,maturity=maturity,notional=1.0,tenor=0.25,pays_notional=True)
    coupon_bond = Bond(startdate=0.0,maturity=2.0,notional=maturity,tenor=0.25,pays_notional=True, fixed_rate=0.03)

    portfolio=[frn, coupon_bond]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 3.,100)
    epe_metric = EPEMetric()
    pfe_metric = PFEMetric(0.9)

    metrics=[epe_metric, pfe_metric]

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    sc=SimulationController(portfolio, model, metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False, exposure_timeline)

    sim_results=sc.run_simulation()

    ees_cbond=sim_results.get_results(0,0)
    ees_frn=sim_results.get_results(1,0)
    pfes_cbond=sim_results.get_results(0,1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for IRS1
    ax1.plot(exposure_timeline, ees_cbond, label='EPE (FRN)', color='red')
    ax1.plot(exposure_timeline, pfes_cbond, label='PFE (FRN)', color='blue', linestyle='--')
    ax1.set_title('Expected and Potential Future Exposure for Floating Rate Note')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Exposure')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot for comparison between IRS1 and IRS2 EE
    ax2.plot(exposure_timeline, ees_cbond, label='EE (FRN)', color='red')
    ax2.plot(exposure_timeline, ees_frn, label='EE (Coupon Bond)', color='green')
    ax2.set_title('Expected Exposure Comparison')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Exposure')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_bonds.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")