from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import matplotlib.pyplot as plt
import os
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from products.european_option import EuropeanOption, OptionType
from products.swap import InterestRateSwap, IRSType


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = VasicekModel(calibration_date=0.,rate=0.03,mean=0.05,mean_reversion_speed=0.02,volatility=0.02)
    exercise_dates = [3.0]
    maturity = 3.0
    strike = 100.0

    underlying=InterestRateSwap(startdate=0.0,enddate=2.0, notional=1.0,fixed_rate=0.03,tenor_fixed=0.25, tenor_float=0.25,irs_type=IRSType.RECEIVER)
    product = EuropeanOption(underlying=underlying,exercise_date=1.5,strike=0.0,option_type=OptionType.CALL)

    portfolio=[product]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 3.,100)
    ee_metric = EPEMetric()
    pfe_metric = PFEMetric(0.9)

    metrics=[ee_metric, pfe_metric]

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    sc=SimulationController(portfolio, model, metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False, exposure_timeline)

    sim_results=sc.run_simulation()

    ees=sim_results.get_results(0,0)
    pfes=sim_results.get_results(0,1)

    plt.figure(figsize=(10, 6))
    plt.plot(exposure_timeline, ees, label='Expected Exposure (EE)', color='red')
    plt.plot(exposure_timeline, pfes, label='Potential Future Exposure (PFE)', color='blue', linestyle='--')

    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Exposure')
    plt.title('Exposure vs. Time')

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Optional: Tight layout
    plt.tight_layout()

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_swaption.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")