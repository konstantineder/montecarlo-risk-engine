from context import *

from common.packages import *
import numpy as np
import matplotlib.pyplot as plt
import os
from controller.controller import SimulationController
from models.black_scholes import *
from metrics.pfe_metric import *
from metrics.epe_metric import *
from products.european_option import *
from products.equity import Equity
from engine.engine import *


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = BlackScholesModel(calibration_date=0.0, spot=100, rate=0.05, sigma=0.2)
    exercise_dates = [3.0]
    maturity = 3.0
    strike = 100.0

    underlying=Equity(id="")
    product = EuropeanOption(underlying=underlying,exercise_date=2.0,strike=100,option_type=OptionType.CALL)

    portfolio=[product]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 3.,100)
    ee_metric = EPEMetric()
    pfe_metric = PFEMetric(0.9)

    metrics=[ee_metric, pfe_metric]

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    sc=SimulationController(portfolio, model, metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.ANALYTICAL, False, exposure_timeline)

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
    plt.tight_layout()

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_european_equity_option.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")