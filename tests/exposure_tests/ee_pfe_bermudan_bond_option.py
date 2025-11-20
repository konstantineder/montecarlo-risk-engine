from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import matplotlib.pyplot as plt
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from metrics.risk_metrics import RiskMetrics
from products.bermudan_option import BermudanOption, OptionType
from products.bond import Bond


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = VasicekModel(calibration_date=0.,rate=0.03,mean=0.05,mean_reversion_speed=0.02,volatility=0.2)
    exercise_dates = [0.5,1.0,1.5,2.0,2.5,3.0]
    maturity = 3.0
    strike = 0.5

    underlying=Bond(
        startdate=0.0,maturity=3.0,
        notional=1.0,
        tenor=3.0, 
        pays_notional=True, 
        fixed_rate=0.0
    )
    product = BermudanOption(
        underlying=underlying, 
        exercise_dates=exercise_dates, 
        strike=strike, 
        option_type=OptionType.CALL
    )

    portfolio=[product]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 4.,1000)
    ee_metric = EPEMetric()
    pfe_metric = PFEMetric(0.9)

    metrics=[ee_metric, pfe_metric]
    risk_metrics=RiskMetrics(metrics=metrics, exposure_timeline=exposure_timeline)

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    sc = SimulationController(
        portfolio=portfolio, 
        model=model, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.EULER, 
        differentiate=False,
    )

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

    out_path = os.path.join(out_dir, "exposure_bermudan_bond_option.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")