from context import *

from common.packages import *
import numpy as np
import matplotlib.pyplot as plt
from controller.controller import SimulationController
from models.black_scholes import BlackScholesModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from metrics.risk_metrics import RiskMetrics
from products.barrier_option import BarrierOption, BarrierOptionType, OptionType
from engine.engine import SimulationScheme


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = BlackScholesModel(calibration_date=0.0, spot=100, rate=0.05, sigma=0.2)
    exercise_dates = [3.0]
    maturity = 3.0
    strike = 100.0


    product = BarrierOption(
        startdate=0.0,
        maturity=2.0,
        strike=strike,
        num_observation_timepoints=10,
        option_type=OptionType.CALL,
        barrier1=130,
        barrier_option_type1=BarrierOptionType.UPANDOUT
    )
    product.set_use_brownian_bridge()

    portfolio=[product]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 3.,100)
    ee_metric = EPEMetric()
    pfe_metric = PFEMetric(0.975)

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
        simulation_scheme=SimulationScheme.ANALYTICAL, 
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

    out_path = os.path.join(out_dir, "exposure_barrier_option.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")