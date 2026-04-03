from context import *

from common.packages import *
from common.enums import SimulationScheme
import numpy as np
import matplotlib.pyplot as plt
from controller.controller import SimulationController
from products.netting_set import NettingSet
from models.black_scholes import BlackScholesModel
from metrics.pfe_metric import PFEMetric
from metrics.epe_metric import EPEMetric
from metrics.risk_metrics import RiskMetrics
from products.flexicall import FlexiCall, EuropeanOption, OptionType
from products.equity import Equity


def build_flexicall_product() -> FlexiCall:
    exercise_dates = [0.5,1.0,1.5,2.0,2.5,3.0]
    strikes = [100.0, 1000.0, 100.0, 1000.0, 100.0, 1000.0]

    underlying = Equity('id')
    underlyings_options = []
    for idx in range(len(exercise_dates)):
        opt = EuropeanOption(
            underlying=underlying,
            exercise_date=exercise_dates[idx],
            strike=strikes[idx],
            option_type=OptionType.CALL
        )
        underlyings_options.append(opt)

    return FlexiCall(
        underlyings=underlyings_options,
        num_exercise_rights=4,
    )


if __name__ == "__main__":
    # # --- CPU/GPU device setup ---
    print(f"Using device: {device}")

    # Setup model and product
    model = BlackScholesModel(calibration_date=0.0, spot=100, rate=0.05, sigma=0.5)
    mpor_days = 10
    mpor_years = mpor_days / 252

    uncollateralized_netting_set = NettingSet(
        name="flexicall_uncollateralized_ns",
        products=[build_flexicall_product()],
    )
    collateralized_netting_set = NettingSet(
        name="flexicall_collateralized_ns",
        products=[build_flexicall_product()],
        margin_period_of_risk=mpor_years,
    )

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 4.,100)
    ee_metric = EPEMetric()
    pfe_metric = PFEMetric(0.9)

    risk_metrics = RiskMetrics(
        metrics=[ee_metric, pfe_metric],
        exposure_timeline=exposure_timeline
    )

    num_paths_mainsim=10000
    num_paths_presim=100000
    num_steps=1
    
    sc = SimulationController(
        netting_sets=[uncollateralized_netting_set, collateralized_netting_set],
        model=model, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.ANALYTICAL, 
        differentiate=False,
    )

    sim_results=sc.run_simulation()

    uncollateralized_ees = sim_results.get_results(uncollateralized_netting_set.get_name(), ee_metric.get_name())
    uncollateralized_pfes = sim_results.get_results(uncollateralized_netting_set.get_name(), pfe_metric.get_name())
    collateralized_ees = sim_results.get_results(collateralized_netting_set.get_name(), ee_metric.get_name())
    collateralized_pfes = sim_results.get_results(collateralized_netting_set.get_name(), pfe_metric.get_name())

    plt.figure(figsize=(10, 6))
    plt.plot(exposure_timeline, uncollateralized_ees, label='Expected Exposure (uncollateralized)', color='red')
    plt.plot(exposure_timeline, uncollateralized_pfes, label='Potential Future Exposure (uncollateralized)', color='blue', linestyle='--')
    plt.plot(exposure_timeline, collateralized_ees, label=f'Expected Exposure (collateralized, MPoR={mpor_days}d)', color='darkorange')
    plt.plot(exposure_timeline, collateralized_pfes, label=f'Potential Future Exposure (collateralized, MPoR={mpor_days}d)', color='navy', linestyle='--')

    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Exposure')
    plt.title('FlexiCall Exposure vs. Time')

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Optional: Tight layout
    plt.tight_layout()
    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "exposure_flexicall.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
