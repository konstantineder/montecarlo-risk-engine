from context import *

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.enums import SimulationScheme
from common.packages import FLOAT, device
from engine.engine import MonteCarloEngine
from models.cirpp import CIRPPModel


def build_hazard_curve() -> dict[float, float]:
    return {
        0.5: 0.010,
        1.0: 0.013,
        2.0: 0.017,
        3.0: 0.021,
        5.0: 0.026,
    }


def build_step_curve(hazard_rates: dict[float, float], horizon: float) -> tuple[np.ndarray, np.ndarray]:
    tenors = np.array(list(hazard_rates.keys()), dtype=float)
    hazards = np.array(list(hazard_rates.values()), dtype=float)

    x = np.concatenate(([0.0], tenors, [horizon]))
    y = np.concatenate(([hazards[0]], hazards, [hazards[-1]]))
    return x, y


if __name__ == "__main__":
    print(f"Using device: {device}")

    hazard_rates = build_hazard_curve()
    horizon = 5.0
    num_paths = 100000
    num_steps = 400
    num_display_paths = 20

    timeline = torch.linspace(0.0, horizon, 251, dtype=FLOAT, device=device)[1:]

    stochastic_model = CIRPPModel(
        calibration_date=0.0,
        asset_id="cp",
        hazard_rates=hazard_rates,
        kappa=0.8,
        theta=0.02,
        volatility=0.08,
        y0=0.01,
        deterministic=False,
    )
    deterministic_model = CIRPPModel(
        calibration_date=0.0,
        asset_id="cp",
        hazard_rates=hazard_rates,
        kappa=0.8,
        theta=0.02,
        volatility=0.08,
        y0=0.01,
        deterministic=True,
    )

    engine = MonteCarloEngine(
        simulation_timeline=timeline,
        simulation_type=SimulationScheme.EULER,
        model=stochastic_model,
        num_paths=num_paths,
        num_steps=num_steps,
        is_pre_simulation=False,
    )
    paths = engine.generate_paths()

    y_paths = paths[:, :, 0]
    log_survival_paths = paths[:, :, 1]
    survival_paths = torch.exp(-log_survival_paths)

    intensity_paths = torch.stack(
        [
            stochastic_model.lambda_t(t, y_paths[:, idx])
            for idx, t in enumerate(timeline)
        ],
        dim=1,
    )

    mean_intensity = intensity_paths.mean(dim=0).detach().cpu().numpy()
    mean_survival = survival_paths.mean(dim=0).detach().cpu().numpy()

    market_survival = torch.stack(
        [
            deterministic_model._market_survival_probability(t)
            for t in timeline
        ]
    ).detach().cpu().numpy()

    times = timeline.detach().cpu().numpy()
    step_x, step_y = build_step_curve(hazard_rates=hazard_rates, horizon=horizon)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for path_idx in range(min(num_display_paths, num_paths)):
        ax1.plot(
            times,
            intensity_paths[path_idx].detach().cpu().numpy(),
            color="steelblue",
            alpha=0.18,
            linewidth=1.0,
        )
    ax1.step(
        step_x,
        step_y,
        where="post",
        color="black",
        linewidth=2.0,
        label="Deterministic market hazard rate",
    )
    ax1.plot(
        times,
        mean_intensity,
        color="darkred",
        linewidth=2.0,
        label="Mean CIR++ intensity",
    )
    ax1.set_ylabel("Intensity / Hazard")
    ax1.set_title("CIR++ Intensity Scenarios vs Deterministic Hazard Curve")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    for path_idx in range(min(num_display_paths, num_paths)):
        ax2.plot(
            times,
            survival_paths[path_idx].detach().cpu().numpy(),
            color="seagreen",
            alpha=0.12,
            linewidth=1.0,
        )
    ax2.plot(
        times,
        mean_survival,
        color="darkgreen",
        linewidth=2.0,
        label="Mean scenario survival",
    )
    ax2.plot(
        times,
        market_survival,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Deterministic market survival",
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Survival Probability")
    ax2.set_title("CIR++ Survival Scenarios vs Deterministic Market Survival")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cirpp_scenarios_vs_deterministic_hazard.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
