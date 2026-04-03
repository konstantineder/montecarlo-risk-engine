from context import *

from common.packages import *
from common.enums import SimulationScheme
import matplotlib.pyplot as plt
import os
import sys
from controller.controller import SimulationController
from products.netting_set import NettingSet
from engine.engine import MonteCarloEngine
from maths.regression import PolyomialRegression
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from request_interface.request_interface import RequestInterface

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage_s2f_cases import (  # noqa: E402
    STORAGE_S2F_SCENARIOS,
    build_model_for_s2f_scenario,
    build_storage_for_s2f_scenario,
    offset_to_date,
)


def get_scenario(name: str):
    return next(scenario for scenario in STORAGE_S2F_SCENARIOS if scenario.name == name)


def constraint_series(product, offsets, *, initial: bool):
    lower = []
    upper = []

    for offset in offsets:
        constraint = (
            product.storage_config.get_initial_volume_constraint(offset)
            if initial
            else product.storage_config.get_volume_constraint(offset)
        )
        lower.append(constraint.vmin)
        upper.append(constraint.vmax)

    return lower, upper


def plot_storage_volume_constraints(
    scenario_name: str = "simplestorage6",
    output_filename: str | None = None,
):
    print(f"Using device: {device}")

    scenario = get_scenario(scenario_name)
    product = build_storage_for_s2f_scenario(scenario)
    model = build_model_for_s2f_scenario(scenario)
    netting_set = NettingSet(name=scenario.name, products=[product])

    controller = SimulationController(
        netting_sets=[netting_set],
        model=model,
        risk_metrics=RiskMetrics(metrics=[PVMetric()]),
        num_paths_mainsim=10000,
        num_paths_presim=10000,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
        regression_function=PolyomialRegression(degree=3),
    )

    request_interface = RequestInterface(controller.model)
    controller.perform_prepocessing(request_interface)

    main_engine = MonteCarloEngine(
        simulation_timeline=controller.simulation_timeline,
        simulation_type=controller.simulation_scheme,
        model=controller.model,
        num_paths=controller.num_paths_mainsim,
        num_steps=controller.num_steps,
        is_pre_simulation=False,
    )

    paths = main_engine.generate_paths()
    resolved_requests = request_interface.resolve_requests(paths)

    state_matrix = torch.full(
        (controller.num_paths_mainsim,),
        product.get_initial_state(),
        dtype=product.get_state_dtype(),
        device=device,
    ).unsqueeze(1)

    offsets = [0.0]
    plot_dates = [scenario.storage.start_date]
    volumes = [product.state_to_volume(0.0, state_matrix[:, 0]).detach().cpu()]

    for time_idx in range(len(product.product_timeline)):
        state_matrix, _ = product.compute_normalized_cashflows(
            time_idx=time_idx,
            model=controller.model,
            resolved_requests=resolved_requests,
            regression_function=controller.regression_function,
            state_transition_matrix=state_matrix,
        )

        next_offset = float(product.next_action_dates[time_idx].item())
        offsets.append(next_offset)
        plot_dates.append(offset_to_date(scenario.storage.start_date, next_offset))
        volumes.append(product.state_to_volume(next_offset, state_matrix[:, 0]).detach().cpu())

    mean_volume = [volume.mean().item() for volume in volumes]
    sample_count = min(15, controller.num_paths_mainsim)

    initial_lower, initial_upper = constraint_series(product, offsets, initial=True)
    optimized_lower, optimized_upper = constraint_series(product, offsets, initial=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for path_idx in range(sample_count):
        path_volume = [volume[path_idx].item() for volume in volumes]
        axes[0].plot(plot_dates, path_volume, color="lightgray", alpha=0.35, linewidth=1.0)

    axes[0].fill_between(
        plot_dates,
        optimized_lower,
        optimized_upper,
        color="steelblue",
        alpha=0.15,
        label="Optimized constraint band",
    )
    axes[0].plot(plot_dates, mean_volume, color="darkgreen", linewidth=2.5, label="Average volume")
    axes[0].plot(plot_dates, optimized_lower, color="navy", linewidth=1.5, linestyle="--", label="Optimized vmin")
    axes[0].plot(plot_dates, optimized_upper, color="firebrick", linewidth=1.5, linestyle="--", label="Optimized vmax")
    axes[0].set_ylabel("Volume")
    axes[0].set_title(f"Storage Volume Path and Constraints Over Time ({scenario.name}, S2F)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(loc="upper left")

    axes[1].plot(plot_dates, initial_lower, color="gray", linewidth=1.5, linestyle=":", label="Initial vmin")
    axes[1].plot(plot_dates, initial_upper, color="black", linewidth=1.5, linestyle=":", label="Initial vmax")
    axes[1].plot(plot_dates, optimized_lower, color="navy", linewidth=1.8, label="Optimized vmin")
    axes[1].plot(plot_dates, optimized_upper, color="firebrick", linewidth=1.8, label="Optimized vmax")
    axes[1].fill_between(plot_dates, optimized_lower, optimized_upper, color="steelblue", alpha=0.12)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Constraint")
    axes[1].set_title("Initial vs Optimized Volume Constraints")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()

    out_dir = os.path.join("tests", "plots", "exposure_tests")
    os.makedirs(out_dir, exist_ok=True)

    if output_filename is None:
        output_filename = f"storage_{scenario.name}_volume_constraints_s2f.png"

    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    return out_path


if __name__ == "__main__":
    plot_storage_volume_constraints()
