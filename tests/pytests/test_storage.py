from context import *

from common.packages import *
from common.enums import SimulationScheme
from controller.controller import SimulationController
from products.netting_set import NettingSet
from metrics.pv_metric import PVMetric
from metrics.risk_metrics import RiskMetrics
from models.schwartz_two_factor import SchwartzTwoFactorModel
from products.storage import Storage, StorageAction
from products.storage_helpers import StorageConfig


THRESHOLD = 1e-10


def build_constant_window_storage(
    *,
    start_date: float = 0.0,
    end_date: float = 4.0,
    initial_amount: float = 4.0,
    injection_cost: float = 1.0,
    withdrawal_cost: float = 1.0,
) -> Storage:
    storage_config = StorageConfig()
    storage_config.add_volume_constraint(start_date, end_date, 0.0, 12.0, 0.0)
    storage_config.add_injection_flexibility(start_date, end_date, 0.0, 3.0)
    storage_config.add_injection_flexibility(start_date, end_date, 6.0, 1.5)
    storage_config.add_withdrawal_flexibility(start_date, end_date, 0.0, 1.0)
    storage_config.add_withdrawal_flexibility(start_date, end_date, 6.0, 2.5)
    storage_config.add_variable_injection_cost(start_date, injection_cost)
    storage_config.add_variable_withdrawal_cost(start_date, withdrawal_cost)

    return Storage(
        asset_id="thegasprice",
        start_date=start_date,
        end_date=end_date,
        initial_amount=initial_amount,
        storage_config=storage_config,
        num_states=4,
    )


def build_shifting_window_storage() -> Storage:
    storage_config = StorageConfig()
    storage_config.add_volume_constraint(0.0, 2.0, 0.0, 12.0, 0.0)
    storage_config.add_volume_constraint(2.0, 3.0, 0.0, 12.0, 0.0)
    storage_config.add_volume_constraint(3.0, 4.0, 3.0, 9.0, 0.0)
    storage_config.add_injection_flexibility(0.0, 4.0, 0.0, 3.0)
    storage_config.add_withdrawal_flexibility(0.0, 4.0, 0.0, 3.0)
    storage_config.add_variable_injection_cost(0.0, 0.0)
    storage_config.add_variable_withdrawal_cost(0.0, 0.0)

    return Storage(
        asset_id="thegasprice",
        start_date=0.0,
        end_date=4.0,
        initial_amount=6.0,
        storage_config=storage_config,
        num_states=4,
    )


def test_injection_transition_is_monotone_and_capacity_limited():
    storage = build_constant_window_storage()
    states = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=FLOAT, device=device)
    current_volumes = storage.state_to_volume(1.0, states)

    next_states = storage.compute_next_state(1.0, 2.0, StorageAction.INJECTION)(states)
    next_volumes = storage.state_to_volume(2.0, next_states)

    expected_volumes = torch.tensor([4.5, 5.5, 6.5, 7.5], dtype=FLOAT, device=device)

    assert torch.all(next_states[1:] >= next_states[:-1])
    assert torch.allclose(next_volumes, expected_volumes, atol=THRESHOLD, rtol=0.0)
    assert torch.all(next_volumes >= current_volumes)
    assert next_volumes[-1].item() == storage.storage_config.get_volume_constraint(2.0).vmax


def test_hold_action_projects_inventory_into_next_window():
    storage = build_shifting_window_storage()
    states = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=FLOAT, device=device)

    held_states = storage.compute_next_state(2.0, 3.0, StorageAction.DO_NOTHING)(states)
    held_volumes = storage.state_to_volume(3.0, held_states)

    expected_volumes = torch.tensor([3.0, 4.0, 8.0, 9.0], dtype=FLOAT, device=device)

    assert torch.allclose(held_volumes, expected_volumes, atol=THRESHOLD, rtol=0.0)
    assert held_states[1].item() == 0.5
    assert held_states[2].item() == 2.5


def test_volume_delta_matches_physical_volume_change_for_each_action():
    storage = build_constant_window_storage()
    states = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=FLOAT, device=device)
    current_volumes = storage.state_to_volume(1.0, states)

    for action in (
        StorageAction.INJECTION,
        StorageAction.WITHDRAWAL,
        StorageAction.DO_NOTHING,
    ):
        next_states = storage.compute_next_state(1.0, 2.0, action)(states)
        next_volumes = storage.state_to_volume(2.0, next_states)
        reported_delta = storage.compute_volume_difference(1.0, 2.0, action)(states)

        assert torch.allclose(
            reported_delta,
            next_volumes - current_volumes,
            atol=THRESHOLD,
            rtol=0.0,
        )


def test_storage_pv_withdraws_initial_inventory():
    storage_config = StorageConfig()
    storage_config.add_volume_constraint(0.0, 2.0, 0.0, 2.0, 0.0)
    storage_config.add_injection_flexibility(0.0, 2.0, 0.0, 1.0)
    storage_config.add_withdrawal_flexibility(0.0, 2.0, 0.0, 1.0)
    storage_config.add_variable_injection_cost(0.0, 0.0)
    storage_config.add_variable_withdrawal_cost(0.0, 0.0)

    product = Storage(
        asset_id="thegasprice",
        start_date=0.0,
        end_date=2.0,
        initial_amount=1.0,
        storage_config=storage_config,
        num_states=3,
    )

    model = SchwartzTwoFactorModel(
        calibration_date=0.0,
        curve_times=[0.0, 2.0],
        curve_values=[10.0, 10.0],
        rate=0.0,
        short_term_mean_reversion=1.0,
        short_term_vol=1e-8,
        long_term_drift=0.0,
        long_term_vol=1e-8,
        rho=0.0,
        asset_id="thegasprice",
    )

    pv_metric = PVMetric()
    controller = SimulationController(
        netting_sets=[NettingSet(name=product.get_name(), products=[product])],
        model=model,
        risk_metrics=RiskMetrics(metrics=[pv_metric]),
        num_paths_mainsim=2000,
        num_paths_presim=2000,
        num_steps=1,
        simulation_scheme=SimulationScheme.ANALYTICAL,
        differentiate=False,
    )

    pv = controller.run_simulation().get_results(product.get_name(), pv_metric.get_name(), evaluation_idx=0)
    assert abs(pv - 10.0) < 1e-3
