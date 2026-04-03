from context import *

import torch

from models.cirpp import CIRPPModel


def test_cirpp_deterministic_path_tracks_market_hazard_curve():
    model = CIRPPModel(
        calibration_date=0.0,
        asset_id="cp",
        hazard_rates={1.0: 0.02, 2.0: 0.03, 5.0: 0.04},
        kappa=0.2,
        theta=0.03,
        volatility=0.01,
        y0=0.02,
        deterministic=True,
    )

    state = model.get_state(num_paths=4)
    dtype = state.dtype
    tensor_device = state.device
    assert torch.allclose(state[:, 0], torch.full((4,), 0.02, dtype=dtype, device=tensor_device))
    assert torch.allclose(state[:, 1], torch.zeros(4, dtype=dtype, device=tensor_device))

    next_state = model.simulate_time_step_euler(
        time1=torch.tensor(0.0, dtype=dtype, device=tensor_device),
        time2=torch.tensor(1.5, dtype=dtype, device=tensor_device),
        state=state,
        corr_randn=torch.zeros((4, 1), dtype=dtype, device=tensor_device),
    )

    assert torch.allclose(next_state[:, 0], torch.full((4,), 0.03, dtype=dtype, device=tensor_device))
    assert torch.allclose(next_state[:, 1], torch.full((4,), 0.03, dtype=dtype, device=tensor_device))

    conditional_survival = model.survival_probability(
        t=torch.tensor(1.0, dtype=dtype, device=tensor_device),
        T=torch.tensor(2.0, dtype=dtype, device=tensor_device),
        y_t=next_state[:, 0],
    )
    expected = torch.exp(torch.tensor(-0.03, dtype=dtype, device=tensor_device))
    assert torch.allclose(
        conditional_survival,
        torch.full((4,), expected, dtype=dtype, device=tensor_device),
    )
