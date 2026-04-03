from __future__ import annotations

from enum import Enum

from products.product import *
from products.storage_helpers import DATE_TOL, StorageConfig
from request_interface.request_types import AtomicRequestType, AtomicRequest


class StorageAction(Enum):
    INJECTION = 0
    WITHDRAWAL = 1
    DO_NOTHING = 2


class Storage(Product):
    def __init__(
        self,
        asset_id: str,
        start_date: float,
        end_date: float,
        initial_amount: float,
        storage_config: StorageConfig,
        num_states: int,
        rollout_interval: float = 1.0,
    ):
        super().__init__(asset_ids=[asset_id])

        if num_states < 2:
            raise ValueError("Storage requires at least two discrete states.")
        if rollout_interval <= 0.0:
            raise ValueError("Rollout interval must be positive.")

        self.start_date = float(start_date)
        self.end_date = float(end_date)
        self.initial_amount = float(initial_amount)
        self.storage_config = storage_config
        self.num_states = num_states
        self.rollout_interval = float(rollout_interval)

        self.storage_config.optimize_volume_constraints(
            start_date=self.start_date,
            end_date=self.end_date,
            rollout_interval=self.rollout_interval,
            initial_volume=self.initial_amount,
        )

        action_dates: list[float] = []
        next_dates: list[float] = []
        date = self.start_date
        while date < self.end_date - DATE_TOL:
            next_date = min(date + self.rollout_interval, self.end_date)
            action_dates.append(date)
            next_dates.append(next_date)
            date = next_date

        self.product_timeline = torch.tensor(action_dates, dtype=FLOAT, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.next_action_dates = torch.tensor(next_dates, dtype=FLOAT, device=device)

        self.numeraire_requests = {
            idx: AtomicRequest(AtomicRequestType.NUMERAIRE, t)
            for idx, t in enumerate(action_dates)
        }
        self.spot_requests = {
            (idx, asset_id): AtomicRequest(AtomicRequestType.SPOT)
            for idx in range(len(action_dates))
        }

    def get_num_states(self):
        return self.num_states

    def get_state_dtype(self):
        return FLOAT

    def get_initial_state(self):
        return 0.0

    def _as_state_tensor(self, state: torch.Tensor | float) -> torch.Tensor:
        if torch.is_tensor(state):
            return state.to(dtype=FLOAT)
        return torch.tensor(state, dtype=FLOAT, device=device)

    def _volume_step(self, vmin: float, vmax: float) -> float:
        return self.storage_config.grid_step(vmin, vmax, self.num_states)

    def _state_scale(self, vmin: float, vmax: float) -> float:
        return self.storage_config.state_scale(vmin, vmax, self.num_states)

    def _volume_from_state(
        self,
        state: torch.Tensor | float,
        *,
        vmin: float,
        vmax: float,
    ) -> torch.Tensor:
        state_tensor = self._as_state_tensor(state)
        return vmin + state_tensor * self._volume_step(vmin, vmax)

    def _state_from_volume(
        self,
        volume: torch.Tensor,
        *,
        vmin: float,
        vmax: float,
    ) -> torch.Tensor:
        scale = self._state_scale(vmin, vmax)
        if scale == 0.0:
            return torch.zeros_like(volume)
        return (volume - vmin) * scale

    def _transition_volume(
        self,
        date: float,
        next_date: float,
        action_type: StorageAction,
        previous_state: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_constraint = self.storage_config.get_volume_constraint(next_date)
        previous_constraint = self.storage_config.get_volume_constraint(date)

        previous_volume = self._volume_from_state(
            previous_state,
            vmin=previous_constraint.vmin,
            vmax=previous_constraint.vmax,
        )
        period = max(next_date - date, 0.0)

        if action_type == StorageAction.INJECTION:
            daily_rate = self.storage_config.interpolate_rate_tensor(
                previous_volume,
                self.storage_config.get_injection_flexibility_slice(date),
            )
            next_volume = torch.clamp(
                previous_volume + daily_rate * period,
                max=next_constraint.vmax,
            )
        elif action_type == StorageAction.WITHDRAWAL:
            daily_rate = self.storage_config.interpolate_rate_tensor(
                previous_volume,
                self.storage_config.get_withdrawal_flexibility_slice(date),
            )
            next_volume = torch.clamp(
                previous_volume - daily_rate * period,
                min=next_constraint.vmin,
            )
        else:
            next_volume = torch.clamp(
                previous_volume,
                min=next_constraint.vmin,
                max=next_constraint.vmax,
            )

        return previous_volume, next_volume

    def compute_next_state(
        self,
        date: float,
        next_date: float,
        action_type: StorageAction,
    ):
        next_constraint = self.storage_config.get_volume_constraint(next_date)

        def mapping(previous_state: torch.Tensor | float) -> torch.Tensor:
            _, next_volume = self._transition_volume(
                date=date,
                next_date=next_date,
                action_type=action_type,
                previous_state=previous_state,
            )
            return self._state_from_volume(
                next_volume,
                vmin=next_constraint.vmin,
                vmax=next_constraint.vmax,
            )

        return mapping

    def compute_volume_difference(
        self,
        date: float,
        next_date: float,
        action_type: StorageAction,
    ):
        def mapping(previous_state: torch.Tensor | float) -> torch.Tensor:
            previous_volume, next_volume = self._transition_volume(
                date=date,
                next_date=next_date,
                action_type=action_type,
                previous_state=previous_state,
            )
            return next_volume - previous_volume

        return mapping

    def state_to_volume(
        self,
        date: float,
        state: torch.Tensor | float,
    ) -> torch.Tensor:
        constraint = self.storage_config.get_volume_constraint(float(date))
        return self._volume_from_state(state, vmin=constraint.vmin, vmax=constraint.vmax)

    def lookup_state_values(
        self,
        values_by_state: torch.Tensor,
        state_matrix: torch.Tensor,
    ) -> torch.Tensor:
        bounded_state = torch.clamp(state_matrix.to(dtype=FLOAT), 0.0, self.num_states - 1.0)
        lower = torch.floor(bounded_state).long()
        upper = torch.ceil(bounded_state).long()
        weight_upper = bounded_state - lower.to(dtype=FLOAT)

        lower_values = values_by_state.gather(dim=1, index=lower)
        upper_values = values_by_state.gather(dim=1, index=upper)
        return lower_values + weight_upper * (upper_values - lower_values)

    def compute_normalized_cashflows(
        self,
        time_idx: int,
        model,
        resolved_requests,
        regression_function: RegressionFunction,
        state_transition_matrix: torch.Tensor,
    ):
        _, num_branches = state_transition_matrix.shape

        date = float(self.product_timeline[time_idx].item())
        next_date = float(self.next_action_dates[time_idx].item())

        inj_state = self.compute_next_state(date, next_date, StorageAction.INJECTION)(
            state_transition_matrix
        )
        wd_state = self.compute_next_state(date, next_date, StorageAction.WITHDRAWAL)(
            state_transition_matrix
        )
        no_state = self.compute_next_state(date, next_date, StorageAction.DO_NOTHING)(
            state_transition_matrix
        )

        inj_delta = self.compute_volume_difference(date, next_date, StorageAction.INJECTION)(
            state_transition_matrix
        )
        wd_delta = self.compute_volume_difference(date, next_date, StorageAction.WITHDRAWAL)(
            state_transition_matrix
        )
        no_delta = self.compute_volume_difference(date, next_date, StorageAction.DO_NOTHING)(
            state_transition_matrix
        )

        spot = self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_requests[0],
            request_type=AtomicRequestType.SPOT,
            time_idx=time_idx,
            asset_id=self.get_asset_id(),
        ).unsqueeze(1).expand(-1, num_branches)

        injection_cost = self.storage_config.get_variable_injection_cost(date)
        withdrawal_cost = self.storage_config.get_variable_withdrawal_cost(date)

        inj_payoff = -inj_delta * (spot + injection_cost)
        wd_payoff = -wd_delta * (spot - withdrawal_cost)
        no_spot = torch.where(no_delta >= 0.0, spot + injection_cost, spot - withdrawal_cost)
        no_payoff = -no_delta * no_spot

        if next_date >= self.end_date - DATE_TOL:
            continuation_inj = torch.zeros_like(inj_payoff)
            continuation_no = torch.zeros_like(no_payoff)
            continuation_wd = torch.zeros_like(wd_payoff)
        else:
            explanatory = self.get_resolved_atomic_request(
                resolved_atomic_requests=resolved_requests[0],
                request_type=AtomicRequestType.SPOT,
                time_idx=time_idx,
                asset_id=self.get_asset_id(),
            )
            continuation_grid = self.evaluate_regression_grid(
                explanatory=explanatory,
                regression_function=regression_function,
                time_idx=time_idx,
            )
            continuation_inj = self.lookup_state_values(continuation_grid, inj_state)
            continuation_no = self.lookup_state_values(continuation_grid, no_state)
            continuation_wd = self.lookup_state_values(continuation_grid, wd_state)

        action_values = torch.stack(
            [
                inj_payoff + continuation_inj,
                no_payoff + continuation_no,
                wd_payoff + continuation_wd,
            ],
            dim=2,
        )
        next_states = torch.stack([inj_state, no_state, wd_state], dim=2)
        payoffs = torch.stack([inj_payoff, no_payoff, wd_payoff], dim=2)

        best_action = torch.argmax(action_values, dim=2, keepdim=True)
        next_state_matrix = torch.gather(next_states, dim=2, index=best_action).squeeze(2)
        cashflows = torch.gather(payoffs, dim=2, index=best_action).squeeze(2)

        numeraire = self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_requests[0],
            request_type=AtomicRequestType.NUMERAIRE,
            time_idx=time_idx,
        ).unsqueeze(1).expand(-1, num_branches)

        return next_state_matrix, cashflows / numeraire
