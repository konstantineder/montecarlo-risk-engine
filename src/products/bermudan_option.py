from products.product import *
from request_interface.request_interface import AtomicRequestType, AtomicRequest
import numpy as np
from collections import defaultdict

class BermudanOption(Product):
    """
    Bermudan option implementation
    Payoff depends on specification of the underlying
    Current product classes supported for underlying: Equity, Bond and Swap
    """
    def __init__(
        self, 
        underlying     : Product, 
        exercise_dates : Union[Sequence[float], NDArray], 
        strike         : float, 
        option_type    : OptionType,
        asset_id       : str | None = None,
    ):
        
        super().__init__(
            asset_ids=[asset_id],
            product_family=ProductFamily.BERMUDAN_EXERCISE,
        )
        self.strike = torch.tensor([strike], dtype=FLOAT, device=device)
        self.option_type = option_type
        self.product_timeline = torch.tensor(exercise_dates, dtype=FLOAT, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.num_exercise_rights = 1

        self.numeraire_requests={
            idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) for idx, t in enumerate(self.modeling_timeline)
            }
        asset_id = self.asset_ids[0]
        self.spot_requests={
            (idx, asset_id): AtomicRequest(AtomicRequestType.SPOT) for idx in range(len(self.modeling_timeline))
            }

        idx=0
        for exercise_date in exercise_dates:
            self.underlying_requests[idx]=underlying.generate_underlying_requests_for_date(exercise_date)
            idx+=1

    def get_num_states(self):
        return 2
    
    def get_initial_state(self):
        return 1

    def payoff(self, spots, model):
        zero = torch.tensor([0.0], device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(spots - self.strike, zero)
        else:
            return torch.maximum(self.strike - spots, zero)

    def compute_immediate_reward(
        self,
        spots: torch.Tensor,
        state_matrix: torch.Tensor,
    ) -> torch.Tensor:
        sign = 1.0 if self.option_type == OptionType.CALL else -1.0
        immediate = torch.clamp(
            sign * (spots.unsqueeze(-1) - self.strike.view(1, 1, 1)),
            min=0.0,
        )
        return immediate.expand(-1, -1, state_matrix.shape[2])

    def state_after_exercise(self, state_matrix: torch.Tensor) -> torch.Tensor:
        return torch.where(
            state_matrix > 0,
            state_matrix - 1,
            state_matrix,
        )

    def continuation_from_coeffs(
        self,
        explanatory: torch.Tensor,
        coeffs_all_states: torch.Tensor,
        state_matrix: torch.Tensor,
        regression_function: RegressionFunction,
    ) -> torch.Tensor:
        regression_matrix = regression_function.get_regression_matrix(explanatory)
        degree = coeffs_all_states.shape[-1]
        expanded_coeffs = coeffs_all_states.unsqueeze(1).expand(
            -1, state_matrix.shape[1], -1, -1
        )
        gather_index = state_matrix.unsqueeze(-1).expand(-1, -1, -1, degree)
        coeffs_per_state = torch.gather(expanded_coeffs, 2, gather_index)
        return (regression_matrix.unsqueeze(0).unsqueeze(2) * coeffs_per_state).sum(dim=3)

    def compute_discrete_exercise_step(
        self,
        spots: torch.Tensor,
        explanatory: torch.Tensor,
        numeraire: torch.Tensor,
        regression_function,
        state_matrix: torch.Tensor,
        coeffs_all_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        immediate = self.compute_immediate_reward(
            spots=spots,
            state_matrix=state_matrix,
        )
        if immediate.ndim == 2:
            immediate = immediate.unsqueeze(-1)
        if immediate.shape[2] != state_matrix.shape[2]:
            immediate = immediate.expand(-1, -1, state_matrix.shape[2])

        if coeffs_all_states is None:
            continuation_not_exercised = torch.zeros_like(immediate)
        else:
            continuation_not_exercised = self.continuation_from_coeffs(
                explanatory=explanatory,
                coeffs_all_states=coeffs_all_states,
                state_matrix=state_matrix,
                regression_function=regression_function,
            )

        should_exercise = (immediate > continuation_not_exercised) & (state_matrix > 0)
        state_after_exercise_matrix = self.state_after_exercise(state_matrix)

        if numeraire.ndim == 1:
            numeraire = numeraire.view(1, -1, 1)
        elif numeraire.ndim == 2:
            numeraire = numeraire.unsqueeze(-1)

        cashflows = immediate * should_exercise.to(dtype=FLOAT) / numeraire
        next_state = torch.where(should_exercise, state_after_exercise_matrix, state_matrix)
        return next_state, cashflows
    
    def compute_normalized_cashflows(
        self,
        time_idx: int,
        model,
        resolved_requests,
        regression_function,
        state_transition_matrix: torch.Tensor,
    ):
        """
        Vectorized version of compute_normalized_cashflows.
        For each path p and hypothetical starting state s0 (column),
        - decide exercise at this time_idx,
        - generate normalized cashflow for this step,
        - update remaining rights state.
        
        Returns:
            next_state_matrix: [num_paths, num_states] (long)
            cashflows:         [num_paths, num_states] (FLOAT)
        """
        if regression_function is None:
            raise ValueError("Discrete exercise evaluation requires a regression function.")
        if state_transition_matrix is None:
            raise ValueError("Discrete exercise evaluation requires a state matrix.")

        asset_id = self.get_asset_id()
        spot = resolved_requests[1][self.underlying_requests[time_idx].get_handle()]
        explanatory = resolved_requests[0][self.spot_requests[(time_idx, asset_id)].handle]
        numeraire = resolved_requests[0][self.numeraire_requests[time_idx].handle]
        coeffs_all_states = (
            None
            if time_idx == len(self.product_timeline) - 1 or self.regression_coeffs is None
            else self.regression_coeffs[time_idx].unsqueeze(0)
        )
        next_state, cashflows = self.compute_discrete_exercise_step(
            spots=spot.view(1, -1),
            explanatory=explanatory,
            numeraire=numeraire,
            regression_function=regression_function,
            state_matrix=state_transition_matrix.unsqueeze(0),
            coeffs_all_states=coeffs_all_states,
        )
        return next_state.squeeze(0), cashflows.squeeze(0)
    
class AmericanOption(BermudanOption):
    def __init__(
        self, 
        underlying, 
        maturity, 
        num_exercise_dates, 
        strike, 
        option_type, 
        asset_id: str | None = None
    ):
        exercise_dates=np.linspace(0.,maturity,num_exercise_dates) if num_exercise_dates>1 else [maturity]
        super().__init__(
            underlying=underlying, 
            exercise_dates=exercise_dates,
            strike=strike,
            option_type=option_type,
            asset_id=asset_id
            )
