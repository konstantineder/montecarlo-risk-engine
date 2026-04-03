from products.european_option import *
from request_interface.request_types import AtomicRequestType, AtomicRequest

class FlexiCall(Product):
    """
    Implementation of FlexiCall for a basket of European options
    Payoff depends on specification of the underlyings of options
    Current product classes supported: Equity, Bond and Swap
    """
    def __init__(
        self, 
        underlyings        : list[EuropeanOption], 
        num_exercise_rights: int, 
        asset_id           : str | None = None,
    ):
        
        super().__init__(
            asset_ids=[asset_id],
            product_family=ProductFamily.FLEXICALL_EXERCISE,
        )
        assert num_exercise_rights <= len(underlyings), \
            "Number of exercise rights cannot exceed number of underlyings"
        assert all(
            opt.option_type == underlyings[0].option_type 
            for opt in underlyings
            ), "All underlyings must have the same option type"
        
        self.underlyings = sorted(underlyings, key=lambda opt: opt.exercise_date.item())
        assert all(
            self.underlyings[i].exercise_date < self.underlyings[i+1].exercise_date 
            for i in range(len(underlyings)-1)
            ), "Exercise dates must be distinct"
        
        exercise_dates = [opt.exercise_date.item() for opt in self.underlyings]
        self.product_timeline = torch.tensor(exercise_dates, dtype=FLOAT, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.num_exercise_rights = num_exercise_rights

        asset_id = self.get_asset_id()
        self.numeraire_requests={
            idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) 
            for idx, t in enumerate(self.modeling_timeline)
            }
        self.spot_requests={
            (idx, asset_id): AtomicRequest(AtomicRequestType.SPOT) 
            for idx in range(len(self.modeling_timeline))
            }
        self.underlying_requests={
            idx: opt.underlying_requests[0] 
            for idx, opt in enumerate(self.underlyings)
            }      

    def get_num_states(self):
        return self.num_exercise_rights + 1
    
    def get_initial_state(self):
        return self.num_exercise_rights

    def compute_immediate_reward(
        self,
        spots: torch.Tensor,
        time_idx: int,
        state_matrix: torch.Tensor,
    ) -> torch.Tensor:
        sign = 1.0 if self.underlyings[0].option_type == OptionType.CALL else -1.0
        strike = self.underlyings[time_idx].strike.view(1, 1, 1)
        immediate = torch.clamp(
            sign * (spots.unsqueeze(-1) - strike),
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
        time_idx: int,
        explanatory: torch.Tensor,
        numeraire: torch.Tensor,
        regression_function,
        state_matrix: torch.Tensor,
        coeffs_all_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        immediate = self.compute_immediate_reward(
            spots=spots,
            time_idx=time_idx,
            state_matrix=state_matrix,
        )
        if immediate.ndim == 2:
            immediate = immediate.unsqueeze(-1)
        if immediate.shape[2] != state_matrix.shape[2]:
            immediate = immediate.expand(-1, -1, state_matrix.shape[2])

        state_after_exercise_matrix = self.state_after_exercise(state_matrix)
        if coeffs_all_states is None:
            continuation_not_exercised = torch.zeros_like(immediate)
            continuation_exercised = torch.zeros_like(immediate)
        else:
            continuation_not_exercised = self.continuation_from_coeffs(
                explanatory=explanatory,
                coeffs_all_states=coeffs_all_states,
                state_matrix=state_matrix,
                regression_function=regression_function,
            )
            continuation_exercised = self.continuation_from_coeffs(
                explanatory=explanatory,
                coeffs_all_states=coeffs_all_states,
                state_matrix=state_after_exercise_matrix,
                regression_function=regression_function,
            )

        should_exercise = (
            (immediate + continuation_exercised > continuation_not_exercised)
            & (state_matrix > 0)
        )

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
        regression_function: RegressionFunction,
        state_transition_matrix: torch.Tensor,
    ):
        """
        Vectorized step over all paths & states.
        Returns:
            next_state_transition_matrix: [N,S] longs
            cashflows:                    [N,S] floats (normalized)
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
            time_idx=time_idx,
            explanatory=explanatory,
            numeraire=numeraire,
            regression_function=regression_function,
            state_matrix=state_transition_matrix.unsqueeze(0),
            coeffs_all_states=coeffs_all_states,
        )
        return next_state.squeeze(0), cashflows.squeeze(0)
