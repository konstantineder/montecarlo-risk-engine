from products.product import *
from request_interface.request_interface import AtomicRequestType, AtomicRequest
import numpy as np
from collections import defaultdict

# Bermudan option implementation
# Payoff depends on specification of the underlying
# Current product classes supported for underlying: Equity, Bond and Swap
class BermudanOption(Product):
    def __init__(self, 
                 underlying     : Product, 
                 exercise_dates : Union[Sequence[float], NDArray], 
                 strike         : float, 
                 option_type    : OptionType
                 ):
        
        super().__init__()
        self.strike = torch.tensor([strike], dtype=FLOAT, device=device)
        self.option_type = option_type
        self.product_timeline = torch.tensor(exercise_dates, dtype=FLOAT, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.num_exercise_rights = 1

        self.numeraire_requests={idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) for idx, t in enumerate(self.modeling_timeline)}
        self.spot_requests={idx: AtomicRequest(AtomicRequestType.SPOT) for idx in range(len(self.modeling_timeline))}

        self.underlying_requests={}
        idx=0
        for exercise_date in exercise_dates:
            self.underlying_requests[idx]=underlying.generate_composite_requests_for_date(exercise_date)
            idx+=1

    def get_atomic_requests(self):
        requests=defaultdict(list)
        for t, req in self.numeraire_requests.items():
            requests[t].append(req)

        for t, req in self.spot_requests.items():
            requests[t].append(req)

        return requests
    
    
    def get_composite_requests(self):
        requests=defaultdict(list)
        for t, req in self.underlying_requests.items():
            requests[t].append(req)

        return requests


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
            cashflows_all:     [num_paths, num_states] (FLOAT)
        """

        _, num_states = state_transition_matrix.shape

        # spot per path
        spot = resolved_requests[1][self.underlying_requests[time_idx].get_handle()]
        immediate_path = self.payoff(spot, model)                                

        # Broadcast immediate payoff to all hypothetical starting states
        immediate_all = immediate_path.unsqueeze(1).expand(-1, num_states)              

        # Continuation value
        if time_idx == len(self.product_timeline) - 1:
            continuation_all = torch.zeros_like(immediate_all)                          
        else:
            # Regression matrix for all paths at this time point
            A = regression_function.get_regression_matrix(spot)                         

            coeffs_all_states = self.regression_coeffs[time_idx] 
            coeffs_per_branch = coeffs_all_states[state_transition_matrix]                       

            # Compute continuation value for all branches
            continuation_all = (
                A.unsqueeze(1)                          
                * coeffs_per_branch        
            ).sum(dim=2)                                

        exercise_left = state_transition_matrix > 0                                                     
        should_exercise = (immediate_all > continuation_all) & exercise_left                

        numeraire = resolved_requests[0][self.numeraire_requests[time_idx].handle]       
        numeraire_all = numeraire.unsqueeze(1).expand(-1, num_states)                      

        cashflows_all = (
            immediate_all * should_exercise.float() / numeraire_all
        )                                                                                  

        # Update remaining "rights"/state after possible exercise
        next_state_transition_matrix = state_transition_matrix - should_exercise.long()                         

        return next_state_transition_matrix, cashflows_all
    
class AmericanOption(BermudanOption):
    def __init__(self, underlying, maturity, num_exercise_dates, strike, option_type):
        exercise_dates=np.linspace(0.,maturity,num_exercise_dates) if num_exercise_dates>1 else [maturity]
        super().__init__(underlying=underlying, exercise_dates=exercise_dates,
                         strike=strike,
                         option_type=option_type)
