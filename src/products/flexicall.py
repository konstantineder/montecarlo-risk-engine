from products.european_option import *
from request_interface.request_interface import AtomicRequestType, AtomicRequest
import numpy as np
from collections import defaultdict

# Implementation of FlexiCall for a basket of European options
# Payoff depends on specification of the underlyings of options
# Current product classes supported: Equity, Bond and Swap
class FlexiCall(Product):
    def __init__(self, 
                 underlyings        : list[EuropeanOption], 
                 num_exercise_rights: int, 
                 ):
        
        super().__init__()
        assert num_exercise_rights <= len(underlyings), "Number of exercise rights cannot exceed number of underlyings"
        assert all(opt.option_type == underlyings[0].option_type for opt in underlyings), "All underlyings must have the same option type"
        
        self.underlyings = sorted(underlyings, key=lambda opt: opt.exercise_date.item())
        assert all(self.underlyings[i].exercise_date < self.underlyings[i+1].exercise_date for i in range(len(underlyings)-1)), "Exercise dates must be distinct"
        exercise_dates = [opt.exercise_date.item() for opt in self.underlyings]
        self.product_timeline = torch.tensor(exercise_dates, dtype=FLOAT, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.num_exercise_rights = num_exercise_rights

        self.numeraire_requests={idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) for idx, t in enumerate(self.modeling_timeline)}
        self.spot_requests={idx: AtomicRequest(AtomicRequestType.SPOT) for idx in range(len(self.modeling_timeline))}
        
        self.underlying_requests={idx: opt.underlying_request[0] for idx, opt in enumerate(self.underlyings)}

            
    def get_atomic_requests(self):
        requests = defaultdict(list)

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
        return self.num_exercise_rights + 1
    
    def get_initial_state(self):
        return self.num_exercise_rights
    
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
        _, S = state_transition_matrix.shape

        opt = self.underlyings[time_idx]
        spot = resolved_requests[1][self.underlying_requests[time_idx].get_handle()]      
        A = regression_function.get_regression_matrix(spot)                                

        immediate = opt.payoff(spot, model).unsqueeze(1).expand(-1, S)                              

        if time_idx == len(self.product_timeline) - 1:
            # No future CFs beyond the last exercise date
            continuation_not_exercised = torch.zeros_like(immediate)                  
            continuation_exercised     = torch.zeros_like(immediate)                  
        else:
            # If not exercised: continuation under current state s
            coeffs_current = self.regression_coeffs[time_idx][state_transition_matrix]    
            continuation_not_exercised = (A.unsqueeze(1) * coeffs_current).sum(dim=2)     

            # If exercised: state decrements by 1 (floor at 0), continuation under (s-1)
            state_after_ex = torch.where(
                state_transition_matrix > 0, state_transition_matrix - 1, state_transition_matrix
            )                                                                             
            coeffs_after_ex = self.regression_coeffs[time_idx][state_after_ex]            
            continuation_exercised = (A.unsqueeze(1) * coeffs_after_ex).sum(dim=2)        

        # Exercise decision: take exercise if immediate + cont(s-1) > cont(s), and rights left
        rights_left     = state_transition_matrix > 0                                     
        should_exercise = (immediate + continuation_exercised > continuation_not_exercised) & rights_left

        # Normalize immediate CFs by numeraire
        numeraire = resolved_requests[0][self.numeraire_requests[time_idx].handle]        
        numeraire_all = numeraire.unsqueeze(1).expand(-1, S)                              

        cashflows = immediate * should_exercise.float() / numeraire_all               

        # Update state (consume one right if exercised)
        next_state_transition_matrix = state_transition_matrix - should_exercise.long()   

        return next_state_transition_matrix, cashflows


