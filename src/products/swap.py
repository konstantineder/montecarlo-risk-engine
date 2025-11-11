from products.product import *
from products.bond import Bond
from request_interface.request_interface import CompositeRequest
from collections import defaultdict
from typing import Union, List, Optional
from models.model import Model

class IRSType(Enum):
    PAYER = 0
    RECEIVER = 1

# Plain Vanilla Interest Rate Swap
class InterestRateSwap(Product):
    def __init__(self, 
                 startdate   : float, # Startdate of the contract
                 enddate     : float, # Maturity of the contract
                 asset_id    : str,
                 notional    : float, # Notional (currently for both fixed and floating leg)
                 fixed_rate  : float, # Fixed rate for fixed leg
                 tenor_fixed : float, # Tenor for fixed rate coupon payments
                 tenor_float : float, # Tenor for floating coupon payments
                 irs_type    : IRSType # Specification of type of payment received (Receiver if fixed payment is received else Payer)
                 ):
        
        super().__init__()
        self.startdate = startdate
        self.enddate = enddate
        self.asset_id = asset_id
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor_fixed = tenor_fixed
        self.tenor_float = tenor_float
        self.irs_type = irs_type

        self.fixed_leg = Bond(startdate=startdate,maturity=enddate,notional=notional,tenor=tenor_fixed,pays_notional=False,fixed_rate=fixed_rate)
        self.floating_leg = Bond(startdate=startdate,maturity=enddate,notional=notional,pays_notional=False,tenor=tenor_float)

        fixed_times = {float(t.item()) for t in self.fixed_leg.modeling_timeline}
        float_times = {float(t.item()) for t in self.floating_leg.modeling_timeline}
        all_times = sorted(fixed_times.union(float_times))
        self.product_timeline = torch.tensor(all_times, dtype=FLOAT)
        self.modeling_timeline = self.product_timeline

        self.regression_timeline = torch.tensor([], dtype=FLOAT, device=device)

    def __eq__(self, other):
        return (
            isinstance(other, InterestRateSwap) and
            self.startdate == other.startdate and
            self.enddate == other.enddate and
            self.notional == other.notional and
            self.fixed_rate == other.fixed_rate and
            self.tenor_fixed == other.tenor_fixed and
            self.tenor_float == other.tenor_float
        )

    def __hash__(self):
        return hash((
            self.startdate,
            self.enddate,
            self.notional,
            self.fixed_rate,
            self.tenor_fixed,
            self.tenor_float
        ))

    def get_atomic_requests(self):
        requests = defaultdict(list)

        fixed_reqs = self.fixed_leg.get_atomic_requests()
        for t, reqs in fixed_reqs.items():
            for req in reqs:
                requests[t].append(req)

        float_reqs = self.floating_leg.get_atomic_requests()
        for t, reqs in float_reqs.items():
            for req in reqs:
                requests[t].append(req)

        return requests

    def get_atomic_requests_for_underlying(self):
        # Update time1 for composite requests with the observation_date
        atomic_requests = defaultdict(list)

        fixed_reqs = self.fixed_leg.get_atomic_requests_for_underlying()
        for t, reqs in fixed_reqs.items():
            for req in reqs:
                atomic_requests[t].append(req)

        float_reqs = self.floating_leg.get_atomic_requests_for_underlying()
        for t, reqs in float_reqs.items():
            for req in reqs:
                atomic_requests[t].append(req)

        return atomic_requests
    
    def generate_composite_requests_for_date(self, observation_date : float):
        swap=InterestRateSwap(observation_date,self.enddate,self.notional,self.fixed_rate,self.tenor_fixed,self.tenor_float,self.irs_type)
        return CompositeRequest(swap)
            
    def get_value(self, resolved_atomic_requests):
        """
        Present value of floating - fixed leg, discounted
        Floating leg: use resolved LIBOR requests
        Fixed leg: use fixed_rate
        """
        fixed_value = self.fixed_leg.get_value(resolved_atomic_requests)
        float_value = self.floating_leg.get_value(resolved_atomic_requests)

        total_value = float_value - fixed_value if self.irs_type == IRSType.RECEIVER else fixed_value - float_value

        return  total_value

    def compute_normalized_cashflows(
            self,
            time_idx: int,
            model: Model,
            resolved_requests: List[dict],
            regression_RegressionFunction: Optional[object] = None,
            state: Optional[torch.Tensor] = None
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        
        total_value = torch.zeros_like(resolved_requests[0][0], dtype=FLOAT, device=device).unsqueeze(1)

        time = self.modeling_timeline[time_idx]

        fixed_cashflow = torch.zeros_like(total_value)
        float_cashflow = torch.zeros_like(total_value)

        if time in self.fixed_leg.modeling_timeline:
            fixed_time_idx = (self.fixed_leg.modeling_timeline == time).nonzero(as_tuple=True)[0].item()
            _, fixed_cashflow = self.fixed_leg.compute_normalized_cashflows(
                fixed_time_idx, model, resolved_requests, regression_RegressionFunction, state
            )

        if time in self.floating_leg.modeling_timeline:
            float_time_idx = (self.floating_leg.modeling_timeline == time).nonzero(as_tuple=True)[0].item()
            _, float_cashflow = self.floating_leg.compute_normalized_cashflows(
                float_time_idx, model, resolved_requests, regression_RegressionFunction, state
            )

        total_value = float_cashflow - fixed_cashflow if self.irs_type == IRSType.RECEIVER else fixed_cashflow - float_cashflow

        return state, total_value

