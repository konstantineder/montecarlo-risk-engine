from __future__ import annotations

from common.packages import *
from enum import Enum
from collections import defaultdict
from numpy.typing import NDArray
from typing import Union, List, Optional, Dict, Any, Sequence
from request_interface.request_types import AtomicRequest, AtomicRequestType, UnderlyingRequest
from maths.regression import RegressionFunction
from models.model import Model

# Enum for option types
class OptionType(Enum):
    CALL = 1
    PUT = 2

class SettlementType(Enum):
    PHYSICAL=0
    CASH=1
    
class Product:
    """Abstract (base) class for financial products."""
    def __init__(self, asset_ids: list[str] | None = None, product_id: int = 0):
        self.asset_ids = asset_ids if asset_ids else [""]
        self.product_id=product_id

        self.spot_requests: dict[tuple[int, str], AtomicRequest] = {}
        self.numeraire_requests: dict[int, AtomicRequest] = {}
        self.libor_requests: dict[tuple[int, str], AtomicRequest] = {}
        
        self.underlying_requests: dict[int, Product] = {}
        
        self.product_timeline: torch.Tensor | None = None
        self.modeling_timeline: torch.Tensor | None = None
        self.regression_timeline: torch.Tensor | None = None

        self.regression_coeffs: torch.Tensor | None = None

    # Return all atomic requests
    def get_atomic_requests(self) -> dict[tuple[int, str], list[AtomicRequest]]:
        requests: dict[tuple[int, str], list[AtomicRequest]] = defaultdict(list)

        for t, req in self.numeraire_requests.items():
            label = (t, "numeraire")
            requests[label].append(req)

        for label, req in self.spot_requests.items():
            requests[label].append(req)
            
        for label, req in self.libor_requests.items():
            requests[label].append(req)

        return requests
    
    # Return all atomic requests for a particular underlying product
    def get_atomic_requests_for_underlying(self) -> dict[tuple[int, str], list[AtomicRequest]]:
        return defaultdict(list)
    
    # Generate underlying requests at a particular date
    def generate_underlying_requests_for_date(self, observation_date: float) -> UnderlyingRequest:
        return UnderlyingRequest(Product)
    
    # Return all underlying requests of a product
    def get_underlying_requests(self) -> dict[int, list[Product]]:
        requests: dict[int, list[Product]] = defaultdict(list)
        for t, req in self.underlying_requests.items():
            requests[t].append(req)

        return requests

    def get_num_states(self):
        return 1
    
    def get_asset_id(self, id: int | None = None):
        return self.asset_ids[id] if id else self.asset_ids[0]
    
    def get_resolved_atomic_request(
        self, 
        resolved_atomic_requests: dict[int, torch.Tensor], 
        request_type: AtomicRequestType,
        time_idx: int, 
        asset_id: str | None = None,
    ) -> torch.Tensor:
        
        if request_type == AtomicRequestType.NUMERAIRE:
            key = time_idx
            handle = self.numeraire_requests.get(key).handle
            return resolved_atomic_requests.get(handle)
        if request_type == AtomicRequestType.SPOT:
            key = (time_idx, asset_id)
            handle = self.spot_requests.get(key).handle
            return resolved_atomic_requests.get(handle)
        if request_type == AtomicRequestType.LIBOR_RATE:
            key = (time_idx, asset_id)
            handle = self.libor_requests.get(key).handle
            return resolved_atomic_requests.get(handle)
        if request_type == AtomicRequestType.FORWARD_RATE:
            key = (time_idx, asset_id)
            handle = self.libor_requests.get(key).handle
            return resolved_atomic_requests.get(handle)
    
    def _allocate_regression_coeffs(self, regression_function: RegressionFunction):
        num_time_points = len(self.regression_timeline) 
        num_states = self.get_num_states()
        degree = regression_function.get_degree()

        self.regression_coeffs = torch.zeros(
            (num_time_points, num_states, degree),
            dtype=FLOAT,
            device=device,
        )

    def get_initial_state(self):
        return 0

    # Abstract method to compute the payoff for the specific product 
    def compute_payoff(self, paths, model):
        raise NotImplementedError
    
    def compute_normalized_cashflows(
        self,
        time_idx: int,
        model: Model,
        resolved_requests: list,
        regression_function: RegressionFunction,
        state_matrix: torch.Tensor | None = None,
    ):
        raise NotImplementedError
    
    # Abstract method to compute the pv using an analytic formula  
    def compute_pv_analytically(self, model: Model) -> torch.Tensor:
        raise NotImplementedError










    


