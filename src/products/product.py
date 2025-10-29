from common.packages import *
from enum import Enum
from collections import defaultdict
from numpy.typing import NDArray
from typing import Union, List, Optional, Dict, Any, Sequence
from request_interface.request_interface import CompositeRequest
from maths.regression import RegressionFunction
from models.model import Model

# Enum for option types
class OptionType(Enum):
    CALL = 1
    PUT = 2

class SettlementType(Enum):
    PHYSICAL=0
    CASH=1

# Abstract (base) class for financial products
class Product:
    def __init__(self, product_id=0):
        self.product_id=product_id
        self.numeraire_requests=[]
        self.spot_requests=[]
        self.product_timeline=torch.tensor([], dtype=FLOAT, device=device)
        self.modeling_timeline = torch.tensor([], dtype=FLOAT, device=device)
        self.regression_timeline = torch.tensor([], dtype=FLOAT, device=device)

        self.regression_coeffs = []

    def get_atomic_requests(self):
        return defaultdict(list)
    
    def get_composite_requests(self):
        return defaultdict(list)
    
    def get_atomic_requests_for_underlying(self):
        return defaultdict(list)
    
    def generate_composite_requests_for_date(self, 
                                             observation_date: float
                                             ) -> CompositeRequest:
        return CompositeRequest(Product)

    def get_num_states(self):
        return 1
    
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
        model,
        resolved_requests,
        regression_function,
        state_matrix: torch.Tensor,
    ):
        raise NotImplementedError
    
    # Abstract method to compute the pv using an analytic formula  
    def compute_pv_analytically(self, model):
        raise NotImplementedError










    


