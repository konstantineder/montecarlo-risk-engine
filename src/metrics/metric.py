from common.packages import *
from enum import Enum
from request_interface.request_types import AtomicRequest
from collections import defaultdict

# Enum for metric types 
class MetricType(Enum):
    PV = "Present Value"
    CE = "Current Exposure"
    EPE = "Expected Positive Exposure"
    ENE = "Expected Negative Exposure"
    PFE = "Potential Future Exposure"
    EEPE = "Effective Expected Positive Exposure"
    CVA = "Credit Valuation Adjustment"

class Metric:
    """Base metric class to be overwritten by all spectific metric implementations."""
    class EvaluationType(Enum):
         ANALYTICAL = "Analytical"
         NUMERICAL = "Numerical"

    def __init__(self, metric_type, evaluation_type):
        self.metric_type=metric_type
        self.evaluation_type=evaluation_type
        
    def _compute_mc_mean_and_error(self, values: torch.Tensor):
        """
        values: tensor [num_paths]
        Returns: (mean, mc_error)
        """
        num_paths = values.shape[0]
        mean = values.mean()
        sigma = values.std(unbiased=True)
        mc_error = sigma / torch.sqrt(torch.tensor(num_paths, dtype=FLOAT, device=device))
        return mean, mc_error
        
    def set_requests(self, exposure_timeline: torch.Tensor) -> None:
        pass
    
    def get_requests(self) -> dict[tuple[int, str], list[AtomicRequest]]:
        requests: dict[tuple[int, str], list[AtomicRequest]] = defaultdict(list)
        return requests
    
    def get_counterparty_ids(self) -> list[str] | None:
        return None

    def get_name(self) -> str:
        return self.metric_type.name.lower()

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical evaluation not implemented.")
    
    def evaluate_numerically(self, **kwargs):
        raise NotImplementedError("Numerical evluation not implemented.")

    def evaluate(self,**kwargs):
        if self.evaluation_type==Metric.EvaluationType.NUMERICAL:
            return self.evaluate_numerically(**kwargs)
        else:
            return self.evaluate_analytically(**kwargs)
        
