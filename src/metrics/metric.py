from common.packages import *
from enum import Enum
from request_interface.request_types import AtomicRequest
from collections import defaultdict

# Enum for metric types 
class MetricType(Enum):
    PV = 0
    CE = 1
    EPE = 2
    ENE = 3
    PFE = 4
    EEPE = 5
    CVA = 6

class Metric:
    """Base metric class to be overwritten by all spectific metric implementations."""
    class EvaluationType(Enum):
         ANALYTICAL=0
         NUMERICAL=1

    def __init__(self, metric_type, evaluation_type):
        self.metric_type=metric_type
        self.evaluation_type=evaluation_type
        
    def set_requests(self, exposure_timeline: torch.Tensor) -> None:
        pass
    
    def get_requests(self) -> dict[tuple[int, str], list[AtomicRequest]]:
        requests: dict[tuple[int, str], list[AtomicRequest]] = defaultdict(list)
        return requests
    
    def get_counterparty_ids(self) -> list[str] | None:
        return None

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical evaluation not implemented.")
    
    def evaluate_numerically(self, **kwargs):
        raise NotImplementedError("Numerical evluation not implemented.")

    def evaluate(self,**kwargs):
        if self.evaluation_type==Metric.EvaluationType.NUMERICAL:
            return self.evaluate_numerically(**kwargs)
        else:
            return self.evaluate_analytically(**kwargs)
        
