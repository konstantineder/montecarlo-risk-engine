from common.packages import *
from enum import Enum

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

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical evaluation not implemented.")
    
    def evaluate_numerically(self, **kwargs):
        raise NotImplementedError("Numerical evluation not implemented.")

    def evaluate(self,**kwargs):
        if self.evaluation_type==Metric.EvaluationType.NUMERICAL:
            return self.evaluate_numerically(**kwargs)
        else:
            return self.evaluate_analytically(**kwargs)
        
