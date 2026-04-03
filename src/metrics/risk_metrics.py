from common.packages import *

from metrics.metric import Metric, MetricType
from enum import Enum
import numpy as np
import torch


class PathwisePrimitive(Enum):
    DISCOUNTED_CASHFLOWS = "discounted_cashflows"
    EXPOSURE_PROFILES = "exposure_profiles"


class RiskMetrics:
    """Collection of metrics to be evaluted during MC simulation."""
    
    def __init__(self, metrics: list[Metric], exposure_timeline: np.ndarray | None = None):
        self.metrics = metrics
        if exposure_timeline is None:
            exposure_timeline = []
            
        self.exposure_timeline = torch.tensor(exposure_timeline, dtype=FLOAT, device=device)
        
        self.any_pv = any(m.metric_type == MetricType.PV for m in metrics)
        self.any_xva = any(m.metric_type == MetricType.CVA for m in metrics)
        self.any_exposure = any(m.metric_type != MetricType.PV for m in metrics)
        required_primitives: list[PathwisePrimitive] = []
        if self.any_pv:
            required_primitives.append(PathwisePrimitive.DISCOUNTED_CASHFLOWS)
        if self.any_exposure:
            required_primitives.append(PathwisePrimitive.EXPOSURE_PROFILES)
        self._required_primitives = frozenset(required_primitives)
        if self.any_exposure:
            assert len(exposure_timeline) > 0, \
                "For exposure simulation at least one exposure time point needs to be provided."

            
        for metric in self.metrics:
            metric.set_requests(exposure_timeline)
            
        self.counterparty_ids: list[str] = []
        for metric in self.metrics:
            cp_ids = metric.get_counterparty_ids()
            if cp_ids is not None:
                for cp_id in cp_ids:
                    self.counterparty_ids.append(cp_id)

    def requires_discounted_cashflows(self) -> bool:
        return self.requires_primitive(PathwisePrimitive.DISCOUNTED_CASHFLOWS)

    def requires_exposure_profiles(self) -> bool:
        return self.requires_primitive(PathwisePrimitive.EXPOSURE_PROFILES)

    def required_pathwise_primitives(self) -> frozenset[PathwisePrimitive]:
        return self._required_primitives

    def requires_primitive(self, primitive: PathwisePrimitive) -> bool:
        return primitive in self._required_primitives
                    
    def evaluate(self,**kwargs) -> list[torch.Tensor]:
        """Evaluate all metrics and return results as a list."""
        results = []
        for metric in self.metrics:
            eval_val = metric.evaluate(**kwargs)
            results.append(eval_val)
        return results
                
            
        
