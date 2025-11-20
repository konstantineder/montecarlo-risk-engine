from common.packages import *
from metrics.metric import Metric, MetricType
from request_interface.request_types import AtomicRequest, AtomicRequestType
from collections import defaultdict
import numpy as np

class CVAMetric(Metric):
    """ Metric for Credit Value Adjustment."""
    def __init__(self, counterparty_id: str, recovery_rate: float, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.CVA, evaluation_type=evaluation_type)
        self.counterparty_id = counterparty_id
        self.recovery_rate = recovery_rate
        
        self.survival_prob_requests: dict[tuple[int, str], AtomicRequest] = {}
        self.cond_survival_prob_requests: dict[tuple[int, str], AtomicRequest] = {}
        
    def get_counterparty_ids(self) -> list[str] | None:
        return [self.counterparty_id]
        
    def set_requests(self, exposure_timeline: np.ndarray) -> None:
        for idx in range(len(exposure_timeline)-1):
            cp = self.counterparty_id
            label = (idx, cp)
            self.cond_survival_prob_requests[label] = AtomicRequest(
                    AtomicRequestType.CONDITIONAL_SURVIVAL_PROBABILITY,
                    time1 = exposure_timeline[idx],
                    time2 = exposure_timeline[idx+1],  
                )
            self.survival_prob_requests[label] = AtomicRequest(
                    AtomicRequestType.SURVIVAL_PROBABILITY,
                )
    
    def get_requests(self) -> dict[tuple[int, str], list[AtomicRequest]]:
        requests: dict[tuple[int, str], list[AtomicRequest]] = defaultdict(list)
        for label, req in self.survival_prob_requests.items():
            requests[label].append(req)
            
        for label, req in self.cond_survival_prob_requests.items():
            requests[label].append(req)
            
        return requests
            
    def _get_survival_probs(self, resolved_requests: list[dict]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        survival_probs: list[torch.Tensor] = []
        cond_survival_probs: list[torch.Tensor] = []
        for _, req in self.survival_prob_requests.items():
            survival_prob = resolved_requests[0][req.handle]
            survival_probs.append(survival_prob)
            
        for _, req in self.cond_survival_prob_requests.items():
            cond_survival_prob = resolved_requests[0][req.handle]
            cond_survival_probs.append(cond_survival_prob)
            
        return (survival_probs, cond_survival_probs)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical CVA not implemented.")

    def evaluate_numerically(
        self, 
        exposures: list[torch.Tensor], 
        resolved_requests: list[dict], 
        **kwargs
    ) -> list[torch.Tensor]:
        """
        kwargs:
            exposures:       list[Tensor] of length N, each (num_paths,)
                             exposure at t_k (already discounted to t=0)
            survival_probs:  list[Tensor] of length N-1, each (num_paths,)
                             survival S(0, t_k) per path
            cond_survival_probs:  list[Tensor] of length N-1, each (num_paths,)
                    cond survival S(t_k, t_{k+1}) per path
        """
        survival_probs, cond_survival_probs = self._get_survival_probs(resolved_requests=resolved_requests)
        
        N = len(exposures)
        assert len(survival_probs) == N - 1, \
            "survival probability for each exposure time point except at last date."

        num_paths = exposures[0].shape[0]
        
        # Pathwise CVA accumulator
        cva_pathwise = torch.zeros(num_paths, dtype=FLOAT, device=device)
        a=0
        for k in range(N-1):
            # positive exposure at t_k, per path
            e_pos = torch.relu(exposures[k])         

            survival = survival_probs[k]               # S(0, t_k) per path
            cond_survival = cond_survival_probs[k]           # S(t_k, t_{k+1}) per path
            default_prob = survival*(1-cond_survival)  
            cva_pathwise += e_pos * default_prob   

        # Monte Carlo average and LGD
        cva = (1.0 - self.recovery_rate) * cva_pathwise.mean()

        return [cva]
