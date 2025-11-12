from metrics.metric import *

class CVAMetric(Metric):
    """ Metric for Credit Value Adjustment."""
    def __init__(self, recovery_rate, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.CVA, evaluation_type=evaluation_type)
        self.recovery_rate = recovery_rate

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical CVA not implemented.")

    def evaluate_numerically(self, exposures, survival_probs, cond_survival_probs, **kwargs):
        """
        kwargs:
            exposures:       list[Tensor] of length N, each (num_paths,)
                             exposure at t_k (already discounted to t=0)
            survival_probs:  list[Tensor] of length N+1, each (num_paths,)
                             survival S(0, t_k) per path
        """
        N = len(exposures)
        assert len(survival_probs) == N - 1, \
            "survival probability for each exposure time point except at last date."

        num_paths = exposures[0].shape[0]
        
        # Pathwise CVA accumulator
        cva_pathwise = torch.zeros(num_paths, dtype=FLOAT, device=device)

        for k in range(N-1):
            # positive exposure at t_k, per path
            e_pos = torch.relu(exposures[k])         

            survival = survival_probs[k]               # S(0, t_k) per path
            cond_survival = cond_survival_probs[k]           # S(t_k, t_{k+1}) per path
            default_prob = survival*(1-cond_survival)                      # ΔPD_k per path

            cva_pathwise += e_pos * default_prob            # already discounted case

        # Monte Carlo average and LGD
        cva = (1.0 - self.recovery_rate) * cva_pathwise.mean()

        return [cva]
