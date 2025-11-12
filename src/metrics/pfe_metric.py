from metrics.metric import *
import numpy as np

class PFEMetric(Metric):
    """Metric representing Potential Future Exposure."""
    def __init__(self, quantile=0.95, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.PFE, evaluation_type=evaluation_type)
        self.quantile = quantile

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical PFE not implemented.")

    def evaluate_numerically(self, exposures, **kwargs):
        pfes=[]
        index = int(np.ceil(self.quantile * exposures[0].shape[0])) - 1
        for e in exposures:
            sorted_vals = e.sort().values
            pfes.append(sorted_vals[index])
        return pfes  # Return for each time point