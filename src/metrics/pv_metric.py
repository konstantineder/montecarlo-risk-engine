from metrics.metric import *

class PVMetric(Metric):
    """Metric to compute present value."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.PV, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical PV not implemented.")

    def evaluate_numerically(self, cfs, **kwargs):
        return [cfs.mean()]