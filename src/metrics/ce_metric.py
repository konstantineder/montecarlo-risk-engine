from metrics.metric import *

class CEMetric(Metric):
    """Current Exposure Metric."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.CE, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical EE not implemented.")

    def evaluate_numerically(self, exposures, **kwargs):
        current_exposures=torch.relu(exposures[0])
        return [current_exposures.mean()]