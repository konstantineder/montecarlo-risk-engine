from metrics.metric import *

class EPEMetric(Metric):
    """Expected Positive Exposure."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.EPE, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical EE not implemented.")

    def evaluate_numerically(self, exposures, **kwargs):
        expected_exposures=[]
        for e in exposures:
            ee = torch.relu(e).mean()
            expected_exposures.append(ee)
        return expected_exposures