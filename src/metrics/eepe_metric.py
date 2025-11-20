from metrics.metric import *

class EEPEMetric(Metric):
    """Metric for Effective Expected Positive Exposure."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.EEPE, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical EEPE not implemented.")

    def evaluate_numerically(self, exposures, **kwargs):
        expected_exposures=[]
        for e in exposures:
            ee = torch.relu(e).mean()
            expected_exposures.append(ee)
        return [self._compute_mc_mean_and_error(torch.stack(expected_exposures))]  # Return for each time point