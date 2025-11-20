from metrics.metric import *

class ENEMetric(Metric):
    """Expected Negative Exposure."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.ENE, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical EE not implemented.")

    def evaluate_numerically(self, exposures: list[torch.Tensor], **kwargs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        expected_exposures=[]
        for e in exposures:
            ne = -torch.relu(-e)
            expected_exposures.append(self._compute_mc_mean_and_error(ne))
        return expected_exposures