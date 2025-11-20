from metrics.metric import *

class EPEMetric(Metric):
    """Expected Positive Exposure."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.EPE, evaluation_type=evaluation_type)

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical EE not implemented.")

    def evaluate_numerically(self, exposures: list[torch.Tensor], **kwargs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        expected_exposures=[]
        for e in exposures:
            pe = torch.relu(e)
            expected_exposures.append(self._compute_mc_mean_and_error(pe))
        return expected_exposures