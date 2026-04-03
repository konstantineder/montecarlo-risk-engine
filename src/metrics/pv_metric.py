from metrics.metric import *

class PVMetric(Metric):
    """Metric to compute present value."""
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.PV, evaluation_type=evaluation_type)

    def evaluate_analytically(self, product=None, model=None, **kwargs):
        if product is None or model is None:
            raise ValueError("Analytical PV evaluation requires both product and model.")

        pv = product.compute_pv_analytically(model)
        pv = pv.squeeze()
        mc_error = torch.zeros_like(pv)
        return [(pv, mc_error)]

    def evaluate_numerically(self, cfs: torch.Tensor, **kwargs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self._compute_mc_mean_and_error(cfs)]
