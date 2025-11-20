from metrics.metric import *
import numpy as np

class PFEMetric(Metric):
    """Metric representing Potential Future Exposure."""
    def __init__(self, quantile=0.95, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.PFE, evaluation_type=evaluation_type)
        self.quantile = quantile
        
    def _compute_quantile_mc_error(self, sorted_values: torch.Tensor, q_index: int) -> torch.Tensor:
        """
        Compute the Monte Carlo error for the quantile estimate using the formula:
        SE(q) = sqrt( q * (1 - q) / (n * f(q)^2) )
        
        where f(q) is the density at the quantile, approximated using finite differences.
        
        Args:
            sorted_values: 1D tensor of sorted exposure values [num_paths]
            q_index: index of the quantile in the sorted array
            quantile: quantile level (e.g., 0.95)   
        Returns:
            Standard error of the quantile estimate.
        """     
        num_paths = sorted_values.shape[0]
        pfe = sorted_values[q_index]

        # --- SPECIAL CASE: quantile is at a flat region (e.g. 0) ---
        # If many values are identical around the quantile, the estimator is exact.
        if q_index == 0 or q_index == num_paths - 1:
            return torch.tensor(0.0, dtype=sorted_values.dtype, device=sorted_values.device)

        # If the quantile value equals its neighbors → density undefined → MC error = 0  
        if sorted_values[q_index - 1] == pfe and sorted_values[q_index + 1] == pfe:
            return torch.tensor(0.0, dtype=sorted_values.dtype, device=sorted_values.device)

        # --- Otherwise use finite differences ---
        f_q = (sorted_values[q_index + 1] - sorted_values[q_index - 1]) / 2.0
        f_q = torch.clamp(f_q, min=1e-6)

        se = torch.sqrt(self.quantile * (1 - self.quantile) / (num_paths * f_q * f_q))
        return se

    def evaluate_analytically(self, **kwargs):
        raise NotImplementedError("Analytical PFE not implemented.")

    def evaluate_numerically(self, exposures: list[torch.Tensor], **kwargs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        exposures: list of 1D torch.Tensors
            each tensor: shape [num_paths] — exposures at a given time point
        Returns:
            list of tuples (pfe_estimate, mc_error)
        """
        results: list[tuple[torch.Tensor, torch.Tensor]] = []

        num_paths = exposures[0].shape[0]
        q_index = int(torch.ceil(torch.tensor(self.quantile * num_paths)).item()) - 1

        for e in exposures:
            sorted_vals = torch.sort(e).values

            # quantile index
            q_index = int(torch.ceil(torch.tensor(self.quantile * num_paths)).item()) - 1
            pfe = sorted_vals[q_index]

            # MC error of quantile
            se = self._compute_quantile_mc_error(sorted_vals, q_index)

            results.append((pfe, se))

        return results