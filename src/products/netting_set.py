from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from common.packages import FLOAT
from products.product import Product


@dataclass
class NettingSet:
    """Container for products that should be valued on a netted basis."""

    name: str
    products: Sequence[Product]
    threshold: float = 0.0
    margin_period_of_risk: float | None = None
    counterparty_id: str | None = None
    collateral_interpolation: str = "linear"

    def __post_init__(self):
        self.products = list(self.products)
        if len(self.products) == 0:
            raise ValueError("A netting set must contain at least one product.")
        if self.threshold < 0.0:
            raise ValueError("Netting set threshold must be non-negative.")
        if self.margin_period_of_risk is not None and self.margin_period_of_risk < 0.0:
            raise ValueError("Netting set margin period of risk must be non-negative.")
        if self.collateral_interpolation not in {"linear", "previous"}:
            raise ValueError(
                "Collateral interpolation must be one of {'linear', 'previous'}."
            )

    def get_name(self) -> str:
        return self.name

    def is_collateralized(self) -> bool:
        return self.margin_period_of_risk is not None

    def get_collateral_query_times(self, exposure_timeline: torch.Tensor) -> torch.Tensor:
        if not self.is_collateralized():
            return torch.zeros(0, dtype=exposure_timeline.dtype, device=exposure_timeline.device)
        delayed_times = exposure_timeline - self.margin_period_of_risk
        return delayed_times[delayed_times >= 0.0]

    def apply_threshold(self, exposures: torch.Tensor) -> torch.Tensor:
        """
        Apply a symmetric exposure threshold.

        Positive netted exposure is reduced by ``threshold`` before it contributes
        to positive-side metrics; negative exposure is reduced in magnitude by the
        same amount. Values inside the threshold band are mapped to zero.
        """
        if exposures.numel() == 0 or self.threshold == 0.0:
            return exposures

        threshold = torch.tensor(
            self.threshold,
            dtype=exposures.dtype if exposures.is_floating_point() else FLOAT,
            device=exposures.device,
        )
        return torch.where(
            exposures > threshold,
            exposures - threshold,
            torch.where(
                exposures < -threshold,
                exposures + threshold,
                torch.zeros_like(exposures),
            ),
        )

    def _interpolate_exposure_profiles(
        self,
        netted_exposures: torch.Tensor,
        exposure_timeline: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        if netted_exposures.numel() == 0:
            return netted_exposures

        before_start = (query_times < exposure_timeline[0]).unsqueeze(1)
        num_dates = exposure_timeline.shape[0]

        if self.collateral_interpolation == "previous":
            previous_idx = torch.searchsorted(exposure_timeline, query_times, right=True) - 1
            previous_idx = torch.clamp(previous_idx, min=0, max=num_dates - 1)
            interpolated = netted_exposures.index_select(0, previous_idx)
            return torch.where(before_start, torch.zeros_like(interpolated), interpolated)

        right_idx = torch.searchsorted(exposure_timeline, query_times)
        right_idx = torch.clamp(right_idx, max=num_dates - 1)
        left_idx = torch.clamp(right_idx - 1, min=0)

        left_times = exposure_timeline.index_select(0, left_idx)
        right_times = exposure_timeline.index_select(0, right_idx)
        left_values = netted_exposures.index_select(0, left_idx)
        right_values = netted_exposures.index_select(0, right_idx)

        denom = right_times - left_times
        weights = torch.where(
            denom > 0.0,
            (query_times - left_times) / denom,
            torch.zeros_like(query_times),
        ).unsqueeze(1)
        interpolated = left_values + weights * (right_values - left_values)
        return torch.where(before_start, torch.zeros_like(interpolated), interpolated)

    def compute_collateral_profile(
        self,
        netted_exposures: torch.Tensor,
        exposure_timeline: torch.Tensor,
        metric_exposure_indices: torch.Tensor | None = None,
        delayed_exposure_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the collateral balance on the exposure grid.

        Exposures are assumed to be discounted to time 0. Under the engine's
        numeraire convention, a collateral balance set to the netting set PV at
        ``t - MPoR`` and accruing at the collateral rate remains equal to the
        discounted exposure profile observed at ``t - MPoR``. The threshold is
        interpreted as the uncollateralized band around zero, so the collateral
        call equals the threshold-adjusted delayed exposure.
        """
        if not self.is_collateralized() or netted_exposures.numel() == 0:
            if metric_exposure_indices is not None:
                return torch.zeros(
                    (metric_exposure_indices.shape[0], netted_exposures.shape[1]),
                    dtype=netted_exposures.dtype,
                    device=netted_exposures.device,
                )
            return torch.zeros_like(netted_exposures)

        if metric_exposure_indices is not None and delayed_exposure_indices is not None:
            metric_netted_exposures = netted_exposures.index_select(0, metric_exposure_indices)
            collateral_profile = torch.zeros_like(metric_netted_exposures)
            valid = delayed_exposure_indices >= 0
            if torch.any(valid):
                delayed_netted_exposures = netted_exposures.index_select(
                    0,
                    delayed_exposure_indices[valid],
                )
                collateral_profile[valid] = self.apply_threshold(delayed_netted_exposures)
            return collateral_profile

        query_times = exposure_timeline - self.margin_period_of_risk
        delayed_netted_exposures = self._interpolate_exposure_profiles(
            netted_exposures=netted_exposures,
            exposure_timeline=exposure_timeline,
            query_times=query_times,
        )
        return self.apply_threshold(delayed_netted_exposures)

    def compute_unsecured_exposure_profiles(
        self,
        netted_exposures: torch.Tensor,
        exposure_timeline: torch.Tensor,
        metric_exposure_indices: torch.Tensor | None = None,
        delayed_exposure_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute unsecured netted exposures after thresholding/collateralization.
        """
        if netted_exposures.numel() == 0:
            return netted_exposures

        metric_netted_exposures = (
            netted_exposures.index_select(0, metric_exposure_indices)
            if metric_exposure_indices is not None
            else netted_exposures
        )

        if not self.is_collateralized():
            return self.apply_threshold(metric_netted_exposures)

        collateral_profile = self.compute_collateral_profile(
            netted_exposures=netted_exposures,
            exposure_timeline=exposure_timeline,
            metric_exposure_indices=metric_exposure_indices,
            delayed_exposure_indices=delayed_exposure_indices,
        )
        return metric_netted_exposures - collateral_profile
