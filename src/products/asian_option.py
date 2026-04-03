from products.product import *
from request_interface.request_types import AtomicRequestType, AtomicRequest
from typing import Sequence


class AsianAveragingType(Enum):
    ARITHMETIC = 0
    GEOMETRIC = 1


class AsianOption(Product):
    def __init__(
        self,
        startdate: float,
        maturity: float,
        strike: float,
        num_observation_timepoints: int,
        option_type: OptionType,
        averaging_type: AsianAveragingType = AsianAveragingType.ARITHMETIC,
        asset_id: str | None = None,
    ):
        super().__init__(
            asset_ids=[asset_id],
            product_family=ProductFamily.ASIAN_PATH_TERMINAL,
        )
        self.maturity = torch.tensor([maturity], dtype=FLOAT, device=device)
        self.strike = torch.tensor([strike], dtype=FLOAT, device=device)
        self.option_type = option_type
        self.averaging_type = averaging_type

        self.product_timeline = torch.tensor([maturity], dtype=FLOAT, device=device)
        self.modeling_timeline = torch.linspace(
            startdate,
            maturity,
            num_observation_timepoints,
            dtype=FLOAT,
            device=device,
        )
        self.regression_timeline = torch.tensor([], dtype=FLOAT, device=device)

        self.numeraire_requests = {
            idx: AtomicRequest(AtomicRequestType.NUMERAIRE, t)
            for idx, t in enumerate(self.modeling_timeline)
        }
        asset = self.get_asset_id()
        self.spot_requests = {
            (idx, asset): AtomicRequest(AtomicRequestType.SPOT)
            for idx in range(len(self.modeling_timeline))
        }

    @staticmethod
    def _average_paths(
        spots: torch.Tensor,
        averaging_type: AsianAveragingType,
    ) -> torch.Tensor:
        if averaging_type == AsianAveragingType.GEOMETRIC:
            return torch.exp(torch.mean(torch.log(spots + 1e-10), dim=1))
        return torch.mean(spots, dim=1)

    @staticmethod
    def _payoff_from_average(
        average: torch.Tensor,
        strike: torch.Tensor,
        option_type: OptionType,
    ) -> torch.Tensor:
        zero = torch.tensor([0.0], dtype=FLOAT, device=average.device)
        if option_type == OptionType.CALL:
            return torch.maximum(average - strike, zero)
        return torch.maximum(strike - average, zero)

    def payoff(self, spots, model):
        average = AsianOption._average_paths(spots, self.averaging_type)
        return AsianOption._payoff_from_average(average, self.strike, self.option_type)

    def compute_normalized_cashflows(
        self,
        time_idx: int,
        model,
        resolved_requests,
        regression_RegressionFunction=None,
        state=None,
    ):
        monitored_paths = torch.stack(
            [
                resolved_requests[0][self.spot_requests[(idx, self.get_asset_id())].handle]
                for idx in range(len(self.modeling_timeline))
            ],
            dim=1,
        )
        numeraire = resolved_requests[0][self.numeraire_requests[len(self.product_timeline) - 1].handle]
        normalized = self.payoff(monitored_paths, model) / numeraire
        return state, normalized.unsqueeze(1)

    def compute_pv_analytically(self, model: Model) -> torch.Tensor:
        raise NotImplementedError("Analytical Asian pricing is not implemented for this product.")
