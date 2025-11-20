from products.product import *
from math import pi
from request_interface.request_types import AtomicRequestType, AtomicRequest
from collections import defaultdict
from typing import Sequence, List, Optional, Dict, Any
from models.model import Model
from models.black_scholes import BlackScholesModel
from models.vasicek import VasicekModel
from products.bond import Bond

class EuropeanOption(Product):
    """
    European option implementation
    Payoff depends on specification of the underlying
    Current product classes supported for underlying: Equity, Bond and Swap
    """
    def __init__(
        self, 
        underlying     : Product, 
        exercise_date  : float, 
        strike         : float,
        option_type    : OptionType,
        asset_id       : str | None = None,
    ):
        
        super().__init__(asset_ids=[asset_id])
        self.exercise_date = torch.tensor([exercise_date], dtype=FLOAT,device=device)
        self.strike = torch.tensor([strike], dtype=FLOAT,device=device)
        self.option_type = option_type
        self.product_timeline=torch.tensor([exercise_date], dtype=FLOAT,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=FLOAT,device=device)
        self.underlying=underlying

        self.numeraire_requests={0: AtomicRequest(AtomicRequestType.NUMERAIRE,exercise_date)}
        self.underlying_requests={0: underlying.generate_underlying_requests_for_date(exercise_date)}

    def payoff(
        self, 
        spots: Sequence[torch.Tensor], 
        model: Model
    ):
        
        zero = torch.tensor([0.0], dtype=FLOAT, device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(spots - self.strike, zero)
        else:
            return torch.maximum(self.strike - spots, zero)

    def compute_normalized_cashflows(
        self, 
        time_idx: int, 
        model: Model, 
        resolved_requests: List[dict], 
        regression_function: RegressionFunction | None = None, 
        state: torch.Tensor | None = None,
    ):
        
        spots=resolved_requests[1][self.underlying_requests[time_idx].get_handle()]
        cfs = self.payoff(spots,model)

        numeraire=self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_requests[0],
            request_type=AtomicRequestType.NUMERAIRE,
            time_idx=time_idx,
        )
        
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs.unsqueeze(1)

    def compute_pv_analytically(self, model: BlackScholesModel):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()

        d1 = (torch.log(spot / self.strike) + (rate + 0.5 * sigma ** 2) * self.exercise_date) / (sigma * torch.sqrt(self.exercise_date))
        d2 = d1 - sigma * torch.sqrt(self.exercise_date)

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return spot * norm.cdf(d1) - self.strike * torch.exp(-rate * self.exercise_date) * norm.cdf(d2)
        else:
            return self.strike * torch.exp(-rate * self.exercise_date) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
    def compute_pv_bond_option_analytically(self, model: VasicekModel):

        if not isinstance(self.underlying, Bond):
            raise TypeError("Expected self.underlying to be of type Bond")

        a=model.get_mean_reversion_speed()
        rate=model.get_rate()
        sigma=model.get_volatility()
        calibration_date=model.calibration_date

        bond_price_at_exercise_date=model.compute_bond_price(calibration_date,self.exercise_date,rate)
        bond_price_at_underlying_maturity=model.compute_bond_price(calibration_date,self.underlying.maturity,rate)

        B_TS=(1-torch.exp(-a*(self.underlying.maturity-self.exercise_date)))/a
        sigma_tilde=sigma*torch.sqrt((1-torch.exp(-2*a*(self.exercise_date-calibration_date)))/(2*a))*B_TS

        d1 = (torch.log(bond_price_at_underlying_maturity / (bond_price_at_exercise_date * self.strike)) + 0.5 * sigma_tilde ** 2) / sigma_tilde
        d2 = d1 - sigma_tilde

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return bond_price_at_underlying_maturity * norm.cdf(d1) - self.strike * bond_price_at_exercise_date* norm.cdf(d2)
        else:
            return self.strike * bond_price_at_exercise_date * norm.cdf(-d2) - bond_price_at_underlying_maturity * norm.cdf(-d1)
        
    def compute_dVegadSigma_analytically(self, model: BlackScholesModel):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()
        T = self.exercise_date

        d1 = (torch.log(spot / self.strike) + (rate+0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2.0 * pi, device=device))  # φ(d1)

        # Vomma (Volga) formula: S * φ(d1) * sqrt(T) * d1 * d2 / σ
        vomma = spot * pdf_d1 * torch.sqrt(T) * d1 * d2 / sigma

        return vomma
    
    def compute_dDeltadSpot_analytically(self, model: BlackScholesModel):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()
        T = self.exercise_date

        d1 = (torch.log(spot / self.strike) + (rate+0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2.0 * pi, device=device))  # φ(d1)

        # Vomma (Volga) formula: S * φ(d1) * sqrt(T) * d1 * d2 / σ
        gamma = pdf_d1/(spot*sigma*torch.sqrt(T))

        return gamma