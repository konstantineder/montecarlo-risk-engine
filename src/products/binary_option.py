from products.product import *
from maths.maths import compute_degree_of_truth
from request_interface.request_types import AtomicRequestType, AtomicRequest
from collections import defaultdict

class BinaryOption(Product):
    """
    Binary (digital) option
    """
    def __init__(
        self, 
        maturity      : float, 
        strike        : float, 
        payment_amount: float, 
        option_type   : OptionType,
        asset_id      : str | None = None,
    ):
        
        super().__init__(
            asset_ids=[asset_id],
            product_family=ProductFamily.BINARY_TERMINAL_PAYOFF,
        )
        self.maturity = torch.tensor([maturity], dtype=FLOAT,device=device)
        self.strike = torch.tensor([strike], dtype=FLOAT,device=device)
        self.option_type = option_type
        self.payment_amount = torch.tensor([payment_amount], dtype=FLOAT,device=device)
        self.product_timeline=torch.tensor([maturity], dtype=FLOAT,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=FLOAT,device=device)

        self.numeraire_requests={0: AtomicRequest(AtomicRequestType.NUMERAIRE,maturity)}
        asset_id = self.get_asset_id()
        self.spot_requests={(0, asset_id): AtomicRequest(AtomicRequestType.SPOT)}

    # Operator overloading of the payoff computation method
    # This method computes the payoff of a binary option based on the option type    
    def payoff(self, spots, model):
        is_strike_above_barrier = compute_degree_of_truth(spots - self.strike,True,1)
        if self.option_type == OptionType.CALL:
            return self.payment_amount*is_strike_above_barrier
        else:
            return self.payment_amount*(1-is_strike_above_barrier)

    # Black-Scholes closed-form pricing for European options (see chapter 1)
    def compute_pv_analytically(self, model):
        spot=model.get_spot()
        rate=model.get_rate()
        sigma=model.get_volatility()

        norm = torch.distributions.Normal(0.0, 1.0)

        d2 = (torch.log(spot / self.strike) + (rate - 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
        if self.option_type == OptionType.CALL:
            return self.payment_amount * torch.exp(-rate * self.maturity) * norm.cdf(d2)
        else:
            return self.payment_amount * torch.exp(-rate * self.maturity) * norm.cdf(-d2)
    
    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_RegressionFunction=None,state=None):
        asset_id = self.get_asset_id()
        spots = resolved_requests[0][self.spot_requests[(0, asset_id)].handle]
        numeraire = resolved_requests[0][self.numeraire_requests[0].handle]
        normalized = self.payoff(spots, model) / numeraire
        return state, normalized.unsqueeze(1)
