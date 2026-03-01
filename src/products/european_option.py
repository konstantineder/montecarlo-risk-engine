from products.product import *
from math import pi
from request_interface.request_types import AtomicRequestType, AtomicRequest
from collections import defaultdict
from typing import Sequence, List, Optional, Dict, Any
from models.model import Model
from models.black_scholes import BlackScholesModel
from models.heston import HestonModel
from models.vasicek import VasicekModel
from products.bond import Bond
from scipy.integrate import quad
import numpy as np

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
        
    def compute_pv_analytically_heston(self, model: HestonModel):
        if not isinstance(model, HestonModel):
            raise TypeError("Expected model to be of type HestonModel")
        return self.heston_call_price(model, self.strike.item(), self.exercise_date.item())
        
    # ============================
    #  Heston characteristic function (λ = 0)
    # ============================

    def heston_cf(self, idx, u, T, S0, r, params):
        """
        Numerically-stable Heston characteristic function (modified form):
        - uses the "stable" branch choice equivalent to replacing d by -d
        - uses g_tilde and exp(-dT) in the log terms

        idx: 1 or 2 (Heston P1/P2)
        u: complex argument (can be real + 0j)
        """
        i = 1j
        kappa, theta, sigma, rho, v0 = params
        x0 = np.log(S0)

        a = kappa * theta

        # Heston conventions (λ = 0)
        if idx == 1:
            b = kappa - rho * sigma
            u_shift = 0.5
        elif idx == 2:
            b = kappa
            u_shift = -0.5
        else:
            raise ValueError("idx must be 1 or 2")

        # ---- d(u) with stable branch (book: choose negative root / avoid branch cut) ----
        # z = (rho*sigma*i*u - b)^2 + sigma^2*(u^2 - 2*i*u*u_shift)
        z = (rho * sigma * i * u - b) ** 2 + sigma ** 2 * (u ** 2 - 2.0 * i * u * u_shift)

        d = np.sqrt(z)

        # book-style: take the root that makes the implementation stable.
        # Equivalent to "replace d by -d" compared to the original Heston form:
        # enforce Re(d) <= 0 (so exp(dT) decays; same as using exp(-dT) with Re(d)>=0)
        if np.real(d) > 0:
            d = -d

        # ---- modified g (uses -d branch) ----
        # g_tilde = (b - rho*sigma*i*u - d) / (b - rho*sigma*i*u + d)
        g = (b - rho * sigma * i * u - d) / (b - rho * sigma * i * u + d)

        # ---- C and D in the stable algebra (uses exp(-dT)) ----
        exp_neg_dT = np.exp(-d * T)

        # Avoid forming 1 - g*exp(...) in an unstable way
        one_minus_g_exp = 1.0 - g * exp_neg_dT
        one_minus_g = 1.0 - g

        C = r * i * u * T + (a / sigma ** 2) * (
            (b - rho * sigma * i * u - d) * T
            - 2.0 * np.log(one_minus_g_exp / one_minus_g)
        )

        D = ((b - rho * sigma * i * u - d) / sigma ** 2) * (
            (1.0 - exp_neg_dT) / one_minus_g_exp
        )

        return np.exp(C + D * v0 + i * u * x0)


    # ============================
    #  Risk-neutral probabilities Q1 & Q2
    # ============================

    def _Qj(self, j, S0, K, T, r, params):
        """
        Q1 und Q2 nach Heston via Fourier-Integral.
        """
        i = 1j

        def integrand(j_idx):
            def f(u_real):
                u = u_real + 0j
                Phi = self.heston_cf(j_idx, u, T, S0, r, params)
                num = np.exp(-i * u * np.log(K)) * Phi
                den = i * u
                return np.real(num / den)
            return f

        if j == 1:
            integral, _ = quad(integrand(1), 0.0, 100.0, limit=200)
        elif j == 2:
            integral, _ = quad(integrand(2), 0.0, 100.0, limit=200)
        else:
            raise ValueError("j muss 1 oder 2 sein")

        return 0.5 + integral / np.pi


    def heston_call_price(self, model: HestonModel, K: float, T: float):
        """
        Heston-Call:
            C = S0 e^{-qT} Q1 - K e^{-rT} Q2
        """
        spot = model.get_spot().item()
        rate = model.get_rate().item()
        kappa = model.get_kappa().item()
        theta = model.get_theta().item()
        rho = model.get_rho().item()
        sigma = model.get_volatility().item()
        v0 = model.get_initial_variance().item()
        
        params = (kappa, theta, sigma, rho, v0)
        
        Q1 = self._Qj(1, spot, K, T, rate, params)
        Q2 = self._Qj(2, spot, K, T, rate, params)
        return spot * Q1 - K * np.exp(-rate * T) * Q2
        
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