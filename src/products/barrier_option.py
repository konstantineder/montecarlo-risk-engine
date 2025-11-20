from products.product import *
from maths.maths import compute_degree_of_truth
from request_interface.request_interface import AtomicRequestType, AtomicRequest
import numpy as np
from collections import defaultdict
from typing import Union, List, Optional

class BarrierOptionType(Enum):
    DOWNANDOUT = "Down-And-Out"
    UPANDOUT = "Up-And-Out"
    DOWNANDIN = "Down-And-In"
    UPANDIN = "Up-And-In"     

# Barrier Option implementation
# Supports both single barrier and double barrier options
class BarrierOption(Product):
    def __init__(
        self, 
        startdate                  : float, # Startdate of the contract           
        maturity                   : float, # Maturity of the option
        strike                     : float, # Strike price
        num_observation_timepoints : int,   # Number of observation timepoints used to check if barrier has been crossed
        option_type                : OptionType, # Set up type of option
        barrier1                   : float,      # first barrier
        barrier_option_type1       : Optional[BarrierOptionType], # Type of first barrier
        barrier2                   : Optional[float] = None,      # Second barrier (none if single barrier option)
        barrier_option_type2       : Optional[BarrierOptionType] = None, # Type of second barrier
        asset_id                   : str | None = None,
    ): 
        
        super().__init__(asset_ids=[asset_id])
        self.strike = torch.tensor([strike], dtype=FLOAT,device=device)
        self.maturity=torch.tensor([maturity], dtype=FLOAT,device=device)
        self.product_timeline=torch.tensor([maturity], dtype=FLOAT,device=device)
        self.modeling_timeline=torch.linspace(startdate, maturity,num_observation_timepoints, dtype=FLOAT,device=device)
        self.regression_timeline=torch.tensor([], dtype=FLOAT,device=device)
        self.barrier1 = torch.tensor([barrier1], dtype=FLOAT,device=device)
        self.barrier_option_type1 = barrier_option_type1
        
        self.barrier2 = barrier2
        if barrier2 is not None:
            self.barrier2 = torch.tensor([barrier2], dtype=FLOAT,device=device) 
        self.barrier_option_type2 = barrier_option_type2

        self.option_type = option_type
        self.use_brownian_bridge = False
        self.use_seed = 12345
        self.rng = np.random.default_rng(12345)

        self.numeraire_requests={
            idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) 
            for idx, t in enumerate(self.modeling_timeline)
            }
        asset_id = self.get_asset_id()
        self.spot_requests={
            (idx, asset_id): AtomicRequest(AtomicRequestType.SPOT) 
            for idx in range(len(self.modeling_timeline))
            }
    
    def set_use_brownian_bridge(self):
        self.use_brownian_bridge = True

    def compute_payoff_without_brownian_bridge(self, paths, model):
        spots_at_maturity = paths[:,-1]
        max_spot = torch.max(paths, dim=1).values
        min_spot = torch.min(paths, dim=1).values

        is_max_below_barrier = compute_degree_of_truth(self.barrier1-max_spot,True)
        is_min_above_barrier = compute_degree_of_truth(min_spot - self.barrier1,True)

        zero=torch.tensor([0.0], device=device)

        payoff = torch.zeros_like(spots_at_maturity)

        if self.barrier_option_type1 == BarrierOptionType.UPANDOUT:
            payoff = torch.maximum(spots_at_maturity - self.strike, zero) * is_max_below_barrier if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * is_max_below_barrier
        if self.barrier_option_type1 == BarrierOptionType.DOWNANDOUT:
            payoff = torch.maximum(spots_at_maturity - self.strike, zero) * is_min_above_barrier if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * is_min_above_barrier
        if self.barrier_option_type1 == BarrierOptionType.UPANDIN:
            payoff = torch.maximum(spots_at_maturity - self.strike, zero) * (1 - is_max_below_barrier) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * (1 - is_max_below_barrier)
        if self.barrier_option_type1 == BarrierOptionType.DOWNANDIN:
            payoff = torch.maximum(spots_at_maturity - self.strike, zero) * (1 - is_min_above_barrier) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * (1 - is_min_above_barrier)
        
        if self.barrier2 is not None and self.barrier_option_type2 is not None:

            is_max_below_barrier2 = compute_degree_of_truth(self.barrier2 - max_spot, True)
            is_min_above_barrier2 = compute_degree_of_truth(min_spot - self.barrier2, True)

            if self.barrier_option_type2 == BarrierOptionType.UPANDOUT:
                payoff *= is_max_below_barrier2
            elif self.barrier_option_type2 == BarrierOptionType.DOWNANDOUT:
                payoff *= is_min_above_barrier2
            elif self.barrier_option_type2 == BarrierOptionType.UPANDIN:
                payoff *= (1 - is_max_below_barrier2)
            elif self.barrier_option_type2 == BarrierOptionType.DOWNANDIN:
                payoff *= (1 - is_min_above_barrier2)

        return payoff
        
    def compute_payoff_with_brownian_bridge(self, spots, model):
        sigma = model.get_volatility()
        spots_at_maturity = spots[:,-1]
        max_spot = torch.max(spots, dim=1).values  # Max spot across each path
        min_spot = torch.min(spots, dim=1).values
        num_spots = spots.shape[1]

        # Precompute Brownian bridge crossing probabilities
        log_spot_barrier = torch.log(spots / self.barrier1)
        log_spot_barrier_next = torch.log(spots[:, 1:] / self.barrier1)
        bridge_probs = torch.exp(-2 * log_spot_barrier[:, :-1] * log_spot_barrier_next / (sigma **2 * (self.maturity / num_spots)))

        is_max_below_barrier = compute_degree_of_truth(self.barrier1-max_spot,True)
        is_min_above_barrier = compute_degree_of_truth(min_spot - self.barrier1,True)

        zero=torch.tensor([0.0], device=device)

        vanilla_payoff = torch.maximum(spots_at_maturity - self.strike, zero) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero)

        # One draw per interval (barrier 1)
        rdns = self.rng.uniform(0, 1, size=bridge_probs.shape)
        rdns = torch.tensor(rdns, dtype=bridge_probs.dtype, device=bridge_probs.device)
        hit_probs1 = compute_degree_of_truth(bridge_probs - rdns,True)
        hit_barrier1 = 1 - torch.prod(1 - hit_probs1, dim=1)

        if self.barrier_option_type1 == BarrierOptionType.UPANDOUT:
            payoff = vanilla_payoff * is_max_below_barrier * (1-hit_barrier1)

        elif self.barrier_option_type1 == BarrierOptionType.DOWNANDOUT:
            payoff = vanilla_payoff * is_min_above_barrier * (1-hit_barrier1)

        elif self.barrier_option_type1 == BarrierOptionType.UPANDIN:
            payoff = vanilla_payoff * (1-is_max_below_barrier)* hit_barrier1

        elif self.barrier_option_type1 == BarrierOptionType.DOWNANDIN:
            payoff = vanilla_payoff * (1-is_min_above_barrier)* hit_barrier1

        else:
            raise NotImplementedError(f"Barrier type {self.barrier_option_type1} not supported.")
        
        if self.barrier2 is not None and self.barrier_option_type2 is not None:
            log_spot_barrier2 = torch.log(spots / self.barrier2)
            log_spot_barrier2_next = torch.log(spots[:, 1:] / self.barrier2)
            bridge_probs2 = torch.exp(-2 * log_spot_barrier2[:, :-1] * log_spot_barrier2_next / (sigma**2 * (self.maturity / num_spots)))

            # One draw per interval (barrier 2)
            rdns2 = self.rng.uniform(0, 1, size=bridge_probs2.shape)
            rdns2 = torch.tensor(rdns2, dtype=bridge_probs2.dtype, device=bridge_probs2.device)
            hit_probs2 = compute_degree_of_truth(bridge_probs2 - rdns2, True)
            hit_barrier2 = 1 - torch.prod(1 - hit_probs2, dim=1)

            max_spot = torch.max(spots, dim=1).values
            min_spot = torch.min(spots, dim=1).values

            is_max_below_barrier2 = compute_degree_of_truth(self.barrier2 - max_spot, True)
            is_min_above_barrier2 = compute_degree_of_truth(min_spot - self.barrier2, True)

            if self.barrier_option_type2 == BarrierOptionType.UPANDOUT:
                payoff *= is_max_below_barrier2 * (1 - hit_barrier2)
            elif self.barrier_option_type2 == BarrierOptionType.DOWNANDOUT:
                payoff *= is_min_above_barrier2 * (1 - hit_barrier2)
            elif self.barrier_option_type2 == BarrierOptionType.UPANDIN:
                payoff *= (1 - is_max_below_barrier2) * hit_barrier2
            elif self.barrier_option_type2 == BarrierOptionType.DOWNANDIN:
                payoff *= (1 - is_min_above_barrier2) * hit_barrier2
            else:
                raise NotImplementedError(f"Barrier type {self.barrier_option_type2} not supported.")

        return payoff

            
    def payoff(self, spots, model):
            if self.use_brownian_bridge==True:
                return self.compute_payoff_with_brownian_bridge(spots, model)
            else:
                return self.compute_payoff_without_brownian_bridge(spots, model)


    def compute_pv_analytically(self, model):
            S=model.get_spot()
            rate=model.get_rate()
            sigma=model.get_volatility()

            B=self.barrier1
            K=self.strike
            T=self.maturity
            sqrtT = torch.sqrt(T)

            norm = torch.distributions.Normal(0.0, 1.0)

            if self.barrier_option_type1==BarrierOptionType.UPANDOUT:
                if self.option_type == OptionType.CALL:

                    d1_spot_strike = (torch.log(S/ K) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d1_spot_barrier = (torch.log(S / B) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d1_barrier_strike = (torch.log(B**2 / (K*S)) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d1_barrier_spot = (torch.log(B / S) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d2_spot_strike = d1_spot_strike - sigma * sqrtT
                    d2_spot_barrier = d1_spot_barrier - sigma * sqrtT
                    d2_barrier_strike = d1_barrier_strike - sigma * sqrtT
                    d2_barrier_spot = d1_barrier_spot - sigma * sqrtT

                    is_spot_below_barrier=S < B

                    term1=norm.cdf(d1_spot_strike)-norm.cdf(d1_spot_barrier)
                    term2=norm.cdf(d1_barrier_strike)-norm.cdf(d1_barrier_spot)
                    term3=norm.cdf(d2_spot_strike)-norm.cdf(d2_spot_barrier)
                    term4=norm.cdf(d2_barrier_strike)-norm.cdf(d2_barrier_spot)

                    term_spot=S * (term1-(B/S)**(1+2*rate/(sigma**2))*term2)
                    term_strike=K * torch.exp(-rate * T) * (term3-(S/B)**(1-2*rate/(sigma**2))*term4)

                    return is_spot_below_barrier*(term_spot-term_strike)
                    
                else:
                    return NotImplementedError(f"Analytical method for up-and-out {self.option_type} not yet implemented")
            
            elif self.barrier_option_type1 == BarrierOptionType.DOWNANDOUT:
                if self.option_type == OptionType.CALL:
                    
                    d1 = (torch.log(S/ K) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d2 = d1 - sigma * sqrtT
                    d1_barrier_strike = (torch.log(B**2 / (K*S)) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrtT)
                    d2_barrier_strike = d1_barrier_strike - sigma * sqrtT

                    factor = (B/S)**(2*rate/sigma**2)

                    term1 = S * norm.cdf(d1) - K * torch.exp(-rate * T) * norm.cdf(d2)
                    term2 = (B/S) * norm.cdf(d1_barrier_strike) - (K/S) * torch.exp(-rate * T) * norm.cdf(d2_barrier_strike)

                    is_above_barrier = S > B
                    return is_above_barrier * (term1 - S * factor * term2)
            
            else:
                return NotImplementedError(f"Analytical method for {self.barrier_option_type1} not yet implemented")

    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_RegressionFunction=None, state=None):
        spots: list[torch.Tensor] = []
        for idx in range(len(self.modeling_timeline)):
            spots.append(
                self.get_resolved_atomic_request(
                    resolved_atomic_requests=resolved_requests[0],
                    request_type=AtomicRequestType.SPOT,
                    time_idx=idx,
                    asset_id=self.get_asset_id(),
                )
            )
        spots=torch.stack(spots, dim=1)
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[0][self.numeraire_requests[len(self.product_timeline)-1].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs.unsqueeze(1)