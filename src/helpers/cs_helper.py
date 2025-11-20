import numpy as np
from typing import List, Tuple
from maths.maths import bisection_search
import torch


class CSHelper:

    def _compute_cds_legs(
        self,
        maturities: List[float],                     # T_i (years), ascending
        payment_days: np.ndarray,                   # coupon dates t_k (years), ascending
        discount_factors_payment_days: np.ndarray,  
        recovery_rate: float,
        hazard_rates: List[float]
    ) -> Tuple[float, float]:
        indices = np.searchsorted(payment_days, maturities)
        time_to_index = dict(zip(maturities, indices))

        deltas = [payment_days[0]] + [payment_days[k]-payment_days[k-1]
                                    for k in range(1, len(payment_days))]
        premium_leg = 0.0
        protection_leg  = 0.0
        survival_prob_prev = 1.0
        prev_time_idx = 0
        for idx, maturity in enumerate(maturities):
            prev_maturity = maturities[idx - 1] if idx > 0 else 0.0
            time_idx = time_to_index[maturity]
            hazard_rate = hazard_rates[idx]
            for k in range(prev_time_idx, time_idx + 1):  # include coupon at t_k = T_i
                payment_date = payment_days[k]
                discount = discount_factors_payment_days[k]
                delta = deltas[k]
                survival_prob = survival_prob_prev * np.exp(-hazard_rate * (payment_date - prev_maturity))
                # premium leg: coupons + AoD (trapezoid)
                accrual = 0.5 * delta * discount * (survival_prob_prev - survival_prob)
                premium_leg += delta * discount * survival_prob + accrual
                # protection leg
                protection_leg += (1.0 - recovery_rate) * discount * (survival_prob_prev - survival_prob)
            prev_time_idx = time_idx
            survival_prob_prev = survival_prob
        return premium_leg, protection_leg

    def bootstrap_hazards(
        self,
        credit_spreads: List[float],                 
        maturities: np.ndarray,                     
        payment_days: np.ndarray,                   
        discount_factors_payment_days: np.ndarray,  
        recovery_rate: float                         
    ) -> List[float]:
        """
        Generalized CDS bootstrap (coupon grid + AoD trapezoid) with piecewise-constant hazards.
        Calls `compute_legs` during root-finding.
        Assumes each maturity T_i is present in `payment_days`.
        """
        assert len(payment_days) == len(discount_factors_payment_days)
        hazard_rates: List[float] = []

        for i, spread in enumerate(credit_spreads):
            # Build F_i(lambda_i) = spread * RPV01(T_i) - Prot(T_i),
            # where RPV01/Prot are computed by compute_legs on data up to T_i
            def F(lam_i: float) -> float:
                tmp_haz = hazard_rates + [lam_i]           # previous solved + candidate
                mats_up_to_i = maturities[: i + 1]         # only up to current maturity
                prem, prot = self._compute_cds_legs(
                    mats_up_to_i,
                    payment_days,
                    discount_factors_payment_days,
                    recovery_rate,
                    tmp_haz
                )
                return spread * prem - prot

            lam_i = bisection_search(F)  # robust and keeps your flow
            hazard_rates.append(lam_i)

        return hazard_rates
    
    def probability_of_default(
        self,
        hazards: torch.Tensor, 
        tenors: torch.Tensor, 
        date: torch.Tensor
    ) -> torch.Tensor:
        """
        hazards[i] applies on (tenors[i-1], tenors[i]] with tenors[-1] >= date.
        Assumes hazards are piecewise-constant across the tenor grid and flat-extended on the last bucket if date > tenors[-1].
        """
        survival_prob, prev = 1.0, 0.0

        for idx, mat in enumerate(tenors):
            if mat <= date:
                dt = mat - prev
                survival_prob *= torch.exp(-hazards[idx] * dt)
                prev = mat
            else:
                break
        else:
            # loop didn't break: date is beyond the last tenor; extend last hazard flat
            idx = len(tenors) - 1

        # final stub within the current bucket up to 'date'
        dt = date - prev
        if dt > 0:
            survival_prob *= torch.exp(-hazards[idx] * dt)

        return 1.0 - survival_prob  # this is the cumulative PD up to 'date'