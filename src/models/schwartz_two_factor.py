from __future__ import annotations

from bisect import bisect_right

from models.model import *
from request_interface.request_interface import AtomicRequestType


class SchwartzTwoFactorModel(Model):
    """
    Two-factor commodity spot model around a prescribed baseline forward curve.

    log S(t) = log F0(t) + x(t) + y(t)

    x(t): short-term mean-reverting factor
    y(t): long-term Brownian factor
    """

    def __init__(
        self,
        calibration_date: float,
        curve_times: list[float],
        curve_values: list[float],
        rate: float,
        short_term_mean_reversion: float,
        short_term_vol: float,
        long_term_drift: float,
        long_term_vol: float,
        rho: float,
        asset_id: str | None = None,
    ):
        asset_ids = [asset_id] if asset_id else None
        super().__init__(
            calibration_date=calibration_date,
            asset_ids=asset_ids,
            simulation_dim=2,
            state_dim=3,
        )

        if len(curve_times) != len(curve_values):
            raise ValueError("curve_times and curve_values must have identical lengths.")
        if len(curve_times) < 2:
            raise ValueError("At least two curve points are required.")
        if any(value <= 0.0 for value in curve_values):
            raise ValueError("Curve values must be strictly positive.")

        self.curve_times = [float(t) for t in curve_times]
        self.curve_values = torch.tensor(curve_values, dtype=FLOAT, device=device)
        self.model_params = [
            torch.tensor(rate, dtype=FLOAT, device=device),
            torch.tensor(short_term_mean_reversion, dtype=FLOAT, device=device),
            torch.tensor(short_term_vol, dtype=FLOAT, device=device),
            torch.tensor(long_term_drift, dtype=FLOAT, device=device),
            torch.tensor(long_term_vol, dtype=FLOAT, device=device),
            torch.tensor(rho, dtype=FLOAT, device=device),
        ]

        one = torch.tensor(1.0, dtype=FLOAT, device=device)
        rho_tensor = self.get_rho()
        self.correlation_matrix = torch.stack(
            [
                torch.stack([one, rho_tensor]),
                torch.stack([rho_tensor, one]),
            ]
        )

    def get_rate(self):
        return self.model_params[0]

    def get_short_term_mean_reversion(self):
        return self.model_params[1]

    def get_short_term_vol(self):
        return self.model_params[2]

    def get_long_term_drift(self):
        return self.model_params[3]

    def get_long_term_vol(self):
        return self.model_params[4]

    def get_rho(self):
        return self.model_params[5]

    def get_model_param_names(self) -> list[str]:
        return [
            "rate",
            "short_term_mean_reversion",
            "short_term_vol",
            "long_term_drift",
            "long_term_vol",
            "rho",
        ]

    def _curve_value(self, time: float | torch.Tensor) -> torch.Tensor:
        time_value = float(time.item()) if torch.is_tensor(time) else float(time)

        if time_value <= self.curve_times[0]:
            return self.curve_values[0]
        if time_value >= self.curve_times[-1]:
            return self.curve_values[-1]

        upper_idx = bisect_right(self.curve_times, time_value)
        lower_idx = upper_idx - 1

        t0 = self.curve_times[lower_idx]
        t1 = self.curve_times[upper_idx]
        v0 = self.curve_values[lower_idx]
        v1 = self.curve_values[upper_idx]

        weight = (time_value - t0) / (t1 - t0)
        return v0 + (v1 - v0) * weight

    def get_state(self, num_paths: int):
        base_spot = self._curve_value(self.calibration_date)
        log_spot = torch.log(base_spot).expand(num_paths)
        short_factor = torch.zeros(num_paths, dtype=FLOAT, device=device)
        long_factor = torch.zeros(num_paths, dtype=FLOAT, device=device)
        return torch.stack([log_spot, short_factor, long_factor], dim=-1)

    def _get_correlation_matrix(self, simulation_scheme: SimulationScheme) -> torch.Tensor:
        return self.correlation_matrix

    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        kappa = self.get_short_term_mean_reversion()
        sigma_short = self.get_short_term_vol()
        sigma_long = self.get_long_term_vol()
        rho = self.get_rho()

        zero = torch.zeros_like(kappa)
        near_zero = torch.isclose(kappa, zero, atol=1e-12, rtol=0.0)
        var_short = torch.where(
            near_zero,
            sigma_short * sigma_short * delta_t,
            sigma_short * sigma_short * (1.0 - torch.exp(-2.0 * kappa * delta_t)) / (2.0 * kappa),
        )
        var_long = sigma_long * sigma_long * delta_t
        cov = rho * torch.sqrt(torch.clamp(var_short * var_long, min=0.0))

        return torch.stack(
            [
                torch.stack([var_short, cov]),
                torch.stack([cov, var_long]),
            ]
        )

    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor,
        state: torch.Tensor,
        corr_randn: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = time2 - time1
        short_factor = state[:, 1:2]
        long_factor = state[:, 2:3]

        kappa = self.get_short_term_mean_reversion()
        mu_long = self.get_long_term_drift()

        zero = torch.zeros_like(kappa)
        near_zero = torch.isclose(kappa, zero, atol=1e-12, rtol=0.0)
        exp_kdt = torch.exp(-kappa * delta_t)
        short_mean = torch.where(near_zero, short_factor, short_factor * exp_kdt)

        short_next = short_mean + corr_randn[:, 0:1]
        long_next = long_factor + mu_long * delta_t + corr_randn[:, 1:2]

        base_spot_next = self._curve_value(time2)
        log_spot_next = torch.log(base_spot_next) + short_next + long_next
        return torch.cat([log_spot_next, short_next, long_next], dim=1)

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor,
        state: torch.Tensor,
        corr_randn: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = time2 - time1
        sqrt_dt = torch.sqrt(delta_t)

        short_factor = state[:, 1:2]
        long_factor = state[:, 2:3]

        kappa = self.get_short_term_mean_reversion()
        sigma_short = self.get_short_term_vol()
        mu_long = self.get_long_term_drift()
        sigma_long = self.get_long_term_vol()

        short_next = short_factor - kappa * short_factor * delta_t + sigma_short * sqrt_dt * corr_randn[:, 0:1]
        long_next = long_factor + mu_long * delta_t + sigma_long * sqrt_dt * corr_randn[:, 1:2]

        base_spot_next = self._curve_value(time2)
        log_spot_next = torch.log(base_spot_next) + short_next + long_next
        return torch.cat([log_spot_next, short_next, long_next], dim=1)

    def resolve_request(self, req, asset_id, state):
        if req.request_type == AtomicRequestType.SPOT:
            return torch.exp(state[:, 0])
        if req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            t = req.time1
            return torch.exp(-self.get_rate() * (t - self.calibration_date))
        if req.request_type == AtomicRequestType.FORWARD_RATE:
            t1 = req.time1
            t2 = req.time2
            return torch.exp(self.get_rate() * (t2 - t1))
        if req.request_type == AtomicRequestType.LIBOR_RATE:
            t1 = req.time1
            t2 = req.time2
            return (torch.exp(self.get_rate() * (t2 - t1)) - 1) / (t2 - t1)
        if req.request_type == AtomicRequestType.NUMERAIRE:
            t = req.time1
            return torch.exp(self.get_rate() * (t - self.calibration_date))

        raise NotImplementedError(f"Request type {req.request_type} not supported.")
