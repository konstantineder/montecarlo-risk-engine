from models.model import *
from request_interface.request_interface import AtomicRequest, AtomicRequestType
from typing import List, Any

class VasicekModel(Model):
    """Vasicek 1-factor model for stochastic interest rates using Ornstein-Uhlenbeck process."""
    def __init__(
        self, 
        calibration_date     : float,  # Calibration date of the model
        rate                 : float,  # Short rate at calibration date
        mean                 : float,  # Long-term mean level the rate reverts to
        mean_reversion_speed : float,  # Speed at which the rate reverts to the long-term mean
        volatility           : float,  # Volatility of the short rate
        asset_id             : str | None = None,
    ):
        
        super().__init__(
            calibration_date=calibration_date, 
            state_dim=2,
            asset_ids=[asset_id]
        )
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(param, dtype=FLOAT, device=device)
            for param in list([rate]) + list([volatility]) + list([mean]) + list([mean_reversion_speed])
        ]

    # Retrieve specific model parameters
    def get_rate(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_mean(self):
        return torch.stack([self.model_params[2]]) 
    
    def get_mean_reversion_speed(self):
        return torch.stack([self.model_params[3]]) 
    
    def get_state(self, num_paths: int) -> torch.Tensor:
        """Return initial state for all paths with shape (num_paths, 2)."""
        r0 = self.get_rate().expand(num_paths)
        log_B0 = torch.zeros_like(r0, dtype=FLOAT, device=device)
        state = torch.stack([r0, log_B0], dim=-1)  # (N, 2)
        return state
    
    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix for time delta."""
        sigma = self.get_volatility()  
        a=self.get_mean_reversion_speed()
        exp_decay = torch.exp(-a * delta_t)
        variance = (sigma ** 2 / (2 * a)) * (1 - exp_decay ** 2)
        cov_matrix = torch.diag(variance)
        return cov_matrix
    
    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        Exact discretization for Vasicek:
            r_{t+Δ} = θ + (r_t - θ) e^{-aΔ} + sqrt( (σ^2 / (2a)) (1 - e^{-2aΔ}) ) * Z
            log_B accumulates ∫ r_s ds numerically (left Riemann) here: log_B_{t+Δ} ≈ log_B_t + r_t Δ
        """
        delta_t = time2 - time1
        r_t = state[:, 0:1]
        log_B_t = state[:, 1:2]
        a = self.get_mean_reversion_speed()
        theta = self.get_mean()

        # advance numeraire accumulator numerically (cheap and stable)
        log_B_t_next = log_B_t + r_t * delta_t

        exp_decay = torch.exp(-a * delta_t)
        mean = theta + (r_t - theta) * exp_decay
        r_next = mean + corr_randn

        return torch.stack([r_next.squeeze(-1), log_B_t_next.squeeze(-1)], dim=-1)
    
    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor    
    ) -> torch.Tensor:
        """
        Euler–Maruyama step:
            r_{t+Δ} = r_t + a(θ - r_t)Δ + σ√Δ Z
            log_B_{t+Δ} ≈ log_B_t + r_t Δ
        """
        delta_t = time2 - time1
        r_t = state[:, 0:1]
        log_B_t = state[:, 1:2]
        a = self.get_mean_reversion_speed()
        theta = self.get_mean()
        sigma = self.get_volatility()

        log_B_t += r_t * delta_t
        drift = a * (theta - r_t) * delta_t
        diffusion = sigma * torch.sqrt(delta_t) * corr_randn
        r_next = r_t + drift + diffusion

        return torch.stack([r_next.squeeze(-1), log_B_t.squeeze(-1)], dim=-1)   
    
    def compute_bond_price(self, time1, time2, rate):
        """Use analytic formula for Zerobond price in Vasicek model."""
        dt = time2 - time1
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        B = (1 - torch.exp(-a * dt)) / a

        term1 = (theta - (sigma**2 / (2 * a**2))) 

        # A(t,T) = exp(alpha(t,T))
        alpha_tt = term1 * (B - dt) - (sigma**2 / (4 * a)) * B**2
        A = torch.exp(alpha_tt)

        return A * torch.exp(-B * rate)
    
    def resolve_request(
        self, 
        req: AtomicRequest, 
        asset_id: str, 
        state: torch.Tensor
    ) -> torch.Tensor:
        """# Resolve requests posed by all products and at each exposure timepoint."""
        if req.request_type == AtomicRequestType.SPOT:
            return state[:,0] 
        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            time = req.time1
            rate=state[:,0]
            return self.compute_bond_price(self.calibration_date,time,rate)
        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            rate = state[:,0] 
            return self.compute_bond_price(time1,time2,rate)
        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            rate=state[:,0] 
            bond_price=self.compute_bond_price(time1,time2,rate)
            return (1/bond_price-1)/(time2-time1)
        elif req.request_type == AtomicRequestType.NUMERAIRE:
            log_B_t=state[:,1]
            return torch.exp(log_B_t) 
