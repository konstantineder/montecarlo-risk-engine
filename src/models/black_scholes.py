from models.model import *
from request_interface.request_interface import AtomicRequestType

class BlackScholesModel(Model):
    """Black-Scholes model for a single asset."""
    
    def __init__(
        self, 
        calibration_date : float, # Date of model calibration
        spot             : float, # Spot price of the asset
        rate             : float, # Risk-free interest rate
        sigma            : float, # Volatility
        asset_id         : str | None = None,
    ):
        asset_ids = [asset_id] if asset_id else None
        super().__init__(
            calibration_date=calibration_date, 
            asset_ids=asset_ids
        )
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(spot, dtype=FLOAT, device=device),
            torch.tensor(sigma, dtype=FLOAT, device=device),
            torch.tensor(rate, dtype=FLOAT, device=device),
        ]

    # Retrieve specific model parameters
    def get_spot(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_rate(self):
        return torch.stack([self.model_params[2]])
    
    def get_state(self, num_paths: int):
        return self.get_spot().expand(num_paths).unsqueeze(-1).clone()
    
    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix for time delta."""
        sigma = self.get_volatility()                
        cov_matrix = torch.diag(sigma * sigma * delta_t)
        return cov_matrix

    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        state:  (markov_dim,)
        randn:  same shape as state (noise already correlated)
        """
        delta_t = time2 - time1
        rate = self.get_rate().squeeze(-1)
        sigma = self.get_volatility().squeeze(-1)
        
        drift = rate * delta_t
        diffusion = corr_randn - 0.5 * delta_t * sigma**2
        return state * torch.exp(drift + diffusion)

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        Euler–Maruyama step for BS multi asset model.
        """
        delta_t = time2 -time1
        rate = self.get_rate().squeeze(-1)
        sigma = self.get_volatility().squeeze(-1)
        
        dS = rate * state * delta_t + sigma * state * torch.sqrt(delta_t) * corr_randn
        state = state + dS
        return state
    
    def resolve_request(self, req, asset_id, state):
        """Resolve requests posed by all products and at each exposure timepoint."""

        if req.request_type == AtomicRequestType.SPOT:
            return state.squeeze(-1)
        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(-rate * (t - self.calibration_date))
        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            t1 = req.time1
            t2 = req.time2
            rate=self.get_rate()
            return torch.exp(rate * (t2-t1))
        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            t1 = req.time1
            t2 = req.time2
            rate=self.get_rate()
            return (torch.exp(rate * (t2-t1))-1)/(t2-t1)
        elif req.request_type == AtomicRequestType.NUMERAIRE:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(rate * (t - self.calibration_date))
        else:
            raise NotImplementedError(f"Request type {req.request_type} not supported.")