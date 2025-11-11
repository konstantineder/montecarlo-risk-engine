from models.model import *
from request_interface.request_interface import AtomicRequestType

class BlackScholesModel(Model):
    """Black-Scholes model for a single asset."""
    
    def __init__(
        self, 
        calibration_date : float, # Date of model calibration
        asset_id         : str,
        spot             : float, # Spot price of the asset
        rate             : float, # Risk-free interest rate
        sigma            : float, # Volatility
    ):
        super().__init__(calibration_date=calibration_date, asset_ids=[asset_id])
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
        spot = self.get_spot()
        return torch.log(spot).expand(num_paths).clone()
    
    def compute_cov_matrix(self, delta_t):
        """Compute covariance matrix for time delta."""
        sigma = self.get_volatility()                # (A,)
        cov_matrix = torch.diag(sigma * sigma * delta_t)
        return cov_matrix
    
    def generate_correlated_randn(self, num_paths: int, delta_t: float) -> torch.Tensor:
        sigma = self.get_volatility()
        z = torch.randn(num_paths, dtype=FLOAT, device=device)
        return sigma * torch.sqrt(delta_t) * z

    def simulate_time_step_analytically(
        self, 
        delta_t: float, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor
    ) -> torch.Tensor:
        """
        state:  (markov_dim,)
        randn:  same shape as state (noise already correlated)
        """
        rate = self.get_rate()
        sigma = self.get_volatility()
        
        drift = rate * delta_t
        diffusion = corr_randn - 0.5 * delta_t * sigma**2
        return state + drift + diffusion

    def simulate_time_step_euler(self, delta_t: float, state: torch.Tensor, num_paths: int) -> torch.Tensor:
        """
        Euler–Maruyama step for BS multi asset model.
        """
        rate = self.get_rate()
        sigma = self.get_volatility()
        spot = torch.exp(state)
        
        z = torch.randn(num_paths, device=device)
        dS = rate * spot * delta_t + sigma * spot * torch.sqrt(delta_t) * z
        spot = spot + dS
        return torch.log(spot)
    
    def resolve_request(self, req, asset_id, state):
        """Resolve requests posed by all products and at each exposure timepoint."""

        if req.request_type == AtomicRequestType.SPOT:
            return torch.exp(state)
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