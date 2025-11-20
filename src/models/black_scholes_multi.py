from models.model import *
from request_interface.request_interface import AtomicRequest, AtomicRequestType
from typing import List, Any, Dict, Set
from numpy.typing import NDArray

class BlackScholesMulti(Model):
    """Black-Scholes Model for multiple assets."""
    
    def __init__(
        self, 
        calibration_date   : float,        # Calibration date of the model
        rate               : float,        # Risk-free interest rate
        asset_ids          : List[str],
        spots              : List[float],  # Spot prices of the assets
        volatilities       : List[float],  # Volatilities of the assets
        correlation_matrix : NDArray[Any]  # Correlations amonng assets
    ):
        
        super().__init__(
            calibration_date=calibration_date,
            simulation_dim=len(asset_ids),
            state_dim=len(spots),
            asset_ids=asset_ids,
        )
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(param, dtype=FLOAT, device=device)
            for param in list(spots) + list(volatilities) + list([rate])
        ]

        self.correlation_matrix = torch.tensor(correlation_matrix, dtype=FLOAT, device=device)
    
    # Retrieve model parameters
    def get_spot(self):
        return torch.stack(self.model_params[:self.num_assets])

    def get_volatility(self):
        return torch.stack(self.model_params[self.num_assets:2*self.num_assets])

    def get_rate(self):
        return self.model_params[2*self.num_assets]
    
    def get_state(self, num_paths: int):
        return self.get_spot().expand(num_paths, self.num_assets).clone()
    
    def _get_correlation_matrix(self) -> torch.Tensor:
        """Compute covrrelation_matrix."""
        return self.correlation_matrix

    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix for time delta."""
        S = torch.diag(self.get_volatility())
        cov_matrix = S @ self.correlation_matrix @ S
        cov_matrix = cov_matrix*delta_t
        return cov_matrix
        
    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor      
    ) -> torch.Tensor:
        """
        state:  (N, markov_dim) or (markov_dim,)
        randn:  same shape as state (noise already correlated)
        """
        delta_t = time2 - time1
        rate = self.get_rate()
        
        sigma = self.get_volatility().reshape(1, -1) 
        drift = (rate - 0.5 * sigma * sigma) * delta_t
        return state * torch.exp(drift + corr_randn)

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
        pass

    def resolve_request(self, req: AtomicRequest, asset_id: str, state: torch.Tensor) -> torch.Tensor:
        """Resolve requests posed by each product and at each exposure timepoint."""
        if req.request_type == AtomicRequestType.SPOT:
            asset_idx = self.asset_ids.index(asset_id)
            state_asset = state[:,asset_idx]
            spot = state_asset
            return spot

        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(-rate * (t - self.calibration_date))

        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            t1, t2 = req.time1, req.time2
            rate=self.get_rate()
            return torch.exp(rate * (t2 - t1))

        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            t1, t2 = req.time1, req.time2
            rate=self.get_rate()
            return (torch.exp(rate* (t2 - t1)) - 1) / (t2 - t1)

        elif req.request_type == AtomicRequestType.NUMERAIRE:
            t = req.time1
            rate=self.get_rate()
            numeraire = torch.exp(rate * (t - self.calibration_date))
            return numeraire

        else:
            raise NotImplementedError(f"Request type {req.request_type} not supported.")
