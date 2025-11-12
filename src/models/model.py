from common.packages import *
from common.enums import SimulationScheme
from request_interface.request_types import AtomicRequest

class Model:
    """
    Base model class
    """
    def __init__(
        self, 
        calibration_date : float, # Calibration date of the model
        simulation_dim   : int = 1,
        state_dim      : int = 1,
        asset_ids        : list[str] | None = None,
    ):
        
        self.calibration_date = torch.tensor([calibration_date], dtype=FLOAT,device=device)
        self.asset_ids: list[str] = asset_ids if asset_ids else [""]
        self.model_params: list[torch.Tensor] = []
        self.num_assets = len(self.asset_ids) 
        self.simulation_dim = simulation_dim
        self.state_dim = state_dim
        
        self._cholesky: dict[tuple[str, float | None], torch.Tensor] = {}
    
    def get_model_params(self):
        return self.model_params
    
    def set_model_params(self, params: torch.Tensor):
        self._params = params 
        
    def generate_correlated_randn(
        self, 
        num_paths: int, 
        simulation_scheme: SimulationScheme, 
        delta_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        dt = float(delta_t) if delta_t is not None else None
        chol = self.get_cholesky(simulation_scheme=simulation_scheme, delta_t=dt)
        if self.simulation_dim == 1:
            z = torch.randn(num_paths, dtype=FLOAT, device=device)
            return z * chol[0]
        else:
            z = torch.randn(num_paths, self.simulation_dim, dtype=FLOAT, device=device)
            return z @ chol.T
        
    def get_cholesky(self, simulation_scheme: SimulationScheme, delta_t: torch.Tensor | None) -> torch.Tensor:
        """Compute Cholesky decomposition of covariance matrix.
        
        if Cholesky matrix has not yet been computed for current time delta
        otherwise retrieve stored matrix
        """
        if simulation_scheme == SimulationScheme.ANALYTICAL:
            key = (simulation_scheme, float(delta_t))
            if key not in self._cholesky:
                cov = self._get_covariance_matrix(delta_t)
                chol = torch.linalg.cholesky(cov)
                self._cholesky[key] = chol
                return chol
            else:
                return self._cholesky[key]
        else:
            key = (simulation_scheme, None)
            if key not in self._cholesky:
                corr = self._get_correlation_matrix()
                chol = torch.linalg.cholesky(corr)
                self._cholesky[key] = chol
                return chol
            else:
                return self._cholesky[key]
                
    def _get_correlation_matrix(self) -> torch.Tensor:
        """Compute correlation matrix"""
        return torch.eye(self.simulation_dim, dtype=FLOAT, device=device)
    
    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix"""
        return torch.eye(self.simulation_dim, dtype=FLOAT, device=device) * delta_t       
    
    def requires_grad(self):
        """
        If differentiation is enabled, put all model parameters on tape
        and accumulate adjoints during simulation via AAD        
        """
        for param in self.model_params:
            param.requires_grad_(True)

    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        Analytic simulation step.
        """
        return NotImplementedError(f"Method not implemented")

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        Euler–Maruyama time step.
        """
        return NotImplementedError(f"Method not implemented")
    
    def simulate_time_step_milstein(self, state: torch.Tensor, randn: torch.Tensor) -> torch.Tensor:
        """
        Milstein time step.
        """
        return NotImplementedError(f"Method not implemented")
    
    def resolve_request(
        self, 
        req: AtomicRequest, 
        asset_id: str, 
        state: torch.Tensor    
    ):
        return NotImplementedError(f"Method not implemented")



    




    
