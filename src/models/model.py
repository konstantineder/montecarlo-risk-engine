from common.packages import *

class Model:
    """
    Base model class
    """
    def __init__(self, 
                 calibration_date : float, # Calibration date of the model
                 asset_ids        : list[str]
                 ):
        
        self.calibration_date = torch.tensor([calibration_date], dtype=FLOAT,device=device)
        self.asset_ids = asset_ids
        self.model_params: list[torch.Tensor] = []
        self.num_assets=1
        self.markov_dim = 1
    
    def get_model_params(self):
        return self.model_params
    
    def set_model_params(self, params: torch.Tensor):
        self._params = params        

    
    def requires_grad(self):
        """
        If differentiation is enabled, put all model parameters on tape
        and accumulate adjoints during simulation via AAD        
        """
        for param in self.model_params:
            param.requires_grad_(True)

    def simulate_time_step_analytically(self, delta_t: float, state: torch.Tensor, num_paths: int) -> torch.Tensor:
        """
        Analytic simulation step.
        """
        return NotImplementedError(f"Method not implemented")
        

    def simulate_time_step_euler(self, state: torch.Tensor, randn: torch.Tensor) -> torch.Tensor:
        """
        Euler–Maruyama time step.
        """
        return NotImplementedError(f"Method not implemented")
    
    def simulate_time_step_milstein(self, state: torch.Tensor, randn: torch.Tensor) -> torch.Tensor:
        """
        Milstein time step.
        """
        return NotImplementedError(f"Method not implemented")
    
    def resolve_request(self, state, requests):
        return NotImplementedError(f"Method not implemented")



    




    
