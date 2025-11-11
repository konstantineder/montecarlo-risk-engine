from enum import Enum, auto
from common.packages import *
from models.model import Model
from typing import Union, List, Optional, Sequence, Dict, Set
from request_interface.request_interface import AtomicRequest, AtomicRequestType

class SimulationScheme(Enum):
    """Enum for simulation schemes."""
    EULER = 0
    MILSTEIN = 1
    ANALYTICAL = 2


class MonteCarloEngine:
    """Monte Carlo engine to simulate paths for pre and main sim."""
    def __init__(self, 
                 simulation_timeline    : torch.Tensor,
                 simulation_type        : SimulationScheme,
                 model                  : Model, 
                 num_paths              : int, 
                 num_steps              : int, 
                 is_pre_simulation      : bool = False
                 ):
        
        self.simulation_type = simulation_type
        self.model = model
        self.num_paths = num_paths
        self.num_steps = num_steps  
        self.simulation_timeline=simulation_timeline

        torch.manual_seed(42 if is_pre_simulation else 43)
    
    def generate_paths(self):
        if self.simulation_type==SimulationScheme.ANALYTICAL:
            return self._generate_paths_analytically(self.simulation_timeline,self.num_paths, self.num_steps)
        elif self.simulation_type==SimulationScheme.EULER:
            return self._generate_paths_euler(self.simulation_timeline,self.num_paths, self.num_steps)
        
    def _generate_paths_analytically(
        self, 
        timeline: torch.Tensor, 
        num_paths: int, 
        num_steps: int,
        ) -> torch.Tensor:
        """Simulate Monte Carlo paths using analytic formulae."""
        
        state = self.model.get_state(num_paths)
        paths: list = []

        t_prev = self.model.calibration_date

        for t_now in timeline:
            dt_total = t_now - t_prev
            dt = dt_total / num_steps
            corr_randn = self.model.generate_correlated_randn(num_paths, dt)

            for _ in range(num_steps):
                state = self.model.simulate_time_step_analytically(
                    delta_t=dt,
                    state=state,
                    corr_randn=corr_randn
                )
            
            paths.append(state)
            t_prev = t_now

        return torch.stack(paths, dim=1)

    def _generate_paths_euler(
        self, 
        timeline: torch.Tensor, 
        num_paths: int, 
        num_steps: int,
        ) -> torch.Tensor:
        """Simulate Monte Carlo paths using Euler-Mayurama scheme."""
        
        state = self.model.get_state(num_paths)
        paths: list = []

        t_prev = self.model.calibration_date

        for t_now in timeline:
            dt_total = t_now - t_prev
            dt = dt_total / num_steps
            corr_randn = self.model.generate_correlated_randn(num_paths, dt)

            for _ in range(num_steps):
                state = self.model.simulate_time_step_euler(
                    delta_t=dt,
                    state=state,
                    corr_randn=corr_randn
                )
            
            paths.append(state)
            t_prev = t_now

        return torch.stack(paths, dim=1)

    
