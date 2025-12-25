from models.model import *
from maths.maths import compute_degree_of_truth
from request_interface.request_interface import AtomicRequestType

import torch
import torch.nn.functional as F

def smooth_pos(x, eps=1e-12, beta=50.0):
    # strictly positive, smooth; approx max(x, eps)
    return eps + F.softplus(x - eps, beta=beta)

def smooth_abs(x, eps=1e-12):
    # smooth |x|
    return torch.sqrt(x*x + eps)

def smooth_cap(x, cap=1e3, beta=50.0):
    # smooth min(x, cap)
    return cap - F.softplus(cap - x, beta=beta)

class HestonModel(Model):
    
    def __init__(
        self, 
        calibration_date : float, # Date of model calibration
        spot             : float, # Spot price of the asset
        rate             : float, # Risk-free interest rate
        sigma            : float, # Volatility
        rho              : float, # Correlation between asset and variance
        kappa            : float, # Speed of mean reversion
        theta            : float, # Long-term variance
        v0               : float, # Initial variance
        asset_id         : str | None = None,
    ):
        asset_ids = [asset_id] if asset_id else None
        super().__init__(
            calibration_date=calibration_date, 
            asset_ids=asset_ids,
            simulation_dim=2,
            state_dim=2,
        )
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(spot, dtype=FLOAT, device=device),
            torch.tensor(sigma, dtype=FLOAT, device=device),
            torch.tensor(rate, dtype=FLOAT, device=device),
            torch.tensor(rho, dtype=FLOAT, device=device),
            torch.tensor(kappa, dtype=FLOAT, device=device),
            torch.tensor(theta, dtype=FLOAT, device=device),
            torch.tensor(v0, dtype=FLOAT, device=device),
        ]
        
        rho = self.get_rho().squeeze()  # scalar tensor
        one = torch.tensor(1.0, dtype=FLOAT, device=device)
        self.correlation_matrix = torch.stack([
            torch.stack([one, rho]),
            torch.stack([rho, one]),
        ])

    # Retrieve specific model parameters
    def get_spot(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_rate(self):
        return torch.stack([self.model_params[2]])
    
    def get_rho(self):
        return torch.stack([self.model_params[3]])
    
    def get_kappa(self):   
        return torch.stack([self.model_params[4]])
    
    def get_theta(self):
        return torch.stack([self.model_params[5]])
    
    def get_initial_variance(self):
        return torch.stack([self.model_params[6]])
    
    def _get_correlation_matrix(self, simulation_scheme: SimulationScheme) -> torch.Tensor:
        """Compute covrrelation_matrix."""
        if simulation_scheme == SimulationScheme.QE:
            return torch.eye(self.simulation_dim, dtype=FLOAT, device=device)
        else:
            return self.correlation_matrix
    
    def get_state(self, num_paths: int):
        spot = self.get_spot()
        log_spot = torch.log(spot).expand(num_paths)
        variance = self.get_initial_variance().expand(num_paths)
        state = torch.stack([log_spot, variance], dim=-1) 
        return state

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor  
    ) -> torch.Tensor:
        """
        Euler–Maruyama step for Heston Model.
        """
        delta_t = time2 -time1
        log_spot = state[:, 0:1]
        variance = state[:, 1:2]
        rate = self.get_rate()
        sigma = self.get_volatility()
        kappa = self.get_kappa()
        theta = self.get_theta()
        
        log_spot_next = log_spot + (rate - 0.5 * variance) * delta_t + torch.sqrt(torch.clamp(variance, min=0.0)) * torch.sqrt(delta_t) * corr_randn[:, 0:1]
        variance_next = variance + kappa * (theta - variance) * delta_t + sigma * torch.sqrt(torch.clamp(variance, min=0.0)) * torch.sqrt(delta_t) * corr_randn[:, 1:2]
        variance_next = torch.clamp(variance_next, min=0.0)  # ensure variance stays non-negative
        state = torch.stack([log_spot_next.squeeze(-1), variance_next.squeeze(-1)], dim=-1)
        return state
    
    def get_cond_variance_variance(self, state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Extract variance from state tensor."""
        variance = state[:, 1:2]
        sigma = self.get_volatility()
        kappa = self.get_kappa()
        theta = self.get_theta()
        
        term1 = variance * sigma ** 2 * torch.exp(-kappa * delta_t) * (1 - torch.exp(-kappa * delta_t)) / kappa
        term2 = theta * sigma ** 2 * (1 - torch.exp(-kappa * delta_t)) ** 2 / (2 * kappa)
        cond_variance = term1 + term2
        return cond_variance
    
    def get_cond_vaiance_mean(self, state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Extract log-spot from state tensor."""
        variance = state[:, 1:2]
        kappa = self.get_kappa()
        theta = self.get_theta()
        
        exp_term = torch.exp(-kappa * delta_t)
        cond_variance_mean = theta + (variance - theta) * exp_term
        return cond_variance_mean
    
    def get_Ks(self, delta_t):
        sigma = self.get_volatility()
        kappa = self.get_kappa()
        theta = self.get_theta()
        rho   = self.get_rho()

        gamma1 = 1.0
        gamma2 = 0.0

        K0 = -rho * kappa * theta / sigma * delta_t
        K1 = (kappa * rho / sigma - 0.5) * gamma1 * delta_t - rho / sigma
        K2 = (kappa * rho / sigma - 0.5) * gamma2 * delta_t + rho / sigma
        K3 = (1.0 - rho * rho) * gamma1 * delta_t
        K4 = (1.0 - rho * rho) * gamma2 * delta_t
        return K0, K1, K2, K3, K4
    
    def simulate_time_step_qe(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor,
        state: torch.Tensor,
        corr_randn: torch.Tensor
    ) -> torch.Tensor:
        """
        Andersen QE variance update + Andersen log-spot update (eq. 33),
        implemented in a numerically safe + AAD-friendly way.

        - Smooth switching around psi_c
        - Smooth mass-at-zero decision for case2 (AAD friendly)
        """
        eps = 1e-12

        dt = time2 - time1
        logS = state[:, 0:1]
        v    = state[:, 1:2]
        #v    = torch.clamp(v, min=0.0)  # keep state physically valid (piecewise grad ok)

        rate  = self.get_rate()

        # --- Conditional moments for CIR variance ---
        m  = self.get_cond_vaiance_mean(state, dt)
        s2 = self.get_cond_variance_variance(state, dt)

        # psi = s^2 / m^2
        psi = s2 / (m * m + eps)

        zV = corr_randn[:, 1:2]              # N(0,1) for variance
        u  = torch.rand_like(m)              # U(0,1) for exp-mixture branch

        # =========================================================
        # Case 1 (quadratic): valid for psi <= 2,
        # Andersen b^2:
        # b2 = 2/psi - 1 + sqrt(2/psi)*sqrt(2/psi - 1)
        # a  = m / (1 + b2)
        # v1 = a (sqrt(b2) + z)^2
        # =========================================================
        invpsi = 1.0 / (psi + eps)
        t = torch.clamp(2.0 * invpsi - 1.0, min=0.0)                 # ensures sqrt argument >= 0
        b2 = 2.0 * invpsi - 1.0 + torch.sqrt(2.0 * invpsi) * torch.sqrt(t)
        b2 = torch.clamp(b2, min=0.0)

        b  = torch.sqrt(b2)
        a  = m / (1.0 + b2)
        v1 = a * (b + zV) ** 2

        # =========================================================
        # Case 2 (exp-mixture): valid for psi >= 1.
        # p    = (psi - 1)/(psi + 1)
        # beta = (1 - p)/m
        # v2   = 0 if u <= p else log((1-p)/(1-u))/beta
        # Make it safe globally via clamping + smooth the indicator for AAD
        # using fuzzy logic.
        # =========================================================
        p = (psi - 1.0) / (psi + 1.0)
        p = torch.clamp(p, min=0.0, max=1.0 - 1e-6)                   
        beta = (1.0 - p) / (m + eps)

        one_minus_u = torch.clamp(1.0 - u, min=eps)
        one_minus_p = torch.clamp(1.0 - p, min=eps)
        v_tail = torch.log(one_minus_p / one_minus_u) / (beta + eps)

        # Smooth version of indicator(u > p): ~0 when u<p (mass at 0), ~1 when u>p (tail)
        w_mass = compute_degree_of_truth(u - p, self.perform_smoothing, 0.3)
        v2 = w_mass * v_tail

        # =========================================================
        # Smooth switch between case1 and case2 around psi_c = 1.5
        # Narrow band to avoid bias/creases
        # =========================================================
        psi_c = 1.5
        w = compute_degree_of_truth(psi - psi_c, self.perform_smoothing, 0.5)          # in [0,1]
        v_next = (1.0 - w) * v1 + w * v2
        #v_next = torch.clamp(v_next, min=0.0)

        # =========================================================
        # Andersen log spot update (eq. 33 form):
        # logS_{t+dt} = logS_t + r dt + K0 + K1 v + K2 v_next + sqrt(K3 v + K4 v_next) Z
        # =========================================================
        K0, K1, K2, K3, K4 = self.get_Ks(dt)

        var_int = K3 * v + K4 * v_next
        var_int = torch.clamp(var_int, min=0.0)
        vol = torch.sqrt(torch.clamp(var_int, min=eps))

        zS = corr_randn[:, 0:1]   # independent N(0,1) under QE

        logS_next = logS + rate * dt + K0 + K1 * v + K2 * v_next + vol * zS

        return torch.stack([logS_next.squeeze(-1), v_next.squeeze(-1)], dim=-1)
    
    def resolve_request(self, req, asset_id, state):
        """Resolve requests posed by all products and at each exposure timepoint."""

        if req.request_type == AtomicRequestType.SPOT:
            log_spot = state[:,0] 
            return torch.exp(log_spot)
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