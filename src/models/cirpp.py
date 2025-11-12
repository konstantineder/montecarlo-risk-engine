# cirpp_model.py
from models.model import *
from request_interface.request_interface import AtomicRequest, AtomicRequestType
from helpers.cs_helper import CSHelper

class CIRPPModel(Model):
    """
    Shifted Cox–Ingersoll–Ross (CIR++) intensity model for stochastic default intensities.

    Intensity:  λ(t) = y(t) + ψ(t)
    y(t) solves: dy = κ(θ - y) dt + σ sqrt(y) dW, with Feller 2κθ > σ² and y0 > 0
    ψ(t) chosen to fit initial market survival curve S_m(0,t).

    Implements:
      - Survival probability S(t,T) under CIR++ (closed form via CIR A,B)
      - Credit spread Sp(t,T) = -(1/(T-t)) * log[ δ + (1-δ) S(t,T) ]    (Theorem 3.1)
      - Intensity requests
      - Basic Euler simulation for y(t)

    References: Theorem 2.1 and 3.1, plus definitions of A(t,T), B(t,T), h, D(t)=d/dt ln A(0,t), E(t)=dB(0,t)/dt. :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        calibration_date: float,
        asset_id: str,
        hazard_rates: dict[float, float],  # piecewise-constant bootstrapped market hazards (1y buckets)
        kappa: float,
        theta: float,
        volatility: float,
        y0: float,
    ):
        super().__init__(
            calibration_date=calibration_date, 
            state_dim=2,
            asset_ids=[asset_id],
        )

        assert 2 * kappa * theta - volatility**2 > 0 and y0 > 0, "Feller condition not met."

        self.model_params = [
            torch.tensor(kappa, dtype=FLOAT, device=device),
            torch.tensor(theta, dtype=FLOAT, device=device),
            torch.tensor(volatility, dtype=FLOAT, device=device),
            torch.tensor(y0, dtype=FLOAT, device=device),
        ]
        
        self.tenors = torch.tensor(list(hazard_rates.keys()), dtype=FLOAT, device=device)
        self.hazard_rates = torch.tensor(list(hazard_rates.values()), dtype=FLOAT, device=device)
        
        self.cs_helper = CSHelper()

    # --------- Helpers to fetch parameters ----------
    def get_kappa(self):
        return torch.stack([self.model_params[0]])

    def get_theta(self):
        return torch.stack([self.model_params[1]])

    def get_sigma(self):
        return torch.stack([self.model_params[2]])

    def get_y0(self):
        return torch.stack([self.model_params[3]])

    # --------- Market intensity & initial survival S_m(0,t) ----------
    def _lambda_market(self, t: torch.Tensor) -> torch.Tensor:
        """
        Piecewise-constant hazard from the provided list.
        For t beyond last bucket, use last value.
        """
        for idx, tenor in enumerate(self.tenors):
            if t <= tenor:
                return self.hazard_rates[idx]
        idx = len(self.tenors) - 1
        return self.hazard_rates[idx]

    # --------- CIR++ shift ψ(t) = λ_m(t) + D(t) - y0 E(t)  ----------
    # h, A, B as in the paper (with T>t). :contentReference[oaicite:4]{index=4}
    def _h(self):
        kappa = self.get_kappa()
        sigma = self.get_sigma()
        return torch.sqrt(kappa * kappa + 2.0 * sigma * sigma)

    def _A(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        kappa = self.get_kappa()
        theta = self.get_theta()
        sigma = self.get_sigma()
        h = self._h()
        dt = torch.as_tensor(T, dtype=FLOAT, device=device) - torch.as_tensor(t, dtype=FLOAT, device=device)
        exp_h_dt = torch.exp(0.5 * (kappa + h) * dt)
        num = 2.0 * h * exp_h_dt
        den = 2.0 * h + (kappa + h) * (torch.exp(h * dt) - 1.0)
        pow_ = (2.0 * kappa * theta) / (sigma * sigma)
        return (num / den).pow(pow_)

    def _B(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        kappa = self.get_kappa()
        sigma = self.get_sigma()
        h = self._h()
        dt = torch.as_tensor(T, dtype=FLOAT, device=device) - torch.as_tensor(t, dtype=FLOAT, device=device)
        e = torch.exp(h * dt) - 1.0
        den = 2.0 * h + (kappa + h) * e
        return (2.0 * e) / den

    # D(t) = d/dt ln A(0,t), E(t) = d/dt B(0,t), both evaluated from 0 to t. :contentReference[oaicite:5]{index=5}
    def _D(self, t: torch.Tensor) -> torch.Tensor:
        kappa = self.get_kappa()
        theta = self.get_theta()
        sigma = self.get_sigma()
        h = self._h()
        et = torch.exp(h * t)
        num = 0.5 * (kappa + h) - (h * (kappa + h) * et) / (2.0 * h + (kappa + h) * (et - 1.0))
        return (2.0 * kappa * theta / (sigma * sigma)) * num

    def _E(self, t: torch.Tensor) -> torch.Tensor:
        kappa = self.get_kappa()
        sigma = self.get_sigma()
        h = self._h()
        et = torch.exp(h * t)
        num = 4.0 * h * h * et
        den = (2.0 * h + (kappa + h) * (et - 1.0)) ** 2
        return num / den

    def psi(self, t: torch.Tensor) -> torch.Tensor:
        lam_m = self._lambda_market(t)
        D = self._D(t)
        E = self._E(t)
        y0 = self.get_y0()
        return lam_m + D - y0 * E

    # --------- Model state, simulation ----------
    def get_state(self, num_paths: int):
        """Return initial state y0 expanded to num_paths."""
        y0 = self.get_y0().expand(num_paths)
        log_B0 = torch.zeros_like(y0, dtype=FLOAT, device=device)
        state = torch.stack([y0, log_B0], dim=-1) 
        return state

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor   
    ) -> torch.Tensor:
        """
        Euler–Maruyama (full-truncation) for CIR:
            y_{t+Δ} = y + κ(θ - y)+ σ sqrt(max(y,0)) sqrt(Δ) z
        """
        delta_t = time2 - time1
        kappa = self.get_kappa()
        theta = self.get_theta()
        sigma = self.get_sigma()
        y = state[:,0:1]
        log_B = state[:,1:2]
        sqrt_y = torch.sqrt(torch.clamp(y, min=0.0))
        y_next = y + kappa * (theta - y) * delta_t + sigma * sqrt_y * torch.sqrt(delta_t) * corr_randn
        log_B_next = log_B + self.lambda_t(time1, y) * delta_t
        return torch.stack([torch.clamp(y_next, min=1e-12).squeeze(-1), log_B_next.squeeze(-1)], dim=-1)

    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor      
    ) -> torch.Tensor:
        """
        Proxy 'analytic' step via moment-matching lognormal proxy for y(t+dt).
        For production, you can swap to exact noncentral-χ² sampling for CIR. :contentReference[oaicite:6]{index=6}
        """
        # compute conditional mean/variance of CIR and sample via lognormal proxy
        delta_t = time2 - time1
        kappa = self.get_kappa().item()
        theta = self.get_theta().item()
        sigma = self.get_sigma().item()
        y = state

        # Conditional mean and variance of CIR
        ekt = torch.exp(torch.tensor(-kappa * delta_t, dtype=FLOAT, device=device))
        m = theta + (y - theta) * ekt
        v = (sigma**2) * (
            y * ekt * (1 - ekt) / kappa
            + 0.5 * theta * (1 - ekt) ** 2 / kappa
        )

        # lognormal proxy (ensure positivity, small floor)
        eps = 1e-12
        var_ratio = torch.clamp(v / (m * m + eps), min=1e-12)
        mu_ln = torch.log(torch.clamp(m, min=eps)) - 0.5 * torch.log1p(var_ratio)
        sig_ln = torch.sqrt(torch.log1p(var_ratio))
        # if corr_randn provided, use it; otherwise standard
        z = corr_randn if corr_randn is not None else torch.randn_like(y)
        y_next = torch.exp(mu_ln + sig_ln * z)
        return torch.clamp(y_next, min=1e-12)

    # --------- Core quantities: λ(t), S(t,T), credit spread ----------
    def lambda_t(self, t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Model intensity at time t: λ(t) = y(t) + ψ(t)."""
        return y_t + self.psi(t)

    def survival_probability(self, t: torch.Tensor, T: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        S(t,T) under CIR++:
          S(t,T) = [S_m(0,T)/S_m(0,t)] * [A(0,t)/A(0,T)] * exp(-B(0,t)y0 + B(0,T)y0) * A(t,T) * exp(-B(t,T)[λ(t)-ψ(t)])
        which simplifies to the standard closed form presented in the paper. :contentReference[oaicite:7]{index=7}
        """
        t = torch.as_tensor(t, dtype=FLOAT, device=device)
        T = torch.as_tensor(T, dtype=FLOAT, device=device)

        # objects at (0,·)
        A0t = self._A(torch.tensor(0.0, dtype=FLOAT, device=device), t)
        A0T = self._A(torch.tensor(0.0, dtype=FLOAT, device=device), T)
        B0t = self._B(torch.tensor(0.0, dtype=FLOAT, device=device), t)
        B0T = self._B(torch.tensor(0.0, dtype=FLOAT, device=device), T)

        # Sm(0,·)
        Sm0_t = 1.0 - self.cs_helper.probability_of_default(
            hazards=self.hazard_rates, 
            tenors=self.tenors,
            date=t
            )
        Sm0_T = 1.0 - self.cs_helper.probability_of_default(
            hazards=self.hazard_rates, 
            tenors=self.tenors,
            date=T
            )

        y0 = self.get_y0()
        A_tT = self._A(t, T)
        B_tT = self._B(t, T)
        lam_t = self.lambda_t(t, y_t)
        psi_t = self.psi(t)

        pref = (Sm0_T / Sm0_t) * (A0t / A0T) * torch.exp(-B0t * y0 + B0T * y0)
        tail = A_tT * torch.exp(-B_tT * (lam_t - psi_t))
        return pref * tail

    def credit_spread(self, t: torch.Tensor, T: torch.Tensor, y_t: torch.Tensor, delta: float = 0.40) -> torch.Tensor:
        """
        Sp(t,T) = -(1/(T-t)) * ln[ δ + (1-δ) S(t,T) ]  (Theorem 3.1). :contentReference[oaicite:8]{index=8}
        """
        S_tT = self.survival_probability(t, T, y_t)
        dt = torch.as_tensor(T, dtype=FLOAT, device=device) - torch.as_tensor(t, dtype=FLOAT, device=device)
        dt = torch.clamp(dt, min=1e-12)
        inside = torch.clamp(torch.as_tensor(delta, dtype=FLOAT, device=device) + (1.0 - delta) * S_tT, min=1e-24)
        return -torch.log(inside) / dt

    # --------- Requests ----------
    def resolve_request(self, req: AtomicRequest, asset_id: str, state: torch.Tensor) -> torch.Tensor:
        """
        Supported custom request types (rename to your real enum values if needed):
          - AtomicRequestType.SURVIVAL_PROBABILITY 
          - AtomicRequestType.CONDITIONAL_SURVIVAL_PROBABILITY   (expects req.time1=t, req.time2=T)
        """
        # state is y(t) for each path
        if req.request_type == AtomicRequestType.CONDITIONAL_SURVIVAL_PROBABILITY:
            t = req.time1
            T = req.time2
            y = state[:,0]

            return self.survival_probability(t, T, y)
        
        if req.request_type == AtomicRequestType.SURVIVAL_PROBABILITY:
            log_B_t=state[:,1]
            return torch.exp(log_B_t) 

        else:
            raise NotImplementedError(f"Request type {req.request_type} not supported by CIRpp.")
