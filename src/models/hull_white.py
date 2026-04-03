from model import *
from request_interface.request_interface import *
from typing import List

# Hull-White 1-Factor Model for Stochastic Interest Rates
# Initial implementation
# TODO: Fix!!
class HullWhiteModel(Model):
    def __init__(self, 
                 calibration_date         : float, 
                 rate                     : float, 
                 initial_forward_curve    : List[float], 
                 forward_curve_derivative : List[float],
                 mean_reversion           : float, 
                 volatility
                 ):
        super().__init__(calibration_date)

        # Model parameters
        self.rate = torch.tensor(rate, dtype=FLOAT, device=device)
        self.sigma = torch.tensor(volatility, dtype=FLOAT, device=device)
        self.mean_reversion = torch.tensor(mean_reversion, dtype=FLOAT, device=device)

        # Forward curve and its derivative are assumed given as values at discrete times
        self.curve_times = torch.linspace(0, 1, steps=len(initial_forward_curve)).to(device)  
        self.forward_curve = torch.tensor(initial_forward_curve, dtype=FLOAT, device=device)
        self.forward_curve_derivative = torch.tensor(forward_curve_derivative, dtype=FLOAT, device=device)

        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            self.rate,
            self.sigma,
            self.mean_reversion,
            *self.forward_curve,
            *self.forward_curve_derivative
        ]
    
    def interpolate(self, t, curve):
        idx = torch.searchsorted(self.curve_times, torch.tensor(t).to(device)) - 1
        idx = torch.clamp(idx, 0, len(self.curve_times) - 2)

        t1 = self.curve_times[idx]
        t2 = self.curve_times[idx + 1]
        y1 = curve[idx]
        y2 = curve[idx + 1]

        return y1 + (y2 - y1) * (t - t1) / (t2 - t1)

    def get_model_param_names(self) -> list[str]:
        curve_names = [f"forward_curve[{idx}]" for idx in range(len(self.forward_curve))]
        derivative_names = [
            f"forward_curve_derivative[{idx}]"
            for idx in range(len(self.forward_curve_derivative))
        ]
        return ["rate", "sigma", "mean_reversion", *curve_names, *derivative_names]
    
    def compute_theta(self, t):
        f_t = self.interpolate(t, self.forward_curve)
        df_dt = self.interpolate(t, self.forward_curve_derivative)
        a = self.mean_reversion
        sigma = self.sigma
        return df_dt + a * f_t + (sigma ** 2 / (2 * a)) * (1 - torch.exp(-2 * a * t))
    
    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        r_t = self.rate.expand(num_paths).clone()
        log_B_t = torch.zeros_like(r_t)
        paths = []

        t_start = self.calibration_date

        for t_end in timeline:
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for step in range(num_steps):
                t = t_start + step * dt
                exp_decay = torch.exp(-self.mean_reversion * dt)
                theta_t = self.compute_theta(t)

                mean = r_t * exp_decay + theta_t * (1 - exp_decay) / self.mean_reversion
                variance = (self.sigma ** 2 / (2 * self.mean_reversion)) * (1 - exp_decay ** 2)

                noise = torch.randn(num_paths, device=device, dtype=FLOAT)
                r_t = mean + torch.sqrt(variance) * noise
                log_B_t += r_t * dt

            paths.append(torch.stack([r_t, log_B_t], dim=1))
            t_start = t_end

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline), 2]
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        r_t = self.rate.expand(num_paths).clone()
        log_B_t = torch.zeros_like(r_t)
        paths = []

        t_start = self.calibration_date

        for t_end in timeline:
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for step in range(num_steps):
                t = t_start + step * dt
                dW = torch.randn(num_paths, device=device, dtype=FLOAT) * torch.sqrt(dt)
                theta_t = self.compute_theta(t)

                dr = (theta_t - self.mean_reversion * r_t) * dt + self.sigma * dW
                r_t += dr
                log_B_t += r_t * dt

            paths.append(torch.stack([r_t, log_B_t], dim=1))
            t_start = t_end

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline), 2]

    def compute_bond_price(self, time1, time2, rate):
        B = (1 - torch.exp(-self.mean_reversion * (time2 - time1))) / self.mean_reversion
        A = torch.exp((B - (time2 - time1)) * self.interpolate(time1, self.forward_curve) -
                      (self.sigma**2 / (4 * self.mean_reversion)) * B**2)
        return A * torch.exp(-B * rate)
    
    def resolve_request(self, req, state):
        if req.request_type == AtomicRequestType.SPOT:
            return state[0]

        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            time = req.time1
            return self.compute_bond_price(self.calibration_date, time, state[0])

        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            return self.compute_bond_price(time1, time2, state[0])

        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            bond_price = self.compute_bond_price(time1, time2, state[0])
            return (1 / bond_price - 1) / (time2 - time1)

        elif req.request_type == AtomicRequestType.NUMERAIRE:
            log_B_t = state[1]
            return torch.exp(log_B_t)
