from models.model import *
from request_interface.request_interface import AtomicRequestType

# Black-Scholes model for a single asset
class BlackScholesModel(Model):
    def __init__(self, 
                 calibration_date : float, # Date of model calibration
                 spot             : float, # Spot price of the asset
                 rate             : float, # Risk-free interest rate
                 sigma            : float  # Volatility
                 ):
        
        super().__init__(calibration_date)
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        # Create leaf params ONCE with gradients enabled
        self.S0    = torch.tensor([spot],  dtype=FLOAT, device=device, requires_grad=True)
        self.sigma = torch.tensor([sigma], dtype=FLOAT, device=device, requires_grad=True)
        self.r     = torch.tensor([rate],  dtype=FLOAT, device=device, requires_grad=True)

        # Keep an ordered tuple for autograd.grad
        self._model_params = (self.S0, self.sigma, self.r)

    # Use these SAME tensors everywhere; do not re-wrap/stack
    def get_model_params(self):
        return self._model_params

    def get_spot(self):
        return self.S0          # no torch.tensor/stack/detach/item

    def get_volatility(self):
        return self.sigma

    def get_rate(self):
        return self.r

    # Simulate Monte Carlo paths using analytic formulae
    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        # use repeat instead of expand+clone (fewer edge cases for grads)
        spot = self.get_spot() * torch.ones(num_paths, dtype=FLOAT, device=device)


        sigma = self.get_volatility()
        rate  = self.get_rate()
        paths = []

        t_start = self.calibration_date
        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                z = torch.randn(num_paths, dtype=FLOAT, device=device)
                spot = spot * torch.exp((rate - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z)

            paths.append(spot)
            t_start = t_end

        return torch.stack(paths, dim=1)

    
    # Simulate Monte Carlo paths applying Euler-Mayurama scheme 
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        spot = self.get_spot().expand(num_paths).clone()
        sigma = self.get_volatility()
        rate= self.get_rate()
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_start = timeline[i]
            t_end = timeline[i + 1]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                z = torch.randn(num_paths, device=device)
                dS = rate * spot * dt + sigma * spot * torch.sqrt(dt) * z
                spot = spot + dS

            paths.append(spot)
            t_start=t_end

        return torch.stack(paths, dim=1)  # shape: [num_paths, len(timeline)]
    
    # Resolve requests posed by all products and at each exposure timepoint
    def resolve_request(self, req, state):

        if req.request_type == AtomicRequestType.SPOT:
            return state
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
            # Add more request types as needed