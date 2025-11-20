from common.packages import *

def symmetric_linear_smoothing(x,is_fuzzy,eps):
    if not is_fuzzy:
        return (x > 0).float()
    return torch.clamp((x + eps) / (2 * eps), min=0.0, max=1.0)
    
def compute_degree_of_truth(x,is_fuzzy,eps=0.05):
    return symmetric_linear_smoothing(x,is_fuzzy,eps)

def sigmoid_smoothing(x, beta=500):
    return torch.sigmoid(beta * x)

def bisection_search(func, low: float = 1e-10, high: float = 5.0,
                     tolerance: float = 1e-12, iters: int = 100) -> float:
    value_low, value_high = func(low), func(high)
    cnt = 0
    while value_low * value_high > 0.0 and cnt < 20:
        high *= 2.0
        value_high = func(high)
        cnt += 1
    if value_low * value_high > 0.0:
        raise RuntimeError("Could not bracket hazard in bucket")
    for _ in range(iters):
        mid = 0.5 * (low + high)
        value_mid = func(mid)
        if abs(value_mid) < tolerance or (high - low) < 1e-12:
            return mid
        if value_low * value_mid <= 0.0:
            high, value_high = mid, value_mid
        else:
            low, value_low = mid, value_mid
    return 0.5 * (low + high)

