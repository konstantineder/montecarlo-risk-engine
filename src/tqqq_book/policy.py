"""Call strikes determined from information available at the decision time."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CallPolicy:
    mode: str = 'fixed'
    vol_sensitivity: float = 1.0
    trend_sensitivity: float = 0.5
    drawdown_sensitivity: float = 0.5
    minimum: float = 0.01
    maximum: float = 0.50

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.call_policy, cfg.call_vol_sensitivity, cfg.call_trend_sensitivity,
                   cfg.call_drawdown_sensitivity, cfg.call_otm_min, cfg.call_otm_max)

    def otm(self, base: float, iv: float, initial_iv: float, trend: float = 0, drawdown: float = 0) -> float:
        if self.mode == 'fixed':
            return base
        value = base * (iv / max(initial_iv, 1e-8)) ** self.vol_sensitivity
        value += self.trend_sensitivity * trend + self.drawdown_sensitivity * drawdown
        return max(self.minimum, min(self.maximum, value))
