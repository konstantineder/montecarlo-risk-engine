from __future__ import annotations
from enum import Enum

class SimulationScheme(Enum):
    """Enum for simulation schemes."""
    EULER = 0
    MILSTEIN = 1
    ANALYTICAL = 2