from context import *

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "exposure_tests"
    / "cva_large_netting_set_surface.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "cva_large_netting_set_surface",
    MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_cva_and_derivatives = _MODULE.compute_cva_and_derivatives


def test_large_netting_set_cva_is_positive_and_derivatives_are_finite():
    values = compute_cva_and_derivatives(
        spot=100.0,
        rate_level=0.03,
        num_europeans=10,
        num_bonds=5,
        num_swaps=50,
        num_paths_mainsim=1024,
        num_paths_presim=1024,
        num_steps=4,
        exposure_timeline=np.linspace(0.0, 4.0, 30),
        deterministic_credit=True,
    )

    assert values["cva"] > 0.0
    assert np.isfinite(values["dcva_dspot"])
    assert np.isfinite(values["dcva_drate"])
