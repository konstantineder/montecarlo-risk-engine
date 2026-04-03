from context import *

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "exposure_tests"
    / "cva_large_netting_set_derivatives.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "cva_large_netting_set_derivatives",
    MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_cva_and_derivatives = _MODULE.compute_cva_and_derivatives


def test_large_netting_set_cva_aad_matches_finite_differences():
    common_kwargs = {
        "num_europeans": 8,
        "num_bonds": 4,
        "num_swaps": 40,
        "num_paths_mainsim": 1024,
        "num_paths_presim": 1024,
        "num_steps": 4,
        "exposure_timeline": np.linspace(0.0, 4.0, 30),
        "deterministic_credit": True,
    }

    aad = compute_cva_and_derivatives(
        spot=100.0,
        rate_level=0.03,
        derivative_method="aad",
        **common_kwargs,
    )
    finite_diff = compute_cva_and_derivatives(
        spot=100.0,
        rate_level=0.03,
        derivative_method="finite_difference",
        **common_kwargs,
    )

    assert np.isfinite(aad["dcva_dspot"])
    assert np.isfinite(aad["dcva_drate"])
    assert np.isfinite(finite_diff["dcva_dspot"])
    assert np.isfinite(finite_diff["dcva_drate"])

    assert abs(aad["dcva_dspot"] - finite_diff["dcva_dspot"]) < 2e-3
    assert abs(aad["dcva_drate"] - finite_diff["dcva_drate"]) < 0.1
