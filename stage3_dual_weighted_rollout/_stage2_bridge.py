"""
Loads stage2_margin_weighting/generate_data.py by explicit file path under a
distinct module name ("stage2_generate_data"), rather than a plain
`from generate_data import ...`. Both stage2 and stage3 have their own file
named generate_data.py (and stage3 also has its own evaluate_mpc.py); a
plain same-name import from either would silently pick up whichever module
Python already has cached under that bare name in sys.modules, or trigger a
circular-import error when stage3's own generate_data.py does the
importing. This module sidesteps that entirely.
"""
from __future__ import annotations

import importlib.util
import os

_STAGE3_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_STAGE3_DIR)
_STAGE2_GEN_DATA_PATH = os.path.join(_ROOT_DIR, "stage2_margin_weighting", "generate_data.py")


def _load():
    spec = importlib.util.spec_from_file_location("stage2_generate_data", _STAGE2_GEN_DATA_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stage2_gd = _load()

true_step = _stage2_gd.true_step
POS_BOUND = _stage2_gd.POS_BOUND
INPUT_BOUND = _stage2_gd.INPUT_BOUND
DT = _stage2_gd.DT
