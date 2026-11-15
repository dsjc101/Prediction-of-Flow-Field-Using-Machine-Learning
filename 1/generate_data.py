"""
generate_data.py -- Runs entrance_solver.solve() (Bodoia & Osterle 1961
method) for every Reynolds number used in this project and caches the
result to outputs/reference_data.npz, for reuse by both training (data loss)
and validation (comparison against Re=300, 600, never trained on).

NOTE: entrance_solver.solve()'s pressure output had a sign bug (P was
integrated with the wrong sign and increased downstream instead of
decreasing -- see entrance_solver.py's inline fix and its docstring
verification against the exact fully-developed dP/dx). This script MUST
be re-run (and the model retrained on the corrected data) after pulling
that fix; any reference_data.npz / checkpoint generated before it has
p-values with the wrong sign.
"""

import os
import numpy as np
import config as cfg
from entrance_solver import solve

os.makedirs("outputs", exist_ok=True)
ALL_RE = cfg.RE_TRAIN + cfg.RE_TEST_INTERPOLATION

data = {}
for Re in ALL_RE:
    print(f"Solving Bodoia-Osterle marching solution for Re={Re} ...")
    sol = solve(Re, nx=500, ny=81)
    data[f"x_{Re}"] = sol["x"]
    data[f"y_{Re}"] = sol["y"]
    data[f"U_{Re}"] = sol["U"]
    data[f"V_{Re}"] = sol["V"]
    data[f"P_{Re}"] = sol["P"]

np.savez("outputs/reference_data.npz", **data)
print("Saved outputs/reference_data.npz")