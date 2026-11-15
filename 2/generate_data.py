"""
generate_data.py -- Runs entrance_solver.solve() (Hornbeck 1964 method)
for every Reynolds number used in this project and caches the result to
outputs/reference_data.npz, for reuse by both training (data loss) and
validation (comparison against Re=300, 600, never trained on). Unchanged
in spirit from the 2-D project's generate_data.py; only the field names
switch from (x,y)->(x,r) and V is now the radial velocity.
"""

import numpy as np
import config as cfg
from entrance_solver import solve

ALL_RE = cfg.RE_TRAIN + cfg.RE_TEST_INTERPOLATION

data = {}
for Re in ALL_RE:
    print(f"Solving Hornbeck (1964) marching solution for Re={Re} ...")
    sol = solve(Re, nx=500, nr=81)
    data[f"x_{Re}"] = sol["x"]
    data[f"r_{Re}"] = sol["r"]
    data[f"U_{Re}"] = sol["U"]
    data[f"V_{Re}"] = sol["V"]
    data[f"P_{Re}"] = sol["P"]

np.savez("outputs/reference_data.npz", **data)
print("Saved outputs/reference_data.npz")