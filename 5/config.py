"""
config.py -- single source of truth for the turbulent-PIPE hybrid RANS+SA PINN.

Geometry : axisymmetric circular pipe, radius R, length L, uniform swirl-free inlet.
           Everything is non-dimensionalised with R and U_b  (x*=x/R, r*=r/R, U*=U/U_b).
Reynolds : Re = U_b * D / nu  with D = 2R   (same Re numbers as the turbulent channel
           report, whose D_h = 4h; so a pipe at the same Re has Re_R = U_b R/nu = Re/2).
Roles    : TRAIN / VAL / TEST exactly as in the turbulent channel report.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------
# Reynolds-number roles  (identical to the turbulent channel report)
# ----------------------------------------------------------------------------------
RE_TRAIN = [10_000, 20_000, 40_000, 80_000]
RE_VAL = [14_000, 56_000, 88_000]          # 14k, 56k interpolation ; 88k mild extrapolation
RE_TEST = [28_000, 96_000, 120_000]        # 28k interpolation ; 96k, 120k extrapolation
RE_ALL = RE_TRAIN + RE_VAL + RE_TEST
ROLE = {**{r: "TRAIN" for r in RE_TRAIN}, **{r: "VAL" for r in RE_VAL},
        **{r: "TEST" for r in RE_TEST}}
ROLE_NOTE = {10_000: "TRAIN", 20_000: "TRAIN", 40_000: "TRAIN", 80_000: "TRAIN",
             14_000: "VAL interp", 56_000: "VAL interp", 88_000: "VAL extrap 1.1x",
             28_000: "TEST interp", 96_000: "TEST extrap 1.2x", 120_000: "TEST extrap 1.5x"}
RE_MIN_TRAIN, RE_MAX_TRAIN = min(RE_TRAIN), max(RE_TRAIN)
RE_ALL_SORTED = sorted(RE_ALL)

# ----------------------------------------------------------------------------------
# Geometry / fluid (non-dimensional)
# ----------------------------------------------------------------------------------
L = 160.0           # pipe length / R  (verified against measured development length by
                    # verify_reference.py: L_dev / L must stay < ~0.7 for every Re)
KAPPA = 0.41

def nu_star(Re):
    """non-dimensional viscosity 1/Re_R, with Re_R = U_b R / nu = Re/2."""
    return 2.0 / Re

# ----------------------------------------------------------------------------------
# Reference-solver numerics
# ----------------------------------------------------------------------------------
N_R = 400           # wall-normal cells over y = 1 - r in [0, 1]
N_X = 800           # stored streamwise stations, x_k = L (k/(N_X-1))^2
MAX_DX = 0.25       # max marching sub-step (in R)
Y_PLUS_FIRST = 0.4  # first-cell-centre y+ at the highest Re

# ----------------------------------------------------------------------------------
# PINN data regimes (same logic as the channel report, lengths in R instead of h)
# ----------------------------------------------------------------------------------
X_DATA_MIN = 0.05
HOLDOUT_BANDS = [(4.0, 7.0), (25.0, 32.0), (70.0, 85.0)]   # withheld from the DENSE data loss
SPARSE_STATIONS = [3.0, 15.0, 50.0, 130.0]                 # SPARSE data = full profiles here
CF_X_MIN = 1.0      # Cf scored for x >= 1 R
R_AXIS_MIN = 0.02   # PDE collocation points kept at r >= this (1/r terms)

# ----------------------------------------------------------------------------------
# Training budgets
# ----------------------------------------------------------------------------------
BUDGETS = {
    # name           adam its, lbfgs its, n_data, n_pde, n_bc
    "smoke":        dict(adam=150,    lbfgs=10,  n_data=256,  n_pde=256,  n_bc=64),
    "cpu_search":   dict(adam=1_500,  lbfgs=0,   n_data=1024, n_pde=1024, n_bc=256),
    "cpu_final":    dict(adam=12_000, lbfgs=200, n_data=2048, n_pde=2048, n_bc=512),
}
NET_DEPTH, NET_WIDTH = 4, 64        # 4 x 64 on CPU.  use --big for 6 x 128 on GPU.

# default loss weights (the searched ones go in outputs/best_weights_*.json)
DEFAULT_WEIGHTS = dict(data=10.0, v_sub=0.05, p_sub=0.1, cont=0.1, mom=0.01, sa=0.05,
                       inlet=5.0, outlet=1.0, sym=1.0, lr=1e-3)

# ----------------------------------------------------------------------------------
# Re-only scaling quantities  (smooth-pipe Prandtl friction law; never uses a solved field)
#   1/sqrt(lam) = 2 log10(Re sqrt(lam)) - 0.8     lam = Darcy friction factor
#   Cf = lam/4 (Fanning),  u_tau/U_b = sqrt(lam/8),  Re_tau = (u_tau/U_b) * Re/2
# ----------------------------------------------------------------------------------
def prandtl_lambda(Re, iters=30):
    s = 4.0 * np.ones_like(np.asarray(Re, dtype=float))     # s = 1/sqrt(lam)
    for _ in range(iters):
        s = 2.0 * np.log10(np.asarray(Re, dtype=float) / s) - 0.8
    return 1.0 / s ** 2

def mckeon_lambda(Re, iters=60):
    """Princeton Superpipe fit, McKeon et al. (2005):  1/sqrt(lam) = 1.920 log10(Re sqrt(lam)) - 0.475."""
    s = 4.0 * np.ones_like(np.asarray(Re, dtype=float))
    for _ in range(iters):
        s = 1.920 * np.log10(np.asarray(Re, dtype=float) / s) - 0.475
    return 1.0 / s ** 2

def est_np(Re):
    lam = prandtl_lambda(Re)
    cf = lam / 4.0
    ut = np.sqrt(lam / 8.0)
    ret = ut * np.asarray(Re, dtype=float) / 2.0
    return cf, ut, ret

def re_tag(Re):
    return f"{int(Re)}"

def ref_path(Re):
    return os.path.join(DATA_DIR, f"ref_Re{re_tag(Re)}.npz")
