"""
config.py -- single source of truth for the turbulent-channel PINN project.

Everything is NON-DIMENSIONAL:
    lengths  / h      (h = channel HALF-height, wall at y=0, centreline at y=1)
    velocity / U_b    (U_b = bulk velocity)
    pressure / (0.5 U_b^2)   -> same "Cp-like" normalisation as the laminar report

Re definition: SAME as the laminar report  ->  Re = U_b * D_h / nu,  D_h = 4h.
So the half-height Reynolds number is  Re_h = U_b h / nu = Re / 4.
(Re_tau ~ 0.0955*(Re/2)^(7/8), Dean 1978; Re=1e4 -> Re_tau~170, Re=1.2e5 -> ~1450.)

Geometry note
-------------
The report's laminar L is not known to me (config.py of that project was not
uploaded). Turbulent flow needs a much longer duct than laminar flow before it
is developed, so L is set here in units of h. verify_reference.py prints the
actual development length of every Re so you can check L is long enough.

Domain reduction: the mean flow is mirror-symmetric about the centreline, so
only the lower half y in [0, h] is computed (symmetry BC at y=h). Predictions
are mirrored back to [-h, h] for plotting. This halves the cost and removes
any chance of a symmetry-breaking PINN solution.
"""
import os
import numpy as np

# ------------------------------------------------------------------ paths
ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(ROOT, "reference_data")
OUT_DIR = os.path.join(ROOT, "outputs")
DNS_DIR = os.path.join(ROOT, "dns_data")
for _d in (REF_DIR, OUT_DIR, DNS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------- geometry
L_OVER_H = 160.0           # streamwise length / half-height (see note above)

# ------------------------------------------------------- Reynolds-number roles
# Roles are FIXED here and enforced in code (data.py):
#   TRAIN : direct data supervision + physics
#   VAL   : used ONLY for early stopping and for the weight search
#   TEST  : touched ONLY by evaluate.py --final  (never during tuning)
# Cawley & Talbot (2010, JMLR 11:2079) -- tuning on the data you report on
# gives optimistic bias; hence VAL and TEST are different Re.
RE_TRAIN = [10_000, 20_000, 40_000, 80_000]
RE_VAL   = [14_000, 56_000, 88_000]          # 14k, 56k interpolation; 88k = mild EXTRAPOLATION (1.1x Re_max)
# 88k is in VAL on purpose: interpolation alone cannot reward the physics terms (a data-fit
# interpolates smoothly in Re), so the weight search needs ONE case where physics can matter.
# TEST extrapolation (96k = 1.2x, 120k = 1.5x) stays untouched; 120k is the stringent one.
RE_TEST_INTERP = [28_000]                    # interpolation (log-midpoint of 20k, 40k)
RE_TEST_EXTRAP = [96_000, 120_000]           # 1.2x and 1.5x the largest train Re
RE_TEST = RE_TEST_INTERP + RE_TEST_EXTRAP
RE_ALL = sorted(set(RE_TRAIN + RE_VAL + RE_TEST))

RE_MIN, RE_MAX = float(min(RE_TRAIN)), float(max(RE_TRAIN))


def re_h(Re):
    """Half-height Reynolds number U_b h / nu."""
    return Re / 4.0


# --------------------------------------------------------------- SA model
# Spalart & Allmaras (1992, AIAA-92-0439; 1994 La Rech. Aerosp. 1:5-21).
# 'SA-noft2' form with the S~ limiter of Allmaras, Johnson & Spalart (2012,
# ICCFD7-1902).
SA = dict(
    cb1=0.1355, cb2=0.622, sigma=2.0 / 3.0, kappa=0.41,
    cw2=0.3, cw3=2.0, cv1=7.1,
)
SA["cw1"] = SA["cb1"] / SA["kappa"] ** 2 + (1.0 + SA["cb2"]) / SA["sigma"]
# Free-stream (inlet) value: nu_tilde_inf = 3 nu (Spalart & Rumsey 2007,
# AIAA J. 45(10):2544) -- the value recommended for fully turbulent inflow.
SA_NUT_INLET_OVER_NU = 3.0


# ------------------------------------------------- Re-only scaling functions
def retau_dean(Re):
    """Friction Reynolds number estimate from Dean (1978, J. Fluids Eng.
    100:215):  Cf = 0.073 Re_2h^-0.25  ->  Re_tau = 0.0955 Re_2h^(7/8),
    Re_2h = U_b(2h)/nu = Re/2.   Depends on Re ONLY (no solved field)."""
    re2h = Re / 2.0
    return 0.0955 * re2h ** 0.875


def utau_over_ub_dean(Re):
    re2h = Re / 2.0
    cf = 0.073 * re2h ** -0.25
    return np.sqrt(cf / 2.0)


# Normalisation of the network outputs (all functions of Re only):
#   u_hat = U / U_b ;   v_hat = V / (VSCALE_FRAC * u_tau,est) ;   p_hat = P / (0.5 U_b^2)
# VSCALE_FRAC: the mean wall-normal velocity is tiny (rms ~ 0.02-0.04 u_tau for
# x>0.5h, measured with the reference solver at Re=1e4 and 4e4), so V/U_b ~ 1e-3
# would be invisible to an MSE loss. Scaling by 0.05*u_tau,est(Re) makes V_hat O(1)
# near the inlet. Re-only function; nothing is learnt from a solved field at run time.
VSCALE_FRAC = 0.05


def v_scale(Re):
    return VSCALE_FRAC * utau_over_ub_dean(Re)


# ----------------------------------------------------------- reference grid
NY_REF = 400               # wall-normal cells (half channel), stretched
NX_REF = 800               # streamwise stations stored (power-2 clustered towards the inlet)
Y_STRETCH_YPLUS1 = 0.4     # target first-cell centre y+ at the HIGHEST Re

# ----------------------------------------------------------- PINN defaults
HIDDEN = 6
WIDTH = 128
SEED = 0


# ---------------------------------------------------------- fast debug mode
# PINN_FAST=1 python <script>   -> coarse reference + separate cache directory,
# for smoke-testing the pipeline in seconds. NEVER use for reported results.
if os.environ.get("PINN_FAST"):
    NY_REF = 120
    NX_REF = 200
    REF_DIR = os.path.join(ROOT, "reference_data_FAST")
    OUT_DIR = os.path.join(ROOT, "outputs_FAST")
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------- data / hold-out settings
X_DATA_MIN = 0.05            # x/h below which the marched solution is under-resolved (BL thinner than ~3 cells)
# Streamwise bands (x/h) withheld from the DATA loss at every training Re.
# Physics (PDE) loss still covers them -> measures within-Re generalisation.
HOLDOUT_X_BANDS = [(4.0, 7.0), (25.0, 32.0), (70.0, 85.0)]
# Fully developed region used for DNS / Poiseuille-type comparisons
X_FD_FRACTION = 0.9          # x >= 0.9 L


# ------------------------------------------------------------ sparse-data mode
# PINN_SPARSE=1 (or  run_all.py --sparse):  data supervision ONLY on full wall-normal profiles at a few
# streamwise stations per TRAIN Re (like a measurement campaign); everywhere else only the physics + BCs act.
# Results go to outputs_sparse/ so they never overwrite the dense-data run. Stations avoid the hold-out bands.
SPARSE = bool(os.environ.get("PINN_SPARSE"))
SPARSE_STATIONS = [3.0, 15.0, 50.0, 130.0]            # x/h
if SPARSE:
    OUT_DIR = OUT_DIR + "_sparse"
    os.makedirs(OUT_DIR, exist_ok=True)