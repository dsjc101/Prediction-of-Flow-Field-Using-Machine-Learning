"""
config.py
=========
Rectangular-duct extension of the 2-D channel / 3-D pipe PINN project.

GEOMETRY CHOICE (documented, not arbitrary):
  - Duct cross-section: height H = 0.10 m, width W = 0.20 m  =>  aspect
    ratio AR = W/H = 2.0 (a standard benchmark ratio tabulated throughout
    Shah, R.K. & London, A.L. (1978), "Laminar Flow Forced Convection in
    Ducts," Academic Press, Ch. 7 -- f*Re and centerline-velocity tables
    are given there for AR = 1, 2, 4, 8, ... so AR=2 is directly checkable
    against published values, not a made-up shape).
  - H = 0.10 m is chosen so the duct's *height* equals the pipe's
    *diameter* (D = 0.10 m) and the channel's full height (2h = 0.10 m)
    in the earlier two projects -- the three geometries share one linear
    scale, only the cross-sectional shape changes, exactly the same
    design choice the pipe report made relative to the channel.
  - Hydraulic diameter Dh = 4A/P (standard definition for internal duct
    flow, e.g. White, F.M. (2006), Viscous Fluid Flow, 3rd ed.,
    McGraw-Hill, Ch. 6) is what Re, the entrance-length correlation, and
    the friction factor are all defined on for non-circular ducts
    (Shah & London 1978, Ch. 1) -- NOT H or W individually.
  - nu = 1e-4 m^2/s, same fluid as both earlier projects (apples-to-apples
    comparison in Section 11-style tables).

Re SWEEP: identical train/test split to the channel and pipe projects,
so the only thing that changes across all three reports is geometry.
"""

import numpy as np

# ---------------------------------------------------------------- geometry
H = 0.10          # duct height [m]  (y in [0, H])
W = 0.20          # duct width  [m]  (z in [0, W])
AR = W / H        # aspect ratio = 2.0, cf. Shah & London (1978) Ch. 7 tables
A_CROSS = H * W                       # cross-sectional area [m^2]
PERIM = 2.0 * (H + W)                 # wetted perimeter [m]
DH = 4.0 * A_CROSS / PERIM            # hydraulic diameter [m] (White 2006, Ch.6)
                                       # = 4*0.02/0.60 = 0.1333 m for H=0.10,W=0.20

L_DOMAIN = 5.0     # streamwise domain length [m] (sized to fit Le at Re=600
                   # below; see generate_data.py which checks this at runtime,
                   # same self-check discipline as the pipe project Sec 4.3)

NU = 1.0e-4        # kinematic viscosity [m^2/s] -- identical to channel & pipe

# --------------------------------------------------------------- Re sweep
# Re defined on hydraulic diameter: Re = U_mean * Dh / nu
# (the standard non-circular-duct convention, Shah & London 1978 Ch.1;
#  White 2006 Ch.6) -- NOT on H, and NOT the same numeric U0 as the pipe
# project for a given Re, because Dh != D here.
RE_TRAIN = [50, 100, 200, 500]
RE_TEST_INTERP = 300
RE_TEST_EXTRAP = 600
RE_ALL = sorted(RE_TRAIN + [RE_TEST_INTERP, RE_TEST_EXTRAP])


def u_mean_for_Re(Re):
    """U_mean = Re * nu / Dh."""
    return Re * NU / DH


# entrance-length correlation functional form Le = K * Dh * Re, K aspect-
# ratio dependent -- functional form after Han, L.S. (1960), "Hydrodynamic
# Entrance Lengths for Incompressible Laminar Flow in Rectangular Ducts,"
# J. Appl. Mech., Trans. ASME, 27 (Series E), 403-409,
# https://doi.org/10.1115/1.3644015 -- the direct rectangular-duct analogue
# of Langhaar (1942) for the circular pipe, cited identically in the pipe
# report. As in the pipe report, K is NOT asserted from a literature table
# here: it is measured self-consistently from THIS project's own marching
# solver (duct_solver.py) at runtime (see generate_data.py), because the
# literature K depends on the convergence criterion (99% vs 99.9%) and on
# aspect ratio in a way not worth mis-quoting from memory. K_GUESS below is
# only used to size L_DOMAIN before the solver has run; it is overwritten
# by the measured value once generate_data.py runs.
K_ENTRANCE_GUESS = 0.075  # order-of-magnitude placeholder for AR=2, see above

# ------------------------------------------------------- network / training
NET_DEPTH = 4          # hidden layers -- UNCHANGED from channel & pipe projects
NET_WIDTH = 80         # width -- UNCHANGED, so any accuracy delta is
                       # attributable to geometry/physics, not capacity
                       # (identical rationale to pipe report Sec 8)

ADAM_ITERS = 2500
ADAM_LR = 8e-4
LBFGS_ITERS = 300

# CHANGED: the PINN now predicts and is scored against all four fields
# (u, v, w, p -- previously only u had reference data to train against;
# see duct_solver.py's Patankar & Spalding (1972)-based recovery of v, w,
# p for the reference data that makes this possible). The single "data"
# term is split into one per predicted quantity so each field's fit can
# be weighted (and searched over, see weight_search.py) independently --
# u remains weighted highest since it is both the dominant velocity
# component and the one with the least-approximated reference (v, w, p
# all pass through the extra Poisson/PPE recovery steps in duct_solver.py
# and so carry more reference-side numerical error than u itself).
# These are STARTING values only, identical in spirit to how the
# original 10:1:5:5:1 set was a starting point later replaced by
# adaptive_weights.py's gradient-norm balancing (Wang, Teng & Perdikaris
# 2021); weight_search.py now searches this space directly and its
# result, not this dict, should be treated as the final reported set.
LOSS_WEIGHTS = dict(data_u=8.733, data_v=0.110, data_w=11.109, data_p=2.421,
                     pde=32.183, inlet=0.683, wall=11.138, outlet=0.639)
# NOTE: no "axis" term here (rectangular ducts have no coordinate
# singularity -- contrast with the pipe's r=0 axis term, Sec 7.1 of the
# pipe report). Two *symmetry-plane* terms (y=H/2, z=W/2) are added
# instead if --quarter-domain sampling is used (see pinn_model.py) --
# these are an optional sampling-efficiency device, not a dimensionality
# reduction, since the true field genuinely varies over the full (y,z)
# cross-section (see README.md Sec 2 for the "why not axisymmetric"
# discussion analogous to the pipe report's Sec 2).

PDE_ADAPTIVE_WEIGHTS = True   # CHANGED: adaptive gradient-norm loss balancing
# (Wang, Teng & Perdikaris 2021) is now the DEFAULT for this project instead
# of an opt-in flag. Rationale: the fixed 10:1:5:5:1 weighting was carried
# over unchanged from the pipe project (Sec 9/12 of the duct report) and was
# never re-tuned for the duct's own loss-term gradient scales; the training
# log showed the pde-residual term moving in the opposite direction from the
# other four terms under those fixed weights, which is exactly the symptom
# the Wang et al. (2021) scheme is designed to correct. train_hybrid.py's
# --fixed-weights flag can be passed to opt back into the old fixed-weight
# behaviour for an A/B comparison.
# Wang, S., Teng, Y. & Perdikaris, P. (2021), "Understanding and mitigating
# gradient flow pathologies in physics-informed neural networks," SIAM J.
# Sci. Comput., 43(5), A3055-A3081, https://doi.org/10.1137/20M1318043

# ------------------------------------------------------------------ paths
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
FIG_DIR = "figures"

# BUG FIX: override the static L_DOMAIN guess above with the ACTUAL
# marched/saved domain length once generate_data.py has run and written it
# (see that file's run_all() for why this matters -- in short, every other
# script normalizes x by L_DOMAIN and samples PDE/BC collocation points in
# [0, L_DOMAIN], so L_DOMAIN must match the real extent of the saved
# reference data or a large fraction of it silently falls outside the
# network's trained input range). This makes the fix automatic: re-run
# generate_data.py once, and every script that imports config afterward
# (train_hybrid.py, weight_search.py, evaluate.py, evaluate_global.py, and
# pinn_model.py via its own `import config`) picks up the corrected value
# with no manual editing of this file required.
import os as _os
_l_domain_file = _os.path.join(DATA_DIR, "L_domain_used.txt")
if _os.path.exists(_l_domain_file):
    with open(_l_domain_file) as _f:
        L_DOMAIN = float(_f.read().strip())

if __name__ == "__main__":
    print(f"Duct: H={H} m, W={W} m, AR={AR:.2f}, Dh={DH:.4f} m")
    print(f"Re sweep: train={RE_TRAIN}, test_interp={RE_TEST_INTERP}, "
          f"test_extrap={RE_TEST_EXTRAP}")
    for Re in RE_ALL:
        print(f"  Re={Re:4d}  U_mean={u_mean_for_Re(Re):.4f} m/s  "
              f"Le_guess={K_ENTRANCE_GUESS*DH*Re:.3f} m")