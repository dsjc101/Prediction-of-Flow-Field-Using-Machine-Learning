"""
config.py -- single source of truth for the circular-pipe (axisymmetric)
entrance-flow PINN project. This is the 3-D-flow follow-up to the
2-D plane-channel project (config.py / entrance_solver.py / model.py /
physics.py / train_hybrid.py / evaluate.py in that order), carried over
with the SAME structure, the SAME hybrid data+physics architecture, and
the SAME train/test Re split, so results are directly comparable.

---------------------------------------------------------------------
WHY AXISYMMETRIC (NOT A FULL 3-D CARTESIAN DOMAIN)
---------------------------------------------------------------------
A circular pipe with a uniform inlet profile and no swirl has NO
dependence on the azimuthal angle theta -- the physics (governing
equations, boundary conditions, forcing) is invariant under rotation
about the pipe axis, so the exact solution u(x, r, theta) = u(x, r) has
no theta-dependence either. This is not an approximation of convenience;
it is the standard formulation for this exact class of problem in the
literature this project's method is drawn from -- Hornbeck (1964)
[https://doi.org/10.1007/BF00382049] solves the developing-pipe-flow
problem in (x, r) for exactly this reason, and Bodoia & Osterle's own
plane-channel method (Appl. Sci. Res. A10, 265-276, 1961; used unchanged
in the 2-D project) is its 2-D-Cartesian counterpart. Recent axisymmetric
PINN work follows the same reduction for jet/pipe-type flows, e.g. an
axisymmetric-RANS PINN for submerged jet grout flow (Acta Geotechnica,
2026, https://doi.org/10.1007/s11440-026-02979-7) collapses the same way
(separate networks for U, V, P over (x, r) only).

So "3-D" here means: the flow domain is a genuine 3-D solid of revolution
(a cylinder), and the model reproduces the full 3-D velocity/pressure
field at ANY (x, y, z) via u(x, r, Re), v_r(x, r, Re) -> (u_x, u_y, u_z)
by projecting the radial velocity onto (y, z) = r*(cos(theta), sin(theta))
-- see model.py's `to_cartesian`. What is exploited is that the physics
itself only needs 2 spatial coordinates (x, r) to be represented exactly;
this is the standard, not a corner-cut, and it is exactly why an
axisymmetric solver/network is tractable at all (a fully general 3-D
Cartesian PINN for this same laminar, swirl-free case would spend most of
its capacity re-learning the theta-invariance the geometry already
guarantees).

---------------------------------------------------------------------
GEOMETRY AND Re SWEEP
---------------------------------------------------------------------
Pipe radius R is chosen, as the channel half-height h was in the 2-D
project, so the Re=600 extrapolation case's development length fits
comfortably inside the domain (see entrance_length() below and the
L sizing note).

Re sweep is KEPT IDENTICAL to the 2-D project (RE_TRAIN =
[50, 100, 200, 500], RE_TEST = [300 interpolation, 600 extrapolation]),
per the assignment: same Re train, same Re test, so the generalization
comparison (train vs. never-seen Re) is apples-to-apples across the two
geometries. Re is defined here on the pipe diameter, Re = U_mean*D/nu,
D = 2R -- the standard internal-pipe-flow Reynolds number (White, F.M.
(2006), Viscous Fluid Flow, 3rd ed., McGraw-Hill, Ch. 6; Munson, Young &
Okiishi, Fundamentals of Fluid Mechanics, Ch. 8) -- exactly analogous to
the 2-D project's Re = U_mean*Dh/nu with Dh = 2h for parallel plates.
"""

import numpy as np

# ---------------------------------------------------------------- geometry
R = 0.05             # pipe radius [m] -- numerically identical to the
                      # 2-D project's channel half-height h, so the two
                      # geometries are the same "size" and only the
                      # cross-sectional shape (slot vs. circle) differs.
D = 2.0 * R           # pipe diameter [m] (the pipe's hydraulic diameter,
                      # Dh = 4*Area/Perimeter = 4*(pi R^2)/(2 pi R) = 2R,
                      # is D itself for a circular pipe -- White (2006)
                      # Ch. 6.)
L = 4.0               # streamwise domain length [m] -- sized, as in the
                      # 2-D project, so the Re=600 extrapolation case's
                      # development length (~3.0 m, Le ~= 0.06*Re*D, see
                      # entrance_length() below) fits comfortably inside
                      # the domain.
X_MIN, X_MAX = 0.0, L
R_MIN, R_MAX = 0.0, R  # axisymmetric radial coordinate: axis (r=0) to wall (r=R)

# ---------------------------------------------------------------- fluid
NU = 1.0e-4           # kinematic viscosity [m^2/s] -- same fluid as the
                       # 2-D project, for direct numerical comparability.

# ---------------------------------------------------------------- Re sweep
# Identical train/test split to the 2-D channel project, by design (see
# module docstring).
RE_TRAIN = [50.0, 100.0, 200.0, 500.0]
RE_TEST_INTERPOLATION = [300.0, 600.0]
ALL_RE_FOR_EVAL = sorted(set(RE_TRAIN + RE_TEST_INTERPOLATION))


def inlet_velocity(Re: float) -> float:
    """U0 = Re * nu / D -- mean (=inlet, uniform-profile) velocity for a
    given Re, from the pipe-diameter Reynolds-number definition
    Re = U_mean*D/nu (White 2006, Ch. 6). Algebraically the same role as
    the 2-D project's inlet_velocity(); D replaces Dh."""
    return Re * NU / D


def poiseuille_profile(r: np.ndarray, Re: float) -> np.ndarray:
    """
    Exact fully-developed Hagen-Poiseuille velocity profile for laminar
    flow in a circular pipe,
        u(r) = 2 * U_mean * (1 - (r/R)^2),
    the closed-form solution of the incompressible Navier-Stokes equations
    for steady, fully-developed, axisymmetric flow in a circular pipe.
    Derived originally by Hagen (1839) and Poiseuille (1840, 1846) from
    experiment; the modern derivation from the Navier-Stokes equations is
    standard, e.g. White, F.M. (2006), Viscous Fluid Flow, 3rd ed.,
    McGraw-Hill, Sec. 3-4 (eq. 3-44); Schlichting, H. & Gersten, K.
    (2017), Boundary-Layer Theory, 9th ed., Springer, Ch. 5. This is the
    circular-pipe counterpart of the 2-D project's plane-Poiseuille
    profile (1.5, not 2, is the peak/mean ratio there -- the different
    prefactor, 2 vs 1.5, is itself a direct consequence of the different
    cross-sectional shape and is one of the two headline numbers this
    project's evaluation checks against). NOT a fit -- exact algebraic PDE
    solution, used as ground truth far downstream, same role as in the
    2-D project.
    """
    U_mean = inlet_velocity(Re)
    return 2.0 * U_mean * (1.0 - (r / R) ** 2)


def poiseuille_dpdx(Re: float) -> float:
    """
    Exact fully-developed (kinematic) axial pressure gradient for
    Hagen-Poiseuille pipe flow:

        dp/dx = -8 * nu * U_mean / R^2

    Derivation (not an independent assumption -- follows directly from
    the SAME exact profile and the SAME governing equation already used
    elsewhere in this project): in the fully-developed region u_x=0,
    v=0, so physics.py's x-momentum residual reduces to
        0 = -dp/dx + nu * (1/r) d/dr( r du/dr ).
    Substituting poiseuille_profile()'s exact profile,
    u(r) = 2*U_mean*(1-(r/R)^2), gives (1/r) d/dr(r du/dr) = -8*U_mean/R^2
    identically, so dp/dx = -8*nu*U_mean/R^2. This constant-gradient
    result is the standard one quoted for Hagen-Poiseuille flow, e.g.
    White, F.M. (2006), Viscous Fluid Flow, 3rd ed., McGraw-Hill, eq.
    (3-44) (there in dynamic-pressure form dP/dx=-8*mu*U_mean/R^2;
    dividing by density gives the kinematic-pressure form used here,
    consistent with the kinematic-pressure convention physics.py and
    model.py already use throughout).
    """
    U_mean = inlet_velocity(Re)
    return -8.0 * NU * U_mean / (R ** 2)


def poiseuille_pressure(x: np.ndarray, Re: float, x_ref: float = None,
                          p_ref: float = 0.0) -> np.ndarray:
    """
    Exact fully-developed kinematic pressure, linear in x (since
    poiseuille_dpdx is constant), gauged so p(x_ref) = p_ref:

        p(x) = p_ref + dp/dx * (x - x_ref)

    Default x_ref=L, p_ref=0 matches BOTH (a) the outlet-referenced
    convention train_hybrid.py's own data-loss target uses for pressure
    (P_shifted = P - P[-1], in train_hybrid.py's _build_interpolators)
    and (b) the network's own loss_outlet BC term, which enforces p=0 at
    x=L during training -- so comparing the network's p(x) against this
    formula under the same gauge is an apples-to-apples check, not one
    complicated by an arbitrary pressure datum.

    Only valid in the fully-developed region (x sufficiently past
    Le(Re)) -- exactly the same caveat poiseuille_profile() already
    carries for u, since both come from the same fully-developed
    assumption.
    """
    if x_ref is None:
        x_ref = L
    return p_ref + poiseuille_dpdx(Re) * (np.asarray(x, dtype=float) - x_ref)


def entrance_length(Re: float) -> float:
    """
    Approximate hydrodynamic entrance (development) length for laminar
    flow in a circular pipe with a uniform inlet profile,
        Le = K * D * Re.

    Functional form (linear in Re, scaling with pipe diameter) is the
    classical result of Langhaar, H.L. (1942), "Steady flow in the
    transition length of a straight tube," Journal of Applied Mechanics,
    9(2), A55-A58, https://doi.org/10.1115/1.4009183 -- solved the
    linearized Navier-Stokes equations for the developing pipe-flow
    problem and obtained a Bessel-function family of profiles whose
    development length scales linearly with Re. K=0.0575 is Langhaar's
    own commonly quoted value (widely reproduced, e.g. Fox, McDonald &
    Pritchard, Introduction to Fluid Mechanics; a rounded K=0.06 is also
    common in textbooks -- see e.g. the Le/D ~ 0.06*Re summary in
    fluid-mechanics course notes and handbooks).

    CAVEAT (same honesty note as the 2-D project's config.py): the
    literature does not agree on a single K any more for pipes than it
    did for channels -- the value depends on the convergence criterion
    used (99% vs 99.9% of centerline velocity) and which approximation of
    the governing equations was solved (Langhaar's own linearization vs.
    Hornbeck's (1964) later finite-difference solution of the full
    parabolized equations, which found "significant differences...for
    both velocity profiles and development lengths" relative to earlier
    linearized theories including Langhaar's). K=0.06 is used here
    because -- exactly as in the 2-D project -- it is close to the value
    this project's OWN reference solver (entrance_solver.py, Hornbeck
    1964 method, 1%-of-centerline-velocity criterion) produces
    self-consistently across Re=50..600; see entrance_solver.py's own
    self-check. Langhaar (1942) is cited for the functional FORM (linear
    scaling with Re) and Hornbeck (1964) for the specific finite-
    difference method this project's own solver reproduces, not for a
    single "correct" numeric K.
    """
    K = 0.06
    return K * D * Re


if __name__ == "__main__":
    print(f"R = {R:.4f} m, D = {D:.4f} m, domain = [{X_MIN},{X_MAX}] x [{R_MIN},{R_MAX}] m (x, r)")
    for Re in RE_TRAIN + RE_TEST_INTERPOLATION:
        U0 = inlet_velocity(Re)
        Le = entrance_length(Re)
        print(f"Re={Re:6.1f}  U0={U0:8.5f} m/s  Le={Le:6.4f} m "
              f"({'inside' if Le < L else 'EXCEEDS'} domain, L={L} m)")