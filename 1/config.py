"""
config.py -- single source of truth for the merged channel-flow PINN
project (hybrid data+physics, entrance/developing-region domain).

Geometry, Re sweep, and entrance-length correlation are carried over
UNCHANGED from the hybrid report (Source 2 / channel_pinn_hybrid_report):
that domain is the physically richer one (models the developing entrance
region, not just the fully-developed profile) and its Re sweep
(train on 50/100/200/500, test on 300 interpolation + 600 extrapolation)
is a strictly harder and more informative generalization test than the
physics-only report's Re sweep. See that report Secs. 2-3 for the full
derivation and citation trail (Bodoia & Osterle, 1961; Chen, 1973;
Shah & London, 1978).

Re-conditioning trick kept from the physics-only report (Source 1 /
Channel_PINN_Report, Sec. 7.1): U_ref(Re)=Re*nu/Dh and the standard
dynamic-pressure scale q(Re)=0.5*U_ref(Re)^2 are used for output
normalization in model.py, rather than the raw-U0 (no 0.5 factor)
normalization used in the hybrid report's own model.py (which is not
included in Source 2's file set and must be reconstructed -- see
model.py's docstring). Both Re definitions here are identical
(Re = U*Dh/nu, Dh=4h -- see the Dh fix below), so the two reports'
normalization schemes are compatible and this only changes an internal
constant.

UPDATE: Dh and entrance_length() below were both corrected from the
originally-carried-over versions (Dh was mislabeled, K=0.05 was
self-referential rather than literature-backed) -- see their docstrings.
L=4.0m is now a generous multiple of every Re's Le (max ~1.3m at Re=600)
rather than a tight fit; there was no reason to shrink it, so it is left
unchanged.
"""

import numpy as np

# ---------------------------------------------------------------- geometry
h = 0.05           # channel half-height [m] (wall-to-centerline distance)
Dh = 4.0 * h        # hydraulic diameter for parallel plates [m].
                    # FIX (see report Sec. 2): Dh = 2 x (full plate gap)
                    # = 2*(2h) = 4h is the standard definition (Dh = 4A/P,
                    # which reduces to 2x the spacing in the wide-duct
                    # limit -- see e.g. Shah & London (1978), Laminar Flow
                    # Forced Convection in Ducts, Academic Press, Ch. 1).
                    # The previous Dh = 2h was actually just the full
                    # channel gap, not the hydraulic diameter -- it made
                    # every Re label in this project non-standard, and
                    # silently broke any comparison against a literature
                    # correlation (like entrance_length() below) that is
                    # itself defined w.r.t. the standard Dh.
                    # CONSEQUENCE: this changes U0(Re) = Re*nu/Dh for every
                    # Re (it is now half of what it was). You MUST rerun
                    # generate_data.py and retrain from scratch after this
                    # change -- the old reference_data.npz and checkpoint
                    # are no longer consistent with this Dh.
L = 4.0             # streamwise domain length [m] -- sized so the Re=600
                    # extrapolation case's development length (~3.0 m,
                    # Le ~= 0.05*Re*Dh) fits comfortably inside the domain
X_MIN, X_MAX = 0.0, L
Y_MIN, Y_MAX = -h, h

# ---------------------------------------------------------------- fluid
NU = 1.0e-4         # kinematic viscosity [m^2/s]

# ---------------------------------------------------------------- Re sweep
RE_TRAIN = [50.0, 100.0, 200.0, 500.0]
RE_TEST_INTERPOLATION = [300.0, 600.0]
ALL_RE_FOR_EVAL = sorted(set(RE_TRAIN + RE_TEST_INTERPOLATION))


def inlet_velocity(Re: float) -> float:
    """U0 = Re * nu / Dh -- mean (=inlet, uniform-profile) velocity for a
    given Re, using the hydraulic-diameter Reynolds-number definition.
    Kept as a separate, explicitly named function (distinct from
    model.u_ref, which is algebraically identical) because it is also
    used to set up the entrance-flow BOUNDARY CONDITION (a physical input
    to the problem) wherever that's a clearer name than "reference
    velocity" (a normalization device) -- see entrance_solver.py and
    model.py's docstrings for the distinction."""
    return Re * NU / Dh


def poiseuille_profile(y: np.ndarray, Re: float) -> np.ndarray:
    """
    Exact fully-developed plane-Poiseuille velocity profile,
        u(y) = 1.5 * U_mean * (1 - (y/h)^2),
    the closed-form solution of the incompressible Navier-Stokes equations
    for steady, fully-developed flow between infinite parallel plates.
    Source: White, F.M. (2006), Viscous Fluid Flow, 3rd ed., McGraw-Hill,
    Sec. 3-5; Schlichting, H. & Gersten, K. (2017), Boundary-Layer Theory,
    9th ed., Springer, Ch. 5. This is an exact algebraic PDE solution, not
    a fit -- the strongest available ground truth far downstream, used in
    both source reports' evaluation.
    """
    U_mean = inlet_velocity(Re)
    return 1.5 * U_mean * (1.0 - (y / h) ** 2)


def poiseuille_v(y: np.ndarray, Re: float) -> np.ndarray:
    """
    Exact fully-developed wall-normal velocity, v(y) = 0.

    For steady, incompressible, fully-developed channel flow the
    streamwise velocity u is independent of x by definition of "fully
    developed" (White 2006, Sec. 3-5; Schlichting & Gersten 2017, Ch. 5).
    Continuity (u_x + v_y = 0) then gives v_y = 0 everywhere, and v = 0
    at both walls (no-slip/no-penetration), so v(y) = 0 identically
    across the section. This is the same exact closed-form solution used
    for poiseuille_profile(); it is included as its own function (rather
    than assumed implicitly) so evaluation code has an explicit, citable
    ground truth for the v-component in the fully-developed region.
    """
    return np.zeros_like(np.asarray(y, dtype=float))


def poiseuille_pressure(x: np.ndarray, Re: float) -> np.ndarray:
    """
    Exact fully-developed kinematic pressure p(x), referenced so that
    p(x=L) = 0 -- the SAME outlet gauge convention used by the PINN's own
    outlet boundary condition (train_hybrid.sample_outlet / loss_outlet)
    and by the Bodoia-Osterle reference shift applied in
    train_hybrid._build_interpolators() (P_shifted = P - P[-1]).

    Derivation (exact NS solution for fully-developed plane Poiseuille
    flow -- White (2006), Viscous Fluid Flow, 3rd ed., Sec. 3-5;
    Schlichting & Gersten (2017), Ch. 5): with v=0 and u=u(y) only, the
    x-momentum equation reduces to
        0 = -dp/dx + nu * d2u/dy2
    and since u(y) = 1.5*U_mean*(1-(y/h)^2) (poiseuille_profile above),
        d2u/dy2 = -3*U_mean/h^2  (constant),
    so dp/dx = -3*nu*U_mean/h^2 is CONSTANT (a known exact result: the
    pressure gradient balances viscous wall shear exactly in fully-
    developed flow). Integrating from the outlet (p(L)=0):
        p(x) = 3*nu*U_mean/h^2 * (L - x).
    This is a function of Re alone (via U_mean=inlet_velocity(Re)) and
    the fixed geometry -- never a solved field -- so, as with
    entrance_length()/poiseuille_profile(), using it as ground truth is
    not solution leakage.
    """
    U_mean = inlet_velocity(Re)
    dpdx = -3.0 * NU * U_mean / h ** 2
    return dpdx * (np.asarray(x, dtype=float) - L)


def entrance_length(Re: float) -> float:
    """
    Hydrodynamic entrance (development) length for laminar flow between
    parallel plates with a uniform inlet profile, using the closed-form
    correlation of:

        Chen, R.-Y. (1973), "Flow in the Entrance Region at Low Reynolds
        Numbers," ASME J. Fluids Eng., 95(1), pp. 153-158.
        DOI: 10.1115/1.3446948

            Le / Dh = 0.315 / (1 + 0.0175*Re) + 0.011*Re

    This is an integral-momentum (boundary-layer) solution of the
    parallel-plate entrance problem, fit to a two-term correlation valid
    from creeping flow up through Re ~ 2000. It is the correlation Shah &
    London (1978, Laminar Flow Forced Convection in Ducts, Academic Press,
    Ch. 1) adopt as the standard reference for this exact geometry, and it
    is still the one cited/reused in current entrance-length studies (e.g.
    Ray & Das (2023), Micromachines, 14(7):1418; Barber & Emerson (2000)).

    REPLACES the previous self-referential K=0.05*Dh*Re rule, which was
    tuned to match this project's own solver rather than an independent
    literature correlation -- not something we can defend as "standard."

    NOTE: this correlation's Re and Dh must both be the standard ones
    (Dh = 2 x full plate gap); see the Dh fix in the geometry block above.
    Both quantities here are functions of Re alone (never of a solved
    field), so using them to set the entrance-region evaluation stations
    is not solution leakage -- same argument as model.py's Sec. 7.2 note
    on output rescaling.
    """
    return Dh * (0.315 / (1.0 + 0.0175 * Re) + 0.011 * Re)


if __name__ == "__main__":
    print(f"Dh = {Dh:.4f} m,  domain = [{X_MIN},{X_MAX}] x [{Y_MIN},{Y_MAX}] m")
    for Re in RE_TRAIN + RE_TEST_INTERPOLATION:
        U0 = inlet_velocity(Re)
        Le = entrance_length(Re)
        print(f"Re={Re:6.1f}  U0={U0:8.5f} m/s  Le={Le:6.4f} m "
              f"({'inside' if Le < L else 'EXCEEDS'} domain, L={L} m)")