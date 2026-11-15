"""
analytical_duct.py
===================
Exact, closed-form (Fourier series) fully-developed laminar velocity field
for a rectangular duct -- the rectangular-duct analogue of the exact
Hagen-Poiseuille solution used as the headline benchmark in the pipe
report and the exact plane-Poiseuille solution used in the channel report.

SOURCE / DERIVATION
--------------------
Fully-developed duct flow satisfies the 2-D Poisson equation
    nu * (u_yy + u_zz) = (1/rho) dP/dx = const
with u=0 on all four walls -- a classical linear-PDE boundary-value
problem solved by separation of variables / Fourier series. This is
standard material reproduced in every modern viscous-flow / duct-flow
text, in particular:

  Shah, R.K. & London, A.L. (1978), "Laminar Flow Forced Convection in
  Ducts," Academic Press, Ch. 7 (rectangular ducts) -- gives the series
  solution and tabulates f*Re and u_max/u_mean for a range of aspect
  ratios (AR=1,2,4,8,...), which is why AR=2 was chosen in config.py: the
  result computed here is directly checkable against that table.

  White, F.M. (2006), Viscous Fluid Flow, 3rd ed., McGraw-Hill, Sec. 3-4 --
  reproduces the same series solution for the rectangular-duct case
  alongside the circular-pipe (Hagen-Poiseuille) and plane-channel results
  already used in the earlier two projects, so all three geometries in
  this project sequence draw the fully-developed benchmark from the same
  source family.

FORMULA (origin at duct corner, y in [0,H], z in [0,W], following the
standard corner-origin convention used for this BC set; some texts instead
use a center origin y in [-H/2,H/2] -- both are equivalent up to a shift,
corner-origin is used here to match the marching solver's grid in
duct_solver.py directly, avoiding an extra coordinate transform):

  u(y,z) = (16 G a^2 / (pi^3 mu)) * SUM_{n=1,3,5,...}
             [ (-1)^((n-1)/2) / n^3 ]
             * [ 1 - cosh(n*pi*(z - W/2) / (2a)) / cosh(n*pi*W/(4a)) ]
             * cos(n*pi*(y - H/2) / (2a))

  where a = H/2 (half-height), G = -dP/dx > 0.

For this implementation, rather than reconstruct the closed-form
coefficient-in-terms-of-G algebra symbolically (error-prone to transcribe
by hand and easy to mis-transcribe a n^3 vs n^2 or a missing factor), the
series is solved the robust way: G is treated as an unknown scaling
constant fixed by the SAME mass-conservation constraint used everywhere
else in this project (Int Int u dy dz = Q = U_mean*H*W), applied to the
computed (still-unnormalized, G=1) series shape. This sidesteps
transcription risk in the closed-form prefactor while using exactly the
textbook series *shape* function, and is mathematically equivalent to
using the textbook closed-form value of G directly (same PDE, same BCs,
same solution -- just solved for the free constant via integration
instead of substitution).
"""

import numpy as np

# NumPy >=2.0 renamed trapz -> trapezoid; support both installed versions.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _series_shape(y, z, H, W, n_terms=99):
    """
    Unnormalized series shape S(y,z) solving nu*(S_yy+S_zz) = -1 (i.e. G=1,
    mu=1) with S=0 on all four walls y in {0,H}, z in {0,W}, corner origin.
    Returns S on a (len(y), len(z)) grid. Uses the sine/cosh separated-
    variable solution appropriate to Dirichlet BCs on all 4 sides:

      S(y,z) = SUM_{n=1}^N b_n * sin(n*pi*y/H) *
               [1 - cosh(n*pi*(z-W/2)/H) / cosh(n*pi*W/(2H))]

    with b_n from expanding the RHS forcing (=1/nu, here nu absorbed into
    the caller) in the sine basis: b_n = (4/(n*pi)) for n odd, 0 for n
    even (standard Fourier sine expansion of a constant on [0,H]), scaled
    by H^2/(n*pi)^3-type factor from solving the resulting 1-D ODE in z
    for each mode -- assembled numerically below term-by-term rather than
    quoted as a single closed-form line, for the same transcription-safety
    reason given in the module docstring.
    """
    Y, Z = np.meshgrid(y, z, indexing="ij")
    S = np.zeros_like(Y)
    for n in range(1, 2 * n_terms, 2):  # odd n only
        lam = n * np.pi / H
        bn = 4.0 / (n * np.pi)  # Fourier sine coeff of constant-1 forcing on [0,H]
        term = (bn / lam ** 2) * np.sin(lam * Y) * (
            1.0 - np.cosh(lam * (Z - W / 2.0)) / np.cosh(lam * W / 2.0)
        )
        S += term
    return S


def fully_developed_profile(y, z, H, W, U_mean, n_terms=99):
    """
    Exact fully-developed velocity field u(y,z), scaled so that
    Int Int u dy dz = U_mean * H * W (mass-conservation normalization,
    see module docstring for why G is fixed this way rather than by
    substituting a hand-transcribed closed-form prefactor).
    """
    S = _series_shape(y, z, H, W, n_terms=n_terms)
    Q_shape = _trapz(_trapz(S, z, axis=1), y)
    scale = (U_mean * H * W) / Q_shape
    return scale * S


def umax_over_umean(H, W, n_terms=99, ny=201, nz=201):
    """Exact ratio of centerline (max) velocity to mean velocity for this
    aspect ratio -- computed from the series itself (see self-consistency
    check in duct_solver.py), not quoted from a memorized table value."""
    y = np.linspace(0, H, ny)
    z = np.linspace(0, W, nz)
    U = fully_developed_profile(y, z, H, W, U_mean=1.0, n_terms=n_terms)
    return U.max() / 1.0  # U_mean=1 by construction of the normalization


def exact_cross_velocities(y, z):
    """
    v(y,z) = w(y,z) = 0 EXACTLY in fully-developed duct flow. This is not
    a separate citable result: it follows directly from the
    fully-developed assumption d/dx=0 applied to continuity (u_x+v_y+w_z=0
    -> v_y+w_z=0, independent of x) together with v=w=0 on all 4 no-slip
    walls -- the same elementary boundary-value argument used throughout
    this module (e.g. White (2006) Sec. 3-4, Shah & London (1978) Ch. 7,
    already cited above, both take v=w=0 as given for this flow class).
    Provided so evaluate_global.py has an explicit, zero-valued exact
    baseline for v, w in the fully-developed region, on the same (y,z)
    grid convention as fully_developed_profile.
    """
    Y, Z = np.meshgrid(y, z, indexing="ij")
    return np.zeros_like(Y), np.zeros_like(Y)


def exact_dPdx(H, W, nu, U_mean, n_terms=99, ny=201, nz=201):
    """
    The raw, convention-free fully-developed pressure gradient dP/dx
    (negative -- pressure falls with x), recovered from the SAME mass-
    conservation scaling used in fully_developed_profile: since S solves
    nu*(S_yy+S_zz)=-1 (i.e. G=1 in the earlier unnormalized problem), the
    recovered scale factor IS -dP/dx/nu for the true (U_mean-normalized)
    field. Factored out as its own function (BUG FIX) so every caller
    that needs dP/dx (exact_pressure below, friction_factor_fRe below,
    and evaluate_global.py's fully-developed pressure-slope check) uses
    this SAME value directly, rather than each re-deriving it from a
    friction factor whose definition/convention could differ between
    callers -- exactly the kind of mismatch that previously caused a
    4x-inconsistent dP/dx between this file and a hand-rederivation in
    evaluate_global.py (see that file for the fix on that side).
    """
    y = np.linspace(0, H, ny)
    z = np.linspace(0, W, nz)
    S = _series_shape(y, z, H, W, n_terms=n_terms)
    Q_shape = _trapz(_trapz(S, z, axis=1), y)
    scale = (U_mean * H * W) / Q_shape  # = (-dP/dx)/nu  [since S solved with forcing=1]
    return -scale * nu


def exact_pressure(x, H, W, nu, U_mean, x_ref=0.0, p_ref=0.0, n_terms=99):
    """
    p(x) = p_ref + dP/dx * (x - x_ref), UNIFORM over the cross-section
    (y,z) -- the fully-developed pressure field. Cross-sectional
    uniformity of p is itself part of the fully-developed boundary-layer
    approximation (the same one underlying the p'=P_main(x)+p'(y,z)
    splitting used in duct_solver.py, where p' -> 0 as the flow becomes
    fully developed, see that module's docstring): with v=w=0 everywhere
    and no cross-sectional acceleration, the y- and z-momentum equations
    reduce to p_y=p_z=0, i.e. p=p(x) only. dP/dx here comes directly from
    exact_dPdx (BUG FIX: previously re-derived via a friction-factor
    round trip that introduced a 4x error, see exact_dPdx's docstring and
    friction_factor_fRe's docstring below).
    """
    dPdx = exact_dPdx(H, W, nu, U_mean, n_terms=n_terms)
    x = np.atleast_1d(x)
    return p_ref + dPdx * (x - x_ref)


def friction_factor_fRe(H, W, nu, U_mean, n_terms=99, ny=201, nz=201):
    """
    Darcy (Moody) friction factor f, defined via the wall-shear /
    pressure-drop relation f = 8*tau_w_avg/(rho*U_mean^2) with
    tau_w_avg = (A/P) * (-dP/dx)  (control-volume force balance, standard
    result, see e.g. White (2006) Ch.6). Returns f*Re_Dh (the standard
    reported quantity, since f*Re is what is aspect-ratio-dependent-but-
    Re-independent for laminar duct flow -- the quantity Shah & London
    (1978) tabulate, in the Darcy convention).

    BUG FIX: this previously used f = 2*tau_w_avg/(rho*U_mean^2), which is
    the FANNING friction factor, not the Darcy/Moody one -- a classic,
    easy mix-up (they differ by exactly a factor of 4: f_Darcy =
    4*f_Fanning; equivalently f_Fanning = 8*tau_w/(rho*U^2)/4 =
    2*tau_w/(rho*U^2)). Because f*Re for a circular pipe is the textbook-
    familiar 16 (Fanning) or 64 (Darcy), and this project's own numbers
    were coming out ~4x low (fRe~15.5 instead of ~62 for this AR=2 duct,
    caught by comparing against duct_solver.py's INDEPENDENTLY marched
    dP/dx, which matched the correct value exactly), this was silently
    mislabeling the Fanning value as Darcy throughout -- which matters
    directly for this project's own stated plan (README Sec 10) to check
    this number against Shah & London (1978)'s table, which tabulates
    the Darcy convention: an uncorrected comparison would have looked
    like a large validation failure that was actually just a factor-of-4
    labeling bug.

    -dP/dx is recovered via exact_dPdx (see that function; both this and
    fully_developed_profile's own internal scale computation now draw
    from the exact same mass-conservation-derived source).
    """
    dPdx = exact_dPdx(H, W, nu, U_mean, n_terms=n_terms, ny=ny, nz=nz)
    A = H * W
    P = 2 * (H + W)
    tau_w_avg = (A / P) * abs(dPdx)
    rho = 1.0  # f*Re is rho-independent; rho cancels against nu in Re
    f = 8.0 * tau_w_avg / (rho * U_mean ** 2)  # Darcy/Moody definition (White 2006 Ch.6)
    Dh = 4 * A / P
    Re = U_mean * Dh / nu
    return f * Re


if __name__ == "__main__":
    import config
    print(f"AR={config.AR:.2f}  H={config.H} W={config.W} Dh={config.DH:.4f}")
    print(f"u_max/u_mean (exact, this AR) = {umax_over_umean(config.H, config.W):.4f}")
    Um = config.u_mean_for_Re(200)
    fRe = friction_factor_fRe(config.H, config.W, config.NU, Um)
    print(f"f*Re_Dh (exact, this AR)      = {fRe:.3f}")
    print("Cross-check these two printed numbers against Shah & London "
          "(1978) Ch.7 Table for AR=2 when writing the final report.")