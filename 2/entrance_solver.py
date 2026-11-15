"""
entrance_solver.py -- Reference-data generator for developing (entrance-
region) axisymmetric circular-pipe flow, implementing the finite-
difference marching method of:

    Hornbeck, R.W. (1964), "Laminar Flow in the Entrance Region of a
    Pipe," Applied Scientific Research, Section A, 13, 224-232.
    DOI: 10.1007/BF00382049

This is the direct circular-pipe counterpart of the 2-D project's
entrance_solver.py, which implements Bodoia & Osterle (1961) for the
plane-channel geometry. Hornbeck (1964) is explicitly the same style of
method applied to the axisymmetric case: it cites Bodoia & Osterle (1961)
itself as prior work, uses the same parabolized-marching strategy, and
the same trick of solving the pressure-gradient-dependent problem twice
(zero-G particular solution + unit-G homogeneous solution) and superposing
to satisfy conservation of mass at each station -- see Hornbeck (1964)
Sec. 2, eqs. (4)-(10) (also reproduced/confirmed against a scan of the
original paper). Only the coordinate system and the finite-difference
molecule for the viscous term change (Cartesian y -> cylindrical r);
the marching/superposition/mass-conservation logic is unchanged.

GOVERNING EQUATIONS (parabolized/boundary-layer approximation, steady,
axisymmetric, no swirl, kinematic pressure) -- Hornbeck (1964) eqs. (1)-(3):

    u du/dx + v du/dr = -(1/rho) dP/dx + nu * (1/r) d/dr( r du/dr )   (axial momentum)
    du/dx + (1/r) d(r v)/dr = 0                                       (continuity)
    2*pi * integral_0^R (u r) dr = Q  (=const, mass flow rate)        (global mass conservation)

As in the 2-D project, the streamwise diffusion term d^2u/dx^2 is dropped
relative to streamwise convection -- valid because the flow does not
recirculate and x-gradients are much smaller than r-gradients except very
close to x=0 (same justification given in Bodoia & Osterle 1961 and
repeated in Hornbeck 1964).

MARCHING PROCEDURE, station i -> i+1 (this project's implementation,
following Hornbeck 1964 Sec. 2):
  1. Linearize the convective coefficient using the previous station's
     u_i(r) (standard parabolic-marching technique, identical to the 2-D
     solver).
  2. Discretize the viscous term (1/r) d/dr(r du/dr) with a
     conservative (control-volume-consistent) central-difference molecule
     in r -- see `_diffusion_coeffs` below -- giving a tridiagonal system
     in u_{i+1}(r) for a FIXED pressure gradient dP/dx.
       - AXIS TREATMENT (r=0): (1/r) d/dr(r du/dr) is 0/0 at r=0. Its
         well-defined limit as r->0 is 2 * d^2u/dr^2 (L'Hopital, using
         symmetry du/dr=0 at the axis) -- this is the standard axis
         treatment for axisymmetric diffusion problems (the same limit
         used, e.g., for axisymmetric heat conduction finite-difference
         schemes; see Patankar, S.V. (1980), Numerical Heat Transfer and
         Fluid Flow, Hemisphere, Sec. 4.2-3 for the general control-volume
         treatment of the r=0 singularity in cylindrical coordinates).
         The axis is therefore an INTERIOR (Neumann/symmetry) node of the
         tridiagonal system, not a Dirichlet boundary -- unlike the 2-D
         channel case, which has two symmetric Dirichlet walls and no
         such singular point.
       - WALL (r=R): no-slip, Dirichlet, u=0 -- as in the 2-D solver.
  3. Solved twice per station (dP/dx=0 particular solution u_p(r);
     dP/dx=1 homogeneous solution u_h(r) with the -1 RHS this induces),
     then superposed: u_{i+1} = u_p + G*u_h, with G (=dP/dx at this
     station) fixed by requiring 2*pi*integral(u_{i+1} r dr) = Q
     (Hornbeck 1964's mass-conservation constraint, eq. (10) there,
     applied via Simpson/trapezoidal quadrature here).
  4. v_{i+1}(r) is recovered from the continuity equation,
     d(rv)/dr = -r du/dx, by integrating from the axis (rv=0 there) using
     the trapezoidal rule -- the axisymmetric analogue of the 2-D
     solver's from-the-wall integration of dv/dy=-du/dx.

This solver is used only to generate reference (x, r, Re) -> (u, v, p)
data for supervised PINN training/testing -- an independent
re-implementation of a named, peer-reviewed numerical method, not an ad
hoc CFD run (same role and same caveat as the 2-D project's solver).
"""

import numpy as np
import config as cfg


def _thomas(a, b, c, d):
    """Standard tridiagonal (Thomas) solve; identical routine to the 2-D
    solver's, reused unchanged -- Thomas, L.H. (1949), unpublished;
    described in Patankar (1980), Sec. 3.3."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def solve(Re: float, nx: int = 400, nr: int = 81, x_max: float = None):
    """
    Returns dict with:
      x  : (nx,) streamwise stations
      r  : (nr,) radial grid, r[0]=0 (axis) .. r[-1]=R (wall)
      U  : (nx, nr) axial velocity
      V  : (nx, nr) radial velocity
      P  : (nx,) centerline-referenced pressure (kinematic), integrated from dP/dx
    """
    if x_max is None:
        x_max = cfg.L
    R = cfg.R
    nu = cfg.NU
    U0 = cfg.inlet_velocity(Re)
    Q = np.pi * R ** 2 * U0  # constant volumetric flow rate (must be conserved), pi R^2 * U_mean

    r = np.linspace(0.0, R, nr)
    dr = r[1] - r[0]
    x = np.linspace(0.0, x_max, nx)
    dx = x[1] - x[0]

    U = np.zeros((nx, nr))
    V = np.zeros((nx, nr))
    P = np.zeros(nx)

    # Inlet: uniform profile (same convention as the 2-D solver)
    U[0, :] = U0
    U[0, -1] = 0.0  # no-slip at the wall (r=R); axis (r=0) has no BC of its own, it's a symmetry point

    n_unk = nr - 1  # unknowns: axis (j=0) through last interior node (j=nr-2); wall (j=nr-1) is Dirichlet=0

    for i in range(nx - 1):
        u_prev = U[i, :]

        a = np.zeros(n_unk)   # sub-diagonal
        b = np.zeros(n_unk)   # diagonal
        c = np.zeros(n_unk)   # super-diagonal
        d0 = np.zeros(n_unk)  # RHS for G=0 (particular solution)
        d1 = np.zeros(n_unk)  # RHS contribution from G (coefficient of -G is -1 for every row)

        # --- axis node (j=0): limit (1/r) d/dr(r du/dr) -> 2 d^2u/dr^2 ---
        # Discretized as 2*(u_1 - u_0)/dr^2 *after* mirroring u_{-1}=u_1
        # (symmetry): 2*(u_1 - 2u_0 + u_1)/dr^2 = 4*(u_1 - u_0)/dr^2.
        diff0 = 4.0 * nu / dr ** 2
        b[0] = u_prev[0] / dx + diff0
        c[0] = -diff0
        d0[0] = u_prev[0] ** 2 / dx
        d1[0] = -1.0

        # --- interior nodes (j=1 .. nr-2): conservative cylindrical FD ---
        for k in range(1, n_unk):
            j = k  # radial index, 1..nr-2
            rj = r[j]
            r_half_p = rj + 0.5 * dr   # r_{j+1/2}
            r_half_m = rj - 0.5 * dr   # r_{j-1/2}
            coef_p = nu * r_half_p / (rj * dr ** 2)
            coef_m = nu * r_half_m / (rj * dr ** 2)
            b[k] = u_prev[j] / dx + coef_p + coef_m
            a[k] = -coef_m
            if j < nr - 2:
                c[k] = -coef_p
            # else: j == nr-2, neighbor is the wall (u=0, Dirichlet) --
            # its contribution is exactly zero and simply doesn't appear.
            d0[k] = u_prev[j] ** 2 / dx
            d1[k] = -1.0

        u_p_int = _thomas(a.copy(), b.copy(), c.copy(), d0.copy())
        u_h_int = _thomas(a.copy(), b.copy(), c.copy(), d1.copy())

        u_p = np.zeros(nr)
        u_h = np.zeros(nr)
        u_p[:-1] = u_p_int
        u_h[:-1] = u_h_int
        # wall already zero by construction (Dirichlet, not part of the system)

        # Enforce mass conservation: 2*pi*integral(u * r) dr = Q
        int_up = np.trapezoid(u_p * r, r)
        int_uh = np.trapezoid(u_h * r, r)
        G = (Q / (2.0 * np.pi) - int_up) / int_uh

        u_new = u_p + G * u_h
        U[i + 1, :] = u_new
        P[i + 1] = P[i] - G * dx  # same sign convention as the 2-D solver

        # Continuity: d(rv)/dr = -r du/dx  =>  r*v(r) = -integral_0^r (r' du/dx) dr'
        dudx = (u_new - u_prev) / dx
        rv = np.zeros(nr)
        integrand = r * dudx
        for j in range(1, nr):
            rv[j] = rv[j - 1] - 0.5 * (integrand[j] + integrand[j - 1]) * dr
        v = np.zeros(nr)
        v[1:] = rv[1:] / r[1:]
        v[0] = 0.0  # v=0 on the axis by symmetry (limit of rv/r as r->0)
        V[i + 1, :] = v

    return dict(x=x, r=r, U=U, V=V, P=P, Re=Re, U0=U0)


if __name__ == "__main__":
    # Self-check against the two independent literature benchmarks used
    # elsewhere in this project (Hagen-Poiseuille exact centerline ratio
    # u_center/U_mean = 2.0 for a circular pipe, vs. 1.5 for the 2-D
    # channel; Langhaar 1942 / Hornbeck 1964 entrance-length scaling) --
    # NOT the training data itself, just a sanity check on the solver,
    # mirroring the 2-D solver's own __main__ self-check.
    for Re in [50, 100, 200, 500, 600]:
        sol = solve(Re)
        u_exact_center = 2.0 * sol["U0"]
        u_final_center = sol["U"][-1, 0]
        ratio = u_final_center / u_exact_center
        Le_corr = cfg.entrance_length(Re)
        centerline = sol["U"][:, 0]
        within = np.abs(centerline - u_exact_center) <= 0.01 * u_exact_center
        idx = np.argmax(within)
        Le_solver = sol["x"][idx] if within[idx] else np.nan
        # global mass-conservation check at outlet
        Q_check = 2 * np.pi * np.trapezoid(sol["U"][-1, :] * sol["r"], sol["r"])
        Q_exact = np.pi * cfg.R ** 2 * sol["U0"]
        print(f"Re={Re:4.0f}  outlet-centerline/exact={ratio:.4f}  "
              f"Le_solver(1%)={Le_solver:.4f} m  Le_correlation={Le_corr:.4f} m  "
              f"mass_flux_err={100*abs(Q_check-Q_exact)/Q_exact:.3f}%")