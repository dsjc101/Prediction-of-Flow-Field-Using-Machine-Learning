"""
entrance_solver.py -- Reference-data generator for developing (entrance-region)
plane channel flow, implementing the finite-difference marching method of:

    Bodoia, J.R. & Osterle, J.F. (1961), "Finite Difference Analysis of Plane
    Poiseuille and Couette Flow Developments," Applied Scientific Research,
    Section A, 10, 265-276. DOI: 10.1007/BF00411919

Method (as described in the paper and in subsequent citing works, e.g. the
finite-difference/parabolic-marching treatment in Patankar, S.V. & Spalding,
D.B. (1970), Heat and Mass Transfer in Boundary Layers, 2nd ed., Intertext,
and reproduced in the AIP Advances 1, 041302 (2011) discussion of the
Bodoia-Osterle technique):

Governing equations are the parabolized (boundary-layer-type) steady,
incompressible Navier-Stokes equations -- the streamwise diffusion term
d^2u/dx^2 is neglected relative to streamwise convection, which is the
standard approximation for duct-entrance problems (valid because the flow
is not recirculating and x-derivatives are much smaller than y-derivatives
except very close to x=0):

    u du/dx + v du/dy = -(1/rho) dP/dx + nu d^2u/dy^2      (streamwise momentum)
    du/dx + dv/dy = 0                                       (continuity)

Marching procedure, station i -> i+1 (this project's implementation of the
Bodoia-Osterle technique):
  1. Linearize the convective coefficient using the previous station's u_i(y)
     (standard technique for parabolic marching problems; the resulting
     system is then linear in the unknown u_{i+1}(y) for a FIXED pressure
     gradient dP/dx).
  2. The finite-difference form of the momentum equation is a tridiagonal
     system in y, solved via the Thomas algorithm (Thomas, L.H. (1949),
     unpublished; the algorithm is a standard tool for parabolic marching
     PDEs, described in essentially every numerical-PDE text, e.g. Patankar
     1980, Numerical Heat Transfer and Fluid Flow, Hemisphere, Sec. 3.3).
  3. Because the system is linear in the (unknown, constant-across-y at each
     station) pressure gradient dP/dx, it is solved twice -- once with
     dP/dx=0 to get a particular solution u_p(y), once with dP/dx=1 to get a
     homogeneous solution u_h(y) -- then superposed: u_{i+1} = u_p + G*u_h.
     G (=dP/dx at this station) is fixed by requiring the integral of
     u_{i+1} over y equal the known, constant mass flow rate (2h*U_mean) --
     this is exactly the constraint Bodoia & Osterle enforce ("adjusting the
     pressure gradient to ensure V**=1" in their nondimensional form).
  4. v_{i+1}(y) is recovered from the continuity equation by integrating
     from the wall (v=0) using the trapezoidal rule.

This solver is used only to generate reference (x,y,Re) -> (u,v,p) data for
supervised PINN training/testing in this project -- it is an independent
re-implementation of a named, peer-reviewed numerical method, not an ad hoc
CFD run.
"""

import numpy as np
import config as cfg


def solve(Re: float, nx: int = 400, ny: int = 81, x_max: float = None):
    """
    Returns dict with:
      x  : (nx,) streamwise stations
      y  : (ny,) wall-normal grid
      U  : (nx, ny) streamwise velocity
      V  : (nx, ny) wall-normal velocity
      P  : (nx,) centerline-referenced pressure (kinematic), integrated from dP/dx
    """
    if x_max is None:
        x_max = cfg.L
    h = cfg.h
    nu = cfg.NU
    U0 = cfg.inlet_velocity(Re)
    Q = 2.0 * h * U0  # constant mass flow rate per unit depth (must be conserved)

    y = np.linspace(-h, h, ny)
    dy = y[1] - y[0]
    x = np.linspace(0.0, x_max, nx)
    dx = x[1] - x[0]

    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    P = np.zeros(nx)

    # Inlet: uniform profile (Section 3.1 of the project report)
    U[0, :] = U0
    U[0, 0] = 0.0
    U[0, -1] = 0.0  # no-slip at the two walls

    for i in range(nx - 1):
        u_prev = U[i, :]

        # Build tridiagonal system for the linearized momentum equation:
        #   u_prev * (u_new - u_prev)/dx = -G + nu * d2u_new/dy2
        # => (u_prev/dx + 2*nu/dy^2) u_new_j - (nu/dy^2) u_new_{j-1} - (nu/dy^2) u_new_{j+1}
        #      = u_prev_j^2/dx - G
        n_int = ny - 2  # interior points (walls are Dirichlet u=0)
        a = np.zeros(n_int)   # sub-diagonal
        b = np.zeros(n_int)   # diagonal
        c = np.zeros(n_int)   # super-diagonal
        d0 = np.zeros(n_int)  # RHS for G=0 (particular solution)
        d1 = np.zeros(n_int)  # RHS contribution from G (coefficient of -G is -1 for all j)

        r = nu / dy ** 2
        for k in range(n_int):
            j = k + 1  # actual y-index (1..ny-2)
            b[k] = u_prev[j] / dx + 2.0 * r
            a[k] = -r
            c[k] = -r
            d0[k] = u_prev[j] ** 2 / dx

        def thomas(a, b, c, d):
            n = len(d)
            cp = np.zeros(n)
            dp = np.zeros(n)
            cp[0] = c[0] / b[0]
            dp[0] = d[0] / b[0]
            for k in range(1, n):
                m = b[k] - a[k] * cp[k - 1]
                cp[k] = c[k] / m
                dp[k] = (d[k] - a[k] * dp[k - 1]) / m
            xsol = np.zeros(n)
            xsol[-1] = dp[-1]
            for k in range(n - 2, -1, -1):
                xsol[k] = dp[k] - cp[k] * xsol[k + 1]
            return xsol

        # Particular solution (G=0)
        u_p_int = thomas(a.copy(), b.copy(), c.copy(), d0.copy())
        # Homogeneous solution contribution from G=1 (RHS = -1 for all interior rows)
        d1[:] = -1.0
        u_h_int = thomas(a.copy(), b.copy(), c.copy(), d1.copy())

        u_p = np.zeros(ny)
        u_h = np.zeros(ny)
        u_p[1:-1] = u_p_int
        u_h[1:-1] = u_h_int
        # walls already zero by construction (Dirichlet, not part of interior system)

        # Enforce mass conservation: integral(u_p + G*u_h) dy = Q
        int_up = np.trapezoid(u_p, y)
        int_uh = np.trapezoid(u_h, y)
        G = (Q - int_up) / int_uh

        u_new = u_p + G * u_h
        U[i + 1, :] = u_new
        # SIGN FIX: the momentum equation above is built as
        #   u_prev*(u_new-u_prev)/dx = -G + nu*d2u/dy2
        # i.e. matched term-by-term against the standard form
        #   u du/dx = -(1/rho) dP/dx + nu d2u/dy2
        # so -(1/rho)dP/dx = -G  =>  dP/dx (kinematic) = +G, not -G. The
        # previous line (`P[i+1] = P[i] - G*dx`) integrated with the
        # opposite sign, which produced a pressure that INCREASED
        # downstream (verified numerically: far-field dP/dx = +0.0120 for
        # Re=200 vs the exact fully-developed value dP/dx =
        # -3*nu*U_mean/h^2 = -0.0120, config.poiseuille_pressure -- same
        # magnitude, flipped sign). Physically the pressure must DECREASE
        # downstream (a favorable gradient driving the flow against
        # viscous friction; White (2006), Viscous Fluid Flow, Sec. 3-5),
        # which is what integrating with +G*dx now gives (confirmed to
        # match the exact asymptote far downstream).
        P[i + 1] = P[i] + G * dx

        # Continuity: dv/dy = -du/dx  => integrate from wall (v=0) upward
        dudx = (u_new - u_prev) / dx
        v = np.zeros(ny)
        for j in range(1, ny):
            v[j] = v[j - 1] - 0.5 * (dudx[j] + dudx[j - 1]) * dy
        V[i + 1, :] = v

    return dict(x=x, y=y, U=U, V=V, P=P, Re=Re, U0=U0)


if __name__ == "__main__":
    # Self-check against the two independent literature benchmarks already
    # used elsewhere in this project (White 2006/Schlichting&Gersten exact
    # Poiseuille profile; Chen 1973/Shah&London 1978 entrance length) --
    # NOT the training data itself, just a sanity check on the solver.
    for Re in [50, 100, 200, 500]:
        sol = solve(Re)
        u_exact_center = 1.5 * sol["U0"]
        u_final_center = sol["U"][-1, len(sol["y"]) // 2]
        ratio = u_final_center / u_exact_center
        Le_corr = cfg.entrance_length(Re)
        # find first x where centerline is within 1% of exact (solver's own Le)
        centerline = sol["U"][:, len(sol["y"]) // 2]
        within = np.abs(centerline - u_exact_center) <= 0.01 * u_exact_center
        idx = np.argmax(within)
        Le_solver = sol["x"][idx] if within[idx] else np.nan
        print(f"Re={Re:4.0f}  outlet-centerline/exact={ratio:.4f}  "
              f"Le_solver(1%)={Le_solver:.4f} m  Le_correlation={Le_corr:.4f} m")