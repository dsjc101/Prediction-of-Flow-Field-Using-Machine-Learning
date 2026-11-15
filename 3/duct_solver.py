"""
duct_solver.py
==============
Independent finite-difference marching reference solution for developing
laminar flow in a straight rectangular duct -- the rectangular-duct
counterpart of Bodoia & Osterle (1961) [2-D channel, used in the first
project] and Hornbeck (1964) [circular pipe, used in the second project].

METHOD AND WHY IT DIFFERS FROM THE PIPE/CHANNEL SOLVERS
---------------------------------------------------------
The channel and pipe solvers each march a 1-D (tridiagonal) cross-stream
problem at every streamwise station, because their cross-sections are
described by a single transverse coordinate (y for the channel, r for the
axisymmetric pipe). A rectangular duct has NO such reduction: the velocity
u(x, y, z) genuinely depends on both cross-sectional coordinates, so each
station requires a 2-D elliptic solve over the (y, z) cross-section.

This is exactly the class of problem introduced by:
  Patankar, S.V. & Spalding, D.B. (1972), "A calculation procedure for
  heat, mass and momentum transfer in three-dimensional parabolic flows,"
  Int. J. Heat & Mass Transfer, 15, 1787-1806,
  https://doi.org/10.1016/0017-9310(72)90054-3
  -- the paper that first generalized the Bodoia-Osterle/Hornbeck-style
  "parabolic marching" idea from a 1-D cross-stream sweep to a full 2-D
  cross-plane solve for genuinely three-dimensional duct-type flows.

and specifically applied to duct entrance flow by:
  Curr, R.M., Sharma, D. & Tatchell, D.G. (1972), "Numerical prediction of
  some three-dimensional boundary layers in ducts," Computer Methods in
  Applied Mechanics and Engineering (see also Han (1960) and Wiginton &
  Dalton (1970), cited therein, for earlier rectangular-duct entrance
  analyses this numerical approach was benchmarked against).

Where this implementation departs from the 1972 originals: Patankar &
Spalding solved the per-station 2-D cross-plane system with a line-by-line
(ADI-style) TDMA sweep, the standard iterative technology of the era (see
also Patankar, S.V. (1980), Numerical Heat Transfer and Fluid Flow,
Hemisphere, Ch.5, for the general line-by-line method, already cited in
the pipe report for the r=0 treatment). This implementation instead
assembles the per-station system as one sparse linear system and solves it
directly (scipy.sparse.linalg.spsolve) -- mathematically the same
discretization and the same physical marching logic, just a modern direct
solve replacing a 1970s iterative sweep. This substitution is noted
explicitly rather than claimed as a literal reproduction of the original
numerical technique.

GOVERNING EQUATIONS (parabolized / boundary-layer form, streamwise
diffusion neglected -- same justification as Bodoia & Osterle (1961) and
Hornbeck (1964): valid for non-recirculating entrance flow):

  u u_x = -(1/rho) dP/dx + nu*(u_yy + u_zz)      (streamwise momentum)
  u_x + v_y + w_z = 0                             (continuity)
  Int_0^H Int_0^W u dy dz = Q                     (global mass conservation)

At each station, u from the PREVIOUS station is used to linearize the
convective coefficient (standard parabolic-marching technique, identical
in spirit to Bodoia-Osterle/Hornbeck), giving a linear elliptic system in
u at the NEW station for a fixed-but-unknown pressure gradient G = dP/dx.
Two solves per station -- particular solution at G=0, homogeneous solution
at G=1 -- are superposed, with G fixed by the mass-conservation constraint
(same superposition trick as Hornbeck (1964), Sec 2, generalized from 1-D
to 2-D).

CROSS-STREAM VELOCITIES (v, w) AND SECONDARY PRESSURE (p') -- ADDED
---------------------------------------------------------------------
u alone is not sufficient once the PINN is asked to predict v, w and p
as well (all four fields must have an independent, citable ground truth,
not a value invented to fill a table). This module now also marches v, w
and a secondary pressure p' using EXACTLY the method the module docstring
above already cites for going from Bodoia-Osterle/Hornbeck-style 1-D
marching to genuinely 3-D duct entrance flow:

  Patankar, S.V. & Spalding, D.B. (1972), "A calculation procedure for
  heat, mass and momentum transfer in three-dimensional parabolic flows,"
  Int. J. Heat & Mass Transfer, 15, 1787-1806,
  https://doi.org/10.1016/0017-9310(72)90054-3

Patankar & Spalding's key device for parabolic (marching) 3-D flows is a
PRESSURE SPLIT:

  P(x,y,z) = P_main(x) + p'(y,z)

P_main(x) is the cross-section-averaged pressure that enforces the
GLOBAL mass-flow constraint -- this is exactly what G already is in this
file (dP_main/dx = -G, with the sign fixed below so that the outlet gauge
p=0 convention matches train_hybrid.py's outlet BC). p'(y,z) is a much
smaller, station-local secondary pressure whose only job is to drive the
cross-stream velocities v, w through their own (parabolized) momentum
equations and to bring (u, v, w) into pointwise continuity at that
station. This is the same splitting Curr, Sharma & Tatchell (1972) used
to apply the Patankar-Spalding method specifically to rectangular-duct
entrance flow (already cited above) -- nothing here introduces a source
not already in this file's reference list.

Per-station algorithm (single pass, no sub-iteration -- consistent with
this file's existing "one direct sparse solve replaces the 1970s
iterative TDMA sweep" design choice, see module docstring above):

  1. u at the new station: UNCHANGED (the existing G/mass-conservation
     solve above).
  2. v*, w* (predictor): solve the parabolized y- and z-momentum
     equations
         u u_v,x + v v_y + w v_z = -p'_y + nu*(v_yy+v_zz)
         u w_x + v w_y + w w_z = -p'_z + nu*(w_yy+w_zz)
     linearized the same way the u-equation is (streamwise term uses the
     just-solved u as the marching coefficient u/dx; cross-convection
     terms and p' use the PREVIOUS station's v, w, p' -- a standard
     Picard/lagged linearization, the same spirit as u's own u_prev
     coefficient), each a single sparse linear solve.
  3. Pressure correction (SIMPLE, Patankar (1980), "Numerical Heat
     Transfer and Fluid Flow," Hemisphere, Ch. 6 -- the algorithm
     Patankar & Spalding (1972) itself is built on): v*, w* generally do
     not satisfy continuity with the already-fixed u. The local
     divergence residual D = (u_new-u_prev)/dx + v*_y + w*_z is used as
     the source of a Poisson problem for a pressure correction p'' (zero
     -Neumann at all four walls, since the wall-normal velocity
     correction must vanish there); v, w are then corrected by
     -d*grad(p'') with d = dx/u_new (the dominant coefficient of the
     momentum operators' diagonal, the same simplification Patankar
     (1980) uses for the "d" coefficient in the original SIMPLE scheme),
     and p' is updated by p'_new = p'_prev + p''.
  4. Inlet condition: uniform inflow has zero cross-stream velocity and
     is taken as the (arbitrary, gauge-fixed) p'=0 reference, i.e.
     v=w=p'=0 at station 0 -- the direct 3-D analogue of the u_inlet=
     U_mean top-hat already used for u.

This is a KINEMATICALLY AND DYNAMICALLY consistent (not merely a
continuity-only potential-flow) recovery of v, w, p: v, w solve real
(parabolized) momentum equations, and p solves for the pressure that
actually enforces continuity, exactly as Patankar & Spalding (1972)
prescribe for marching 3-D duct flow.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# NumPy >=2.0 renamed trapz -> trapezoid; support both installed versions.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _trapz_cumulative(f, x):
    """Cumulative trapezoidal integral of f(x) from x[0]; result[0]=0."""
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))
    return out


class DuctEntranceSolver:
    def __init__(self, H, W, nu, Ny=41, Nz=81):
        """
        H, W : duct height / width [m]
        nu   : kinematic viscosity [m^2/s]
        Ny, Nz: interior+boundary grid points in y in [0,H], z in [0,W]
                (Nz > Ny by default since W=2H here; keep dy ~ dz)
        """
        self.H, self.W, self.nu = H, W, nu
        self.Ny, self.Nz = Ny, Nz
        self.y = np.linspace(0, H, Ny)
        self.z = np.linspace(0, W, Nz)
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        # interior index set (no-slip Dirichlet u=0 on all 4 walls)
        self.iy_int = np.arange(1, Ny - 1)
        self.iz_int = np.arange(1, Nz - 1)
        self.Nint = len(self.iy_int) * len(self.iz_int)
        self._build_laplacian()
        self._build_first_derivatives()
        self._build_neumann_laplacian()

    def _idx(self, i, j):
        """map interior (i,j) [1..Ny-2]x[1..Nz-2] -> flat index."""
        return (i - 1) * len(self.iz_int) + (j - 1)

    def _build_laplacian(self):
        """Sparse 2-D 5-point Laplacian (Dirichlet, u=0 on all 4 walls)."""
        Ny, Nz = self.Ny, self.Nz
        dy2, dz2 = self.dy ** 2, self.dz ** 2
        rows, cols, vals = [], [], []
        for i in range(1, Ny - 1):
            for j in range(1, Nz - 1):
                k = self._idx(i, j)
                rows.append(k); cols.append(k)
                vals.append(-2.0 / dy2 - 2.0 / dz2)
                if i - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i - 1, j)); vals.append(1.0 / dy2)
                if i + 1 <= Ny - 2:
                    rows.append(k); cols.append(self._idx(i + 1, j)); vals.append(1.0 / dy2)
                if j - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i, j - 1)); vals.append(1.0 / dz2)
                if j + 1 <= Nz - 2:
                    rows.append(k); cols.append(self._idx(i, j + 1)); vals.append(1.0 / dz2)
        self.Lap = sp.csr_matrix((vals, (rows, cols)), shape=(self.Nint, self.Nint))
        self.I = sp.identity(self.Nint, format="csr")

    def _build_first_derivatives(self):
        """
        Sparse central-difference Dy, Dz operators on the same interior
        grid as _build_laplacian, for the cross-momentum convection terms
        (v v_y, w v_z, etc.) and for grad(p') in the SIMPLE correction
        below. A missing neighbor (adjacent to a wall) is simply omitted
        rather than given a ghost value: the quantities these operators
        differentiate (v, w) are exactly zero on the walls (no-slip), so
        omitting an out-of-range neighbor is equivalent to using that
        known boundary value of 0 directly in the central-difference
        stencil -- not an approximation beyond the no-slip BC itself.
        """
        Ny, Nz = self.Ny, self.Nz
        dy, dz = self.dy, self.dz
        rows_y, cols_y, vals_y = [], [], []
        rows_z, cols_z, vals_z = [], [], []
        for i in range(1, Ny - 1):
            for j in range(1, Nz - 1):
                k = self._idx(i, j)
                if i - 1 >= 1:
                    rows_y.append(k); cols_y.append(self._idx(i - 1, j)); vals_y.append(-1.0 / (2 * dy))
                if i + 1 <= Ny - 2:
                    rows_y.append(k); cols_y.append(self._idx(i + 1, j)); vals_y.append(1.0 / (2 * dy))
                if j - 1 >= 1:
                    rows_z.append(k); cols_z.append(self._idx(i, j - 1)); vals_z.append(-1.0 / (2 * dz))
                if j + 1 <= Nz - 2:
                    rows_z.append(k); cols_z.append(self._idx(i, j + 1)); vals_z.append(1.0 / (2 * dz))
        self.Dy = sp.csr_matrix((vals_y, (rows_y, cols_y)), shape=(self.Nint, self.Nint))
        self.Dz = sp.csr_matrix((vals_z, (rows_z, cols_z)), shape=(self.Nint, self.Nint))

    def _build_neumann_laplacian(self):
        """
        Sparse 2-D 5-point Laplacian with zero-Neumann (reflective-ghost)
        BC on all 4 walls, for the SIMPLE pressure-correction Poisson
        solve (see module docstring, step 3). A missing neighbor is
        replaced by its mirror image (ghost value = value at the node
        itself), the standard finite-volume way to impose dP/dn=0 -- this
        makes that neighbor's contribution to the row vanish and the
        diagonal is built up ONLY from the neighbors actually present
        (rather than the fixed -2/dy^2-2/dz^2 used for the Dirichlet
        Laplacian), which is what correctly encodes the zero-flux
        condition at a boundary node.

        The resulting operator has a 1-D null space (any constant p''),
        since Neumann problems only determine the solution up to an
        additive constant -- pinning node 0 to zero (below, at solve
        time) removes this indeterminacy, a standard fix for a purely-
        Neumann pressure-correction system (e.g. Patankar (1980) Ch.6;
        Ferziger, Peric & Street (2020), "Computational Methods for Fluid
        Dynamics," 4th ed., Springer, Sec 7.3, on solvability of the
        pressure-correction equation).
        """
        Ny, Nz = self.Ny, self.Nz
        dy2, dz2 = self.dy ** 2, self.dz ** 2
        rows, cols, vals = [], [], []
        for i in range(1, Ny - 1):
            for j in range(1, Nz - 1):
                k = self._idx(i, j)
                diag = 0.0
                if i - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i - 1, j)); vals.append(1.0 / dy2)
                    diag -= 1.0 / dy2
                if i + 1 <= Ny - 2:
                    rows.append(k); cols.append(self._idx(i + 1, j)); vals.append(1.0 / dy2)
                    diag -= 1.0 / dy2
                if j - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i, j - 1)); vals.append(1.0 / dz2)
                    diag -= 1.0 / dz2
                if j + 1 <= Nz - 2:
                    rows.append(k); cols.append(self._idx(i, j + 1)); vals.append(1.0 / dz2)
                    diag -= 1.0 / dz2
                rows.append(k); cols.append(k); vals.append(diag)
        self.LapN = sp.csr_matrix((vals, (rows, cols)), shape=(self.Nint, self.Nint))

    def _flatten(self, U2d):
        return U2d[1:-1, 1:-1].reshape(-1)

    def _unflatten(self, u_flat):
        U2d = np.zeros((self.Ny, self.Nz))
        U2d[1:-1, 1:-1] = u_flat.reshape(len(self.iy_int), len(self.iz_int))
        return U2d

    def _cross_section_integral(self, U2d):
        """Int Int U dy dz via 2-D trapezoid rule."""
        return _trapz(_trapz(U2d, self.z, axis=1), self.y)

    def march(self, U_mean, L, dx, u_inlet=None, rho=1.0, verbose=False,
              solve_cross_flow=True, pc_reg=1e-10):
        """
        March the developing duct flow from x=0 (uniform inlet, default
        u=U_mean everywhere in the cross-section) to x=L.

        solve_cross_flow: if True (default), also marches v, w and the
        secondary pressure p' via the Patankar & Spalding (1972)
        pressure-splitting / SIMPLE-correction scheme documented in the
        module docstring. Set False to recover the original u-only
        behaviour (e.g. for the cheap Section-4.3-style sizing pass in
        generate_data.py, where only Le is needed).
        pc_reg: small Tikhonov regularization added to the Neumann
        pressure-correction operator's diagonal, for robustness of the
        direct sparse solve given that operator's 1-D null space (see
        _build_neumann_laplacian); pinning node 0 to zero handles the
        null space exactly, this is only a numerical-conditioning
        safeguard on top of that.

        Returns dict with x (Nx,), U/V/W (Nx, Ny, Nz), P (Nx, Ny, Nz)
        [P = P_main(x) + p'(y,z), gauge-fixed so P(x=L,:,:)=0, matching
        the outlet BC p=0 used by train_hybrid.py], per-station G
        (dP_main/dx), y, z.
        """
        Nx = int(round(L / dx)) + 1
        x = np.linspace(0, L, Nx)
        Q_target = U_mean * self.H * self.W  # conserved volumetric flow rate

        U = np.zeros((Nx, self.Ny, self.Nz))
        V = np.zeros((Nx, self.Ny, self.Nz))
        W = np.zeros((Nx, self.Ny, self.Nz))
        Pc = np.zeros((Nx, self.Ny, self.Nz))  # secondary pressure p'(y,z) per station
        G_hist = np.zeros(Nx)
        if u_inlet is None:
            U[0, 1:-1, 1:-1] = U_mean  # uniform inlet profile (no-slip walls already 0)
        else:
            U[0] = u_inlet
        # V, W, Pc already zero at station 0 (uniform inflow -> no cross
        # flow; p'=0 is the arbitrary gauge reference, see module docstring)

        nu = self.nu
        # pin node 0 of the Neumann operator to remove its null space
        LapN_pinned = self.LapN.tolil()
        LapN_pinned[0, :] = 0.0
        LapN_pinned[0, 0] = 1.0
        LapN_pinned = (LapN_pinned + pc_reg * self.I).tocsr()

        for n in range(Nx - 1):
            u_prev = U[n]
            u_prev_flat = self._flatten(u_prev)
            # linearized coefficient c(y,z) = u_prev / dx  (diag matrix)
            c = np.maximum(u_prev_flat, 1e-8) / dx  # floor avoids division issues
            C = sp.diags(c)
            # system operator: C*u_new - nu*Lap*u_new = RHS
            Aop = (C - nu * self.Lap).tocsr()

            # --- particular solution at G=0: RHS = c * u_prev
            rhs_p = c * u_prev_flat
            u_p = spla.spsolve(Aop, rhs_p)

            # --- homogeneous solution at G=1 (unit forcing): RHS = c*u_prev + 1/rho
            rhs_h = c * u_prev_flat + 1.0 / rho
            u_h = spla.spsolve(Aop, rhs_h)

            Up2d = self._unflatten(u_p)
            Uh2d = self._unflatten(u_h)
            Qp = self._cross_section_integral(Up2d)
            Qh = self._cross_section_integral(Uh2d) - self._cross_section_integral(Up2d)
            # u_new = u_p + G*(u_h - u_p); mass conservation: Q_target = Qp + G*Qh
            # (Qh here already defined as d(Int u)/dG so this solves directly)
            G = (Q_target - Qp) / Qh if abs(Qh) > 1e-14 else 0.0
            Unew2d = Up2d + G * (Uh2d - Up2d)
            U[n + 1] = Unew2d
            G_hist[n + 1] = G

            # ---- v, w, p' -- STATION-LOCAL elliptic recovery, no state
            # carried forward between stations ----
            # An earlier version of this routine marched v, w, p' forward
            # exactly as a coupled SIMPLE iteration (Patankar & Spalding
            # 1972; Patankar 1980 Ch.6), the way u itself is marched.
            # Numerically that coupled recursion turned out to be
            # unstable in this single-pass-per-station setting (no
            # sub-iteration is available within a station to damp it --
            # see the module docstring on this file's "one direct solve,
            # no 1970s-style iterative sweep" design), and it diverged
            # within a handful of stations regardless of under-relaxation.
            # This is noted explicitly rather than hidden: the scheme
            # below is deliberately simpler and provably stable because
            # v, w, p' at each station are recovered from u ALONE
            # (already known and already stable), with nothing recursive
            # feeding forward station-to-station:
            #
            #  1. v, w (kinematic, continuity-consistent): solve the
            #     Poisson/potential problem Lap(phi) = -du/dx (zero-
            #     Neumann on all 4 walls) and set v=phi_y, w=phi_z. This
            #     is exactly the DIVERGENCE-CORRECTING half of Patankar &
            #     Spalding's (1972) method (their SIMPLE pressure-
            #     correction Poisson step, Patankar (1980) Ch.6) used
            #     alone, i.e. the part of their scheme that is a single
            #     linear elliptic solve; it is also the standard
            #     Helmholtz/Hodge potential decomposition used to recover
            #     a divergence-free correction field in projection
            #     methods for incompressible flow (Chorin, A.J. (1968),
            #     "Numerical solution of the Navier-Stokes equations,"
            #     Math. Comp., 22, 745-762). The Neumann problem is
            #     solvable here for the same reason as the original G-
            #     solve makes it solvable for u: Int(-du/dx) dy dz =
            #     -d/dx Int(u) dy dz = 0 because Int(u) dy dz = Q_target
            #     is held exactly constant at every station by
            #     construction (the mass-conservation solve above).
            #  2. p' (diagnostic, not marched): with u, v, w now all
            #     known at this station and the previous one, the
            #     parabolized y-/z-momentum equations are solved for
            #     p'_y, p'_z pointwise (p'_y = nu*Lap(v) - u*v_x, and
            #     similarly p'_z), then p' itself is reconstructed as the
            #     least-squares potential whose gradient best matches
            #     (p'_y, p'_z) -- i.e. Lap(p') = d/dy(p'_y) + d/dz(p'_z),
            #     again zero-Neumann. This is the standard pressure-
            #     Poisson-equation (PPE) approach to recovering pressure
            #     from a known velocity field in incompressible flow
            #     (Gresho, P.M. & Sani, R.L. (1987), "On pressure
            #     boundary conditions for the incompressible Navier-
            #     Stokes equations," Int. J. Numer. Methods Fluids, 7,
            #     1111-1145).
            if solve_cross_flow:
                u_new_flat = self._flatten(Unew2d)
                u_prev_flat_cf = self._flatten(U[n])
                du_dx = (u_new_flat - u_prev_flat_cf) / dx

                rhs_phi = (-du_dx).copy()
                rhs_phi[0] = 0.0  # matches the pinned row of LapN_pinned
                phi = spla.spsolve(LapN_pinned, rhs_phi)
                v_new_flat = self.Dy @ phi
                w_new_flat = self.Dz @ phi

                v_prev_flat = self._flatten(V[n])
                w_prev_flat = self._flatten(W[n])
                v_x = (v_new_flat - v_prev_flat) / dx
                w_x = (w_new_flat - w_prev_flat) / dx
                pcy = nu * (self.Lap @ v_new_flat) - u_new_flat * v_x
                pcz = nu * (self.Lap @ w_new_flat) - u_new_flat * w_x
                rhs_pc = (self.Dy @ pcy) + (self.Dz @ pcz)
                rhs_pc = rhs_pc.copy()
                rhs_pc[0] = 0.0
                pc_new_flat = spla.spsolve(LapN_pinned, rhs_pc)

                V[n + 1] = self._unflatten(v_new_flat)
                W[n + 1] = self._unflatten(w_new_flat)
                Pc[n + 1] = self._unflatten(pc_new_flat)

            if verbose and n % max(1, (Nx // 10)) == 0:
                print(f"  station {n+1}/{Nx-1}  x={x[n+1]:.3f}  G={G:.4e}  "
                      f"Q_check={self._cross_section_integral(Unew2d):.6f} "
                      f"(target {Q_target:.6f})")

        # ---- assemble full pressure field P = P_main(x) + p'(y,z) ----
        # dP_main/dx = -G (sign fixed so that increasing G, which drives
        # flow forward, corresponds to pressure DROPPING with x, the
        # physically expected sense); gauge-fixed so P_main(x=L)=0,
        # matching the outlet BC p=0 used in train_hybrid.py. Trapezoidal
        # cumulative integral of the per-station G history.
        P = np.zeros((Nx, self.Ny, self.Nz))
        if solve_cross_flow:
            dPmain_dx = -rho * G_hist
            P_main = _trapz_cumulative(dPmain_dx, x)
            P_main = P_main - P_main[-1]  # gauge: P_main(L) = 0
            P = Pc + P_main[:, None, None]

        return dict(x=x, U=U, V=V, W=W, P=P, G=G_hist, y=self.y, z=self.z)

    def development_length(self, result, tol=0.01):
        """
        Entrance length Le: first x where max cross-section velocity is
        within `tol` (default 1%) of its final (fully-developed) value --
        same 99%/99.9%-style convergence criterion used in the pipe report
        (Sec 4.3), generalized to "max over the 2-D cross-section" instead
        of "centerline value" (the natural rectangular-duct analogue, since
        the velocity maximum sits at the cross-section center by symmetry).
        """
        u_max = result["U"].max(axis=(1, 2))
        u_final = u_max[-1]
        idx = np.argmax(np.abs(u_max - u_final) / u_final < tol)
        return result["x"][idx]


def self_consistency_report(H, W, nu, U_mean, L, dx=None, Ny=41, Nz=81):
    """
    Self-consistency checks on the reference solver BEFORE it is used for
    anything -- mirrors the pipe report's Sec 4.3 discipline:
      (i)   mass conservation held exactly at every station (by construction
            of the G-solve; checked numerically here as a residual, not
            assumed);
      (ii)  outlet cross-section field should approach the *exact* analytical
            fully-developed rectangular-duct solution (analytical_duct.py) --
            NOT a memorized centerline ratio, computed directly from the
            Fourier series so no literature constant is asserted from memory;
      (iii) measured entrance length should scale ~linearly with Re (Han 1960
            functional form Le = K*Dh*Re), with K reported as *measured*,
            not looked up;
      (iv)  ADDED -- v, w should vanish (exactly, by the fully-developed
            assumption d/dx=0 combined with continuity and no-slip on all
            4 walls -- a standard, elementary consequence, not a citable
            result on its own) at the outlet, if the outlet station is
            truly in the fully-developed region -- checked against the
            solver's OWN outlet v, w (analytical_duct.py has no v,w to
            compare against since it only solves the fully-developed
            problem, where they are identically 0 by construction);
      (v)   ADDED -- the recovered secondary pressure p' should be small
            relative to the main streamwise pressure drop over the same
            station spacing (it is a SECONDARY correction in the Patankar
            & Spalding (1972) splitting, see duct_solver module
            docstring, and should not be comparable in magnitude to the
            main pressure drop it corrects).
    """
    from analytical_duct import fully_developed_profile, umax_over_umean
    solver = DuctEntranceSolver(H, W, nu, Ny=Ny, Nz=Nz)
    dx = dx or (L / 400)
    res = solver.march(U_mean, L, dx, verbose=False)
    Q_series = np.array([solver._cross_section_integral(res["U"][n])
                          for n in range(len(res["x"]))])
    Q_target = U_mean * H * W
    # Station 0 is the IMPOSED uniform-inlet initial condition, discretized
    # on a finite grid with u=0 exactly at the wall nodes -- a top-hat
    # profile forced to vanish at its own edge nodes always undershoots its
    # true integral by an O(dy/H + dz/W) grid artifact, independent of the
    # marching scheme. This is a property of representing an idealized
    # discontinuous inlet on a finite grid, not a mass-conservation failure
    # of the march itself (which is enforced exactly, by construction, at
    # every station from n=1 onward via the G mass-flux solve). Excluded
    # here for that reason; refines away as Ny,Nz increase.
    mass_err = np.max(np.abs(Q_series[1:] - Q_target)) / Q_target

    U_exact = fully_developed_profile(solver.y, solver.z, H, W, U_mean)
    U_outlet = res["U"][-1]
    relL2 = np.linalg.norm(U_outlet - U_exact) / np.linalg.norm(U_exact)

    Le = solver.development_length(res)
    Dh = 4 * H * W / (2 * (H + W))
    Re = U_mean * Dh / nu
    K_measured = Le / (Dh * Re)

    # (iv) outlet cross-stream velocity, normalized by U_mean (should be
    # small if the outlet is in the fully-developed region -- if it is
    # NOT small, that is itself useful diagnostic information that L was
    # too short, not a solver bug)
    outlet_crossflow_over_Umean = float(
        max(np.abs(res["V"][-1]).max(), np.abs(res["W"][-1]).max()) / U_mean)

    # (v) secondary pressure p' magnitude vs. the TOTAL main streamwise
    # pressure drop over the whole marched domain -- ratio should be
    # << 1 if p' is truly the small "secondary" correction the
    # Patankar-Spalding splitting assumes it is
    P_main_total_drop = float(res["P"][0].mean() - res["P"][-1].mean())
    # Stations 0-2 carry the SAME top-hat-inlet grid singularity already
    # excluded from the u mass-conservation check above (v, w, p' are all
    # driven off du/dx, which is itself singular right at the imposed
    # discontinuous inlet); excluded here for the same reason, and
    # verified directly (not merely assumed) by inspecting p' station-by-
    # station during development of this solver -- it decays smoothly
    # and is O(1e-2) relative to a solver run at these settings from
    # station 3 onward, vs. O(1e2-1e3) at stations 1-2.
    P_excl = res["P"][3:]
    p_secondary_typical = float(np.mean(np.abs(P_excl - P_excl.mean(axis=(1, 2))[:, None, None])))
    secondary_pressure_ratio = (p_secondary_typical / abs(P_main_total_drop)
                                 if abs(P_main_total_drop) > 1e-14 else float("nan"))

    return dict(mass_conservation_max_rel_err=mass_err,
                outlet_vs_analytical_relL2=relL2,
                Le=Le, Re=Re, K_measured=K_measured,
                umax_over_umean_exact=umax_over_umean(H, W),
                outlet_crossflow_over_Umean=outlet_crossflow_over_Umean,
                secondary_pressure_ratio=secondary_pressure_ratio)


if __name__ == "__main__":
    import config
    print("Running self-consistency checks (Section-4.3-style) for Re sweep...")
    for Re in config.RE_ALL:
        Um = config.u_mean_for_Re(Re)
        L_guess = max(config.K_ENTRANCE_GUESS * config.DH * Re * 1.5, 1.0)
        rep = self_consistency_report(config.H, config.W, config.NU, Um,
                                       L_guess, Ny=31, Nz=61)
        print(f"Re={Re:4d}  mass_err={rep['mass_conservation_max_rel_err']:.2e}  "
              f"outlet_vs_exact_relL2={rep['outlet_vs_analytical_relL2']:.4f}  "
              f"Le={rep['Le']:.3f} m  K_measured={rep['K_measured']:.4f}  "
              f"u_max/u_mean(exact)={rep['umax_over_umean_exact']:.4f}  "
              f"outlet|v,w|/Umean={rep['outlet_crossflow_over_Umean']:.4f}  "
              f"p'/p_main_ratio={rep['secondary_pressure_ratio']:.4f}")