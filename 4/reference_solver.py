"""
reference_solver.py -- independent reference for developing TURBULENT channel flow.

This is the turbulent counterpart of the Bodoia-Osterle marching solver used in
the laminar report, and it is deliberately a DIFFERENT numerical method from the
PINN's own physics loss (full elliptic RANS), so agreement is an independent check.

Method
------
* Parabolised ("boundary-layer") RANS between parallel plates, marched in x from a
  uniform inlet (Patankar & Spalding 1970, 1972; Bodoia & Osterle 1961).
* Closure: Spalart-Allmaras one-equation model (SA 1992/94), SA-noft2 form, with
  the S~ limiter of Allmaras, Johnson & Spalart (2012). Inlet nu~ = 3 nu
  (Spalart & Rumsey 2007).
* Implicit (backward-Euler) in x, 2nd-order FV/FD on a wall-stretched grid in y,
  Picard iteration for the nonlinear coefficients at each station.
* Constant mass flux enforced by superposition of a particular and a homogeneous
  solution (same trick as the laminar report): U = Ua - G*Ub, G = dP/dx.
* Wall-resolved (first cell y+ < 1): no wall functions.

Non-dimensionalisation: h = 1 (half height), U_b = 1, nu = 1/Re_h, Re_h = Re/4.
Only the lower half y in [0,1] is solved (symmetry at y=1).

Pressure: in the boundary-layer approximation P = P(x) only. P is returned as
  P_hat = p / (0.5 U_b^2), gauge-fixed so P_hat(x=L) = 0 (same as the report).
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

import config as C


# ------------------------------------------------------------------- grid
def make_y_grid(N, dy1):
    """Wall-clustered faces: y_f(s) = 1 + tanh(b(s-1))/tanh(b), first cell = dy1."""
    def first(b):
        return 1.0 - np.tanh(b * (1.0 - 1.0 / N)) / np.tanh(b) - dy1
    b = brentq(first, 1e-3, 30.0)
    s = np.linspace(0.0, 1.0, N + 1)
    yf = 1.0 + np.tanh(b * (s - 1.0)) / np.tanh(b)
    yf[0], yf[-1] = 0.0, 1.0
    yc = 0.5 * (yf[1:] + yf[:-1])
    return yf, yc


def default_dy1():
    """First-cell height so that y+_c(first cell centre) = Y_STRETCH_YPLUS1 at the
    highest Re of the project (Re_tau from Dean)."""
    retau_max = C.retau_dean(max(C.RE_ALL))
    return 2.0 * C.Y_STRETCH_YPLUS1 / retau_max


def make_x_stations(L, nx, power=2.0):
    s = np.linspace(0.0, 1.0, nx)
    return L * s ** power


# --------------------------------------------------------------- SA terms
def sa_functions(nut_tilde, nu, dUdy, d):
    """Return nu_t, S_tilde, and the destruction coefficient cw1*fw/d^2 (so that
    destruction = coef * nut_tilde^2)."""
    sa = C.SA
    chi = nut_tilde / nu
    fv1 = chi ** 3 / (chi ** 3 + sa["cv1"] ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    nut = nut_tilde * fv1
    Om = np.abs(dUdy)
    Sbar = nut_tilde * fv2 / (sa["kappa"] ** 2 * d ** 2)
    cv2, cv3 = 0.7, 0.9
    S_reg = Om + Sbar
    S_neg = Om + Om * (cv2 ** 2 * Om + cv3 * Sbar) / ((cv3 - 2 * cv2) * Om - Sbar + 1e-30)
    St = np.where(Sbar >= -cv2 * Om, S_reg, S_neg)
    St = np.maximum(St, 1e-12)
    r = np.minimum(nut_tilde / (St * sa["kappa"] ** 2 * d ** 2), 10.0)
    g = r + sa["cw2"] * (r ** 6 - r)
    fw = g * ((1.0 + sa["cw3"] ** 6) / (g ** 6 + sa["cw3"] ** 6)) ** (1.0 / 6.0)
    return nut, St, sa["cw1"] * fw / d ** 2


# ---------------------------------------------------------- FD helpers (y)
class Stencil:
    """Non-uniform 3-point central first-derivative weights at cell centres with
    Dirichlet (wall, value 0) below and mirror-symmetry above."""

    def __init__(self, yf, yc):
        N = len(yc)
        self.N = N
        self.yf, self.yc = yf, yc
        self.dy = np.diff(yf)
        ym = np.empty(N); yp = np.empty(N)
        ym[1:] = yc[:-1]; ym[0] = 0.0                    # wall node at y = 0
        yp[:-1] = yc[1:]; yp[-1] = 2.0 - yc[-1]          # mirror ghost about y = 1
        hm = yc - ym
        hp = yp - yc
        self.a = -hp / (hm * (hm + hp))
        self.b = (hp - hm) / (hp * hm)
        self.c = hm / (hp * (hm + hp))
        # face-interpolation weight: value_f = w*q[j-1] + (1-w)*q[j]  (face j)
        w = (yc[1:] - yf[1:-1]) / (yc[1:] - yc[:-1])
        self.w = w
        # distances between neighbouring centres (west of cell j), wall for j=0
        self.dW = np.empty(N); self.dE = np.empty(N)
        self.dW[0] = yc[0]; self.dW[1:] = yc[1:] - yc[:-1]
        self.dE[:-1] = yc[1:] - yc[:-1]; self.dE[-1] = np.inf   # symmetry: zero flux

    def ddy(self, q, q_wall=0.0):
        """Central derivative at cell centres; wall value q_wall; mirror at top."""
        qm = np.empty_like(q); qp = np.empty_like(q)
        qm[1:] = q[:-1]; qm[0] = q_wall
        qp[:-1] = q[1:]; qp[-1] = q[-1]
        return self.a * qm + self.b * q + self.c * qp

    def face_avg(self, q, q_wall):
        """Face values (N+1): wall face = q_wall; interior linear; top = q[-1]."""
        f = np.empty(self.N + 1)
        f[0] = q_wall
        f[1:-1] = self.w * q[:-1] + (1.0 - self.w) * q[1:]
        f[-1] = q[-1]
        return f

    def tri_diffusion(self, coef_face):
        """Return (lower, diag, upper) of the operator  -(1/dy) d/dy( coef dq/dy ),
        q = 0 at the wall, zero flux at the top."""
        Dm = coef_face[:-1] / (self.dy * self.dW)
        Dp = coef_face[1:] / (self.dy * self.dE)
        Dp[-1] = 0.0
        return -Dm, Dm + Dp, -Dp


def _solve_tri(lower, diag, upper, rhs):
    N = len(diag)
    ab = np.zeros((3, N))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


# ------------------------------------------------------------------ solver
def _sa_eval(st, nu, nt_n, nt, U, V, a0, d, dUdy):
    """Residual F(nt) of the implicit SA station equation plus the pieces needed
    for its (approximate) tridiagonal Jacobian."""
    sa = C.SA
    sig = sa["sigma"]
    _, St, destr = sa_functions(nt, nu, dUdy, d)
    coef_f = st.face_avg(nu + nt, nu) / sig
    lo, di, up = st.tri_diffusion(coef_f)
    dnt = st.ddy(nt)
    diff = di * nt
    diff[1:] += lo[1:] * nt[:-1]
    diff[:-1] += up[:-1] * nt[1:]
    F = (a0 * (nt - nt_n) + V * dnt + diff - sa["cb2"] / sig * dnt ** 2
         - sa["cb1"] * St * nt + destr * nt ** 2)
    return F, St, destr, lo, di, up, dnt


def _sa_newton(st, nu, nt_n, nt0, U, V, dx, d, max_it=40, tol=1e-8):
    """Solve the implicit SA station equation
        a0 (nt - nt_n) + V dnt/dy - (1/sig)[ d/dy((nu+nt) dnt/dy) + cb2 dnt^2 ]
        - cb1 S~ nt + cw1 fw (nt/d)^2 = 0
    by Levenberg-Marquardt-damped Newton with monotone residual acceptance.
    Why not Picard with explicit production? Next to the wall cb1*S ~ O(10) while
    the convective transit time U/dx is ~0.01, so production is stiff: Picard
    does not converge there and the answer depends on the iteration count.
    Damped Newton (pseudo-transient continuation, Kelley & Keyes 1998) is the
    standard remedy and always lands on the positive root."""
    sa = C.SA
    sig = sa["sigma"]
    a0 = np.maximum(U, 1e-8) / dx
    dUdy = st.ddy(U)
    nt = np.maximum(nt0, 1e-12)
    F, St, destr, lo, di, up, dnt = _sa_eval(st, nu, nt_n, nt, U, V, a0, d, dUdy)
    rn = np.linalg.norm(F)
    lam = 0.1 * np.max(sa["cb1"] * St + a0)
    lam_min = 1e-6 * np.max(a0)
    for it in range(max_it):
        cr = 2.0 * sa["cb2"] / sig * dnt
        jl = lo + V * st.a - cr * st.a
        jd = di + a0 + V * st.b - cr * st.b - sa["cb1"] * St + 2.0 * destr * nt
        ju = up + V * st.c - cr * st.c
        accepted = False
        for trial in range(12):
            delta = _solve_tri(jl, jd + lam, ju, -F)
            nt_try = np.maximum(nt + delta, 1e-12)
            Ft, St_t, destr_t, lo_t, di_t, up_t, dnt_t = _sa_eval(
                st, nu, nt_n, nt_try, U, V, a0, d, dUdy)
            rt = np.linalg.norm(Ft)
            if rt < rn or rt < 1e-13:
                accepted = True
                break
            lam *= 4.0
        step = np.max(np.abs(nt_try - nt)) / (nu + np.max(nt_try))
        if not accepted:
            break
        nt, F, St, destr, lo, di, up, dnt, rn = nt_try, Ft, St_t, destr_t, lo_t, di_t, up_t, dnt_t, rt
        lam = max(lam / 3.0, lam_min)
        if step < tol:
            break
    return nt


def solve(Re, N=C.NY_REF, nx=C.NX_REF, L=C.L_OVER_H, dy1=None, max_dx=0.25,
          outer=3, verbose=False, nut_inlet=None):
    """March the SA-RANS boundary-layer equations. Returns a dict of fields on
    (x_stations, y_cell_centres)."""
    Reh = C.re_h(Re)
    nu = 1.0 / Reh
    dy1 = default_dy1() if dy1 is None else dy1
    yf, yc = make_y_grid(N, dy1)
    st = Stencil(yf, yc)
    dy = st.dy
    d = yc.copy()                                       # distance to nearest wall
    xs = make_x_stations(L, nx)

    U = np.ones(N)
    nin = C.SA_NUT_INLET_OVER_NU if nut_inlet is None else nut_inlet
    nut_t = np.full(N, nin * nu)
    V = np.zeros(N)

    Uall = np.empty((nx, N)); Vall = np.empty((nx, N)); NTall = np.empty((nx, N))
    G_all = np.zeros(nx)
    tauw = np.zeros(nx)
    Uall[0], Vall[0], NTall[0] = U, V, nut_t

    def wall_shear_rate(Uc):
        # 2nd-order wall derivative: quadratic through (0,0), (yc0,U0), (yc1,U1)
        h1, h2 = yc[0], yc[1]
        return (h2 ** 2 * Uc[0] - h1 ** 2 * Uc[1]) / (h1 * h2 * (h2 - h1))

    x_prev = 0.0
    G = 0.0
    for k in range(1, nx):
        x_target = xs[k]
        nsub = max(1, int(np.ceil((x_target - x_prev) / max_dx)))
        edges = np.linspace(x_prev, x_target, nsub + 1)
        for sub in range(nsub):
            dx = edges[sub + 1] - edges[sub]
            Un, nutn = U.copy(), nut_t.copy()
            Vit, nutit, Unew = V.copy(), nut_t.copy(), Un.copy()
            for it in range(outer):
                nut_turb, _, _ = sa_functions(np.maximum(nutit, 1e-30), nu, st.ddy(Unew), d)
                nu_ef = st.face_avg(nu + nut_turb, nu)
                lo, di, up = st.tri_diffusion(nu_ef)
                a0 = np.maximum(Un, 1e-8) / dx
                lo = lo + Vit * st.a
                di = di + a0 + Vit * st.b
                up = up + Vit * st.c
                Ua = _solve_tri(lo, di, up, a0 * Un)
                Ub = _solve_tri(lo, di, up, np.ones(N))
                G = (np.sum(Ua * dy) - 1.0) / np.sum(Ub * dy)
                U_it = Ua - G * Ub
                Vf = np.zeros(N + 1)
                Vf[1:] = -np.cumsum(dy * (U_it - Un) / dx)
                V_it = 0.5 * (Vf[1:] + Vf[:-1])
                nt_new = _sa_newton(st, nu, nutn, nutit, U_it, V_it, dx, d)
                change = max(np.max(np.abs(U_it - Unew)),
                             np.max(np.abs(nt_new - nutit)) / (nu + np.max(nt_new)))
                Unew, Vit, nutit = U_it, V_it, nt_new
                if change < 1e-7:
                    break
            U, V, nut_t = Unew, Vit, nutit
        x_prev = x_target
        Uall[k], Vall[k], NTall[k] = U, V, nut_t
        G_all[k] = G
        tauw[k] = nu * wall_shear_rate(U)
        if verbose and k % 200 == 0:
            print(f"  Re={Re:.0f} station {k}/{nx} x={x_target:8.3f}  dPdx={G:+.4e}  Cf={2*tauw[k]:.5f}")
    tauw[0] = tauw[1]
    G_all[0] = G_all[1]

    P = np.zeros(nx)
    P[1:] = np.cumsum(0.5 * (G_all[1:] + G_all[:-1]) * np.diff(xs))
    P -= P[-1]
    nut_turb_all = np.empty_like(NTall)
    for k in range(nx):
        nut_turb_all[k], _, _ = sa_functions(np.maximum(NTall[k], 1e-30), nu, st.ddy(Uall[k]), d)
    return dict(Re=float(Re), Reh=Reh, nu=nu, x=xs, y=yc, yf=yf, U=Uall, V=Vall,
                P_hat=2.0 * P, dPdx=G_all, nut_tilde=NTall, nut=nut_turb_all,
                Cf=2.0 * tauw, utau=np.sqrt(np.abs(tauw)),
                Retau=np.sqrt(np.abs(tauw)) * Reh)


def development_length(sol, tol=0.01):
    """x/h after which Cf stays within tol of its final value (fully-developed point)."""
    cf = sol["Cf"]; x = sol["x"]
    ref = cf[-1]
    bad = np.where(np.abs(cf - ref) > tol * ref)[0]
    return x[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(x) else x[0]


if __name__ == "__main__":
    import time
    import sys
    Re = float(sys.argv[1]) if len(sys.argv) > 1 else 40000.0
    t0 = time.time()
    s = solve(Re, verbose=True)
    print("time %.1fs" % (time.time() - t0))
    print("Cf_fd=%.5f  Dean=%.5f  Re_tau=%.1f (Dean %.1f)  Lbar_dev=%.1f" % (
        s["Cf"][-1], 0.073 * (Re / 2) ** -0.25, s["Retau"][-1], C.retau_dean(Re), development_length(s)))
