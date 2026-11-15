"""
reference_solver.py -- independent PARABOLIC marching solver for developing turbulent PIPE flow
with the Spalart-Allmaras (SA-noft2, Allmaras-Johnson-Spalart limiter) closure.

It is the axisymmetric counterpart of the channel report's solver (Patankar-Spalding style):
  * streamwise diffusion dropped, P = P(x) only                    (boundary-layer form)
  * U U_x + V U_r = -dP/dx + (1/r) d/dr [ r nu_e dU/dr ]
  * U_x + (1/r) d(rV)/dr = 0
  * 2 * int_0^1 U r dr = 1   (U_b = 1, constant mass flux)         via G-superposition
  * SA transport eq. solved with a damped (Levenberg-Marquardt) Newton iteration per station.

Non-dimensionalisation:  lengths / R,  velocities / U_b,  nu* = 2/Re,  P kinematic (P_hat = 2 P).
Wall distance coordinate y = 1 - r  (wall y=0, axis y=1).  Wall-normal velocity Vy = -V_radial.
The stored V is the physical RADIAL velocity (positive outward), V = -Vy.

Usage:  python reference_solver.py                 (all 10 Re, parallel)
        python reference_solver.py --re 10000 28000
"""
import argparse, os, time, sys
import numpy as np
from scipy.linalg import solve_banded
import config as C

# ---- SA constants (NASA TMR) ---------------------------------------------------------
CB1, CB2, SIG, KAP = 0.1355, 0.622, 2.0 / 3.0, 0.41
CW1 = CB1 / KAP ** 2 + (1.0 + CB2) / SIG
CW2, CW3, CV1 = 0.3, 2.0, 7.1
CV2, CV3 = 0.7, 0.9


def make_grid(N, re, beta_target_re=None):
    """tanh-stretched faces in y in [0,1], clustered at the wall."""
    re_hi = beta_target_re if beta_target_re else max(C.RE_ALL)
    _, _, ret_hi = C.est_np(re_hi)
    dy1 = 2.0 * C.Y_PLUS_FIRST / ret_hi

    def first(beta):
        s = 1.0 / N
        return 1.0 + np.tanh(beta * (s - 1.0)) / np.tanh(beta)
    lo, hi = 1e-3, 30.0
    for _ in range(200):               # first(beta) decreases with beta
        mid = 0.5 * (lo + hi)
        if first(mid) > dy1:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    s = np.linspace(0.0, 1.0, N + 1)
    yf = 1.0 + np.tanh(beta * (s - 1.0)) / np.tanh(beta)
    yf[0], yf[-1] = 0.0, 1.0
    return yf


class Pipe:
    def __init__(self, Re, N=C.N_R):
        self.Re = float(Re)
        self.nu = C.nu_star(Re)
        self.N = N
        self.yf = make_grid(N, Re)
        self.rf = 1.0 - self.yf                       # radius at faces (wall=1 ... axis=0)
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])
        self.rc = 1.0 - self.yc
        self.vol = 0.5 * (self.rf[:-1] ** 2 - self.rf[1:] ** 2)   # int r dy per cell
        self.dyc = np.diff(self.yc)                   # centre-to-centre distance, faces 1..N-1
        # face geometric coefficients for  r * (dU/dy)  fluxes
        self.wface = np.zeros(N + 1)
        self.wface[0] = self.rf[0] / self.yc[0]       # wall face: U=0 at y=0, distance yc[0]
        self.wface[1:N] = self.rf[1:N] / self.dyc
        self.wface[N] = 0.0                           # axis: r = 0  -> no flux
        # linear interpolation weight for face values of cell quantities (faces 1..N-1)
        self.lam = (self.yf[1:N] - self.yc[:-1]) / self.dyc
        # central first-derivative stencil (non-uniform), incl. wall (value 0) and axis (mirror)
        N_ = N
        ym = np.concatenate(([0.0], self.yc[:-1]))                  # y_{j-1}  (wall for j=0)
        yp = np.concatenate((self.yc[1:], [2.0 - self.yc[-1]]))     # y_{j+1}  (mirror for j=N-1)
        hm, hp = self.yc - ym, yp - self.yc
        self.am = -hp / (hm * (hm + hp))
        self.a0 = (hp - hm) / (hm * hp)
        self.ap = hm / (hp * (hm + hp))
        self.a0[-1] += self.ap[-1]                                 # mirror: U_N = U_{N-1}
        self.ap_eff = self.ap.copy(); self.ap_eff[-1] = 0.0

    # ---- helpers ------------------------------------------------------------------
    def ddy(self, q):
        out = self.a0 * q
        out[1:] += self.am[1:] * q[:-1]
        out[:-1] += self.ap_eff[:-1] * q[1:]
        return out                                                   # wall value 0 -> am[0]*0

    def face(self, q, wall_val=0.0):
        f = np.empty(self.N + 1)
        f[0] = wall_val
        f[1:self.N] = (1 - self.lam) * q[:-1] + self.lam * q[1:]
        f[self.N] = q[-1]
        return f

    def nut_from_nt(self, nt):
        chi = nt / self.nu
        fv1 = chi ** 3 / (chi ** 3 + CV1 ** 3)
        return nt * fv1

    def wall_shear_grad(self, U):
        """dU/dy at the wall from the quadratic through (0,0), (yc0,U0), (yc1,U1)."""
        y0, y1 = self.yc[0], self.yc[1]
        return (U[0] * y1 ** 2 - U[1] * y0 ** 2) / (y0 * y1 * (y1 - y0))

    # ---- momentum -----------------------------------------------------------------
    def solve_momentum(self, Uo, Ustar, Vystar, nut, dx):
        N = self.N
        nue_c = self.nu + nut
        nue_f = self.face(nue_c, wall_val=self.nu)
        wf = self.wface * nue_f                          # r*nu_e/dy at every face
        # diffusion operator D U = (1/vol_j) [ wf[j+1](U_{j+1}-U_j) - wf[j](U_j - U_{j-1}) ]
        lower = np.zeros(N); diag = np.zeros(N); upper = np.zeros(N)
        diag += -(wf[1:] + wf[:-1]) / self.vol
        lower[1:] = wf[1:N] / self.vol[1:]
        upper[:-1] = wf[1:N] / self.vol[:-1]
        # matrix A = diag(U*/dx) + convective(Vy* d/dy) - D
        conv_m = Vystar * self.am
        conv_0 = Vystar * self.a0
        conv_p = Vystar * self.ap_eff
        ab = np.zeros((3, N))
        ab[1] = Ustar / dx + conv_0 - diag
        ab[0, 1:] = conv_p[:-1] - upper[:-1]          # super-diagonal  A[j, j+1]
        ab[2, :-1] = conv_m[1:] - lower[1:]           # sub-diagonal    A[j, j-1]
        rhs_a = Ustar * Uo / dx
        sol = solve_banded((1, 1), ab, np.column_stack((rhs_a, np.ones(N))))
        Ua, Ub = sol[:, 0], sol[:, 1]
        G = (np.sum(Ua * self.vol) - 0.5) / np.sum(Ub * self.vol)
        return Ua - G * Ub, G

    def wall_normal_velocity(self, U, Uo, dx):
        """returns Vy at cell centres (positive towards the axis) from continuity"""
        rVy = np.zeros(self.N + 1)                      # r*Vy at faces, 0 at the wall
        rVy[1:] = np.cumsum(-self.vol * (U - Uo) / dx)
        Vf = np.zeros(self.N + 1)
        Vf[:-1] = rVy[:-1] / np.maximum(self.rf[:-1], 1e-14)
        Vf[-1] = 0.0                                    # axis
        return 0.5 * (Vf[:-1] + Vf[1:]), rVy[-1]

    # ---- Spalart-Allmaras ------------------------------------------------------------
    def sa_residual(self, nt, nto, U, Vy, dUdy, dx):
        nu = self.nu
        chi = nt / nu
        fv1 = chi ** 3 / (chi ** 3 + CV1 ** 3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        d = self.yc
        Om = np.abs(dUdy)
        Sb = nt * fv2 / (KAP ** 2 * d ** 2)
        den = (CV3 - 2 * CV2) * Om - Sb
        Sb_lim = np.where(Sb >= -CV2 * Om, Sb,
                          Om * (CV2 ** 2 * Om + CV3 * Sb) / np.where(np.abs(den) < 1e-30, 1e-30, den))
        St = Om + Sb_lim
        St = np.maximum(St, 1e-12)
        r = np.minimum(nt / (St * KAP ** 2 * d ** 2), 10.0)
        g = r + CW2 * (r ** 6 - r)
        fw = g * ((1 + CW3 ** 6) / (g ** 6 + CW3 ** 6)) ** (1.0 / 6.0)
        prod = CB1 * St * nt
        dest = CW1 * fw * (nt / d) ** 2
        # diffusion (conservative) + cb2 |grad|^2
        nf = self.face(nt, wall_val=0.0)
        dn = np.zeros(self.N + 1)
        dn[0] = nt[0] / self.yc[0]
        dn[1:self.N] = (nt[1:] - nt[:-1]) / self.dyc
        flux = self.rf * (nu + nf) * dn                   # r (nu + nt) dnt/dy at faces ; r=0 at axis
        diff = (flux[1:] - flux[:-1]) / self.vol
        dndy = self.ddy(nt)
        conv = U * (nt - nto) / dx + Vy * dndy
        return conv - prod + dest - (diff + CB2 * dndy ** 2) / SIG

    def solve_sa(self, nto, U, Vy, dx, nt0):
        N = self.N
        dUdy = self.ddy(U)
        nt = np.maximum(nt0.copy(), 1e-14)
        res = self.sa_residual(nt, nto, U, Vy, dUdy, dx)
        nrm = np.sqrt(np.mean(res ** 2))
        lam = 1e-3
        ncol = np.arange(N) % 3
        for it in range(60):
            # 3-colour finite-difference tridiagonal Jacobian
            eps = 1e-6 * np.maximum(nt, self.nu)
            ab = np.zeros((3, N))
            for c in range(3):
                pert = nt.copy()
                m = ncol == c
                pert[m] += eps[m]
                rp = self.sa_residual(pert, nto, U, Vy, dUdy, dx)
                dr = (rp - res)
                idx = np.where(m)[0]
                ab[1, idx] = dr[idx] / eps[idx]
                ok = idx[idx > 0]
                ab[0, ok] = dr[ok - 1] / eps[ok]            # dR_{j-1}/dn_j   (super-diagonal)
                ok = idx[idx < N - 1]
                ab[2, ok] = dr[ok + 1] / eps[ok]            # dR_{j+1}/dn_j   (sub-diagonal)
            accepted = False
            for _ in range(25):
                abl = ab.copy()
                abl[1] += lam * np.maximum(np.abs(ab[1]), 1.0 / dx)
                try:
                    step = solve_banded((1, 1), abl, -res)
                except Exception:
                    lam *= 10.0
                    continue
                nn = np.maximum(nt + step, 0.0)
                nr = self.sa_residual(nn, nto, U, Vy, dUdy, dx)
                nn_rms = np.sqrt(np.mean(nr ** 2))
                if np.isfinite(nn_rms) and nn_rms < nrm:
                    nt, res = nn, nr
                    lam = max(lam * 0.3, 1e-10)
                    accepted = True
                    break
                lam *= 5.0
            if not accepted:
                break
            nrm = nn_rms
            if nrm < 1e-9 * max(1.0, np.sqrt(np.mean((U / dx) ** 2))):
                break
        return nt

    # ---- main march ------------------------------------------------------------------
    def march(self, L=C.L, nx=C.N_X, max_dx=C.MAX_DX, n_outer=3, verbose=False):
        N = self.N
        xs = L * (np.arange(nx) / (nx - 1)) ** 2
        U = np.ones(N)
        nt = 3.0 * self.nu * np.ones(N)
        Uall = np.zeros((nx, N)); Vall = np.zeros((nx, N)); Nall = np.zeros((nx, N))
        G_all = np.zeros(nx); Cf = np.zeros(nx); mass = np.zeros(nx)
        Uall[0] = U; Nall[0] = nt
        Pk = 0.0
        P = np.zeros(nx)
        Vy = np.zeros(N)
        Cf[0] = np.nan
        for k in range(1, nx):
            xa, xb = xs[k - 1], xs[k]
            nsub = max(1, int(np.ceil((xb - xa) / max_dx)))
            dx = (xb - xa) / nsub
            for s in range(nsub):
                Uo, nto = U.copy(), nt.copy()
                Ust, nts = U.copy(), nt.copy()
                Vyst = Vy.copy()
                for it in range(n_outer):
                    nut = self.nut_from_nt(nts)
                    Unew, G = self.solve_momentum(Uo, Ust, Vyst, nut, dx)
                    Vyn, rVy_axis = self.wall_normal_velocity(Unew, Uo, dx)
                    ntn = self.solve_sa(nto, Unew, Vyn, dx, nts)
                    Ust, Vyst, nts = Unew, Vyn, ntn
                U, Vy, nt = Ust, Vyst, nts
                Pk += G * dx
            P[k] = Pk
            G_all[k] = G
            Uall[k] = U; Vall[k] = -Vy; Nall[k] = nt
            Cf[k] = 2.0 * self.nu * self.wall_shear_grad(U)
            mass[k] = abs(2.0 * np.sum(U * self.vol) - 1.0)
            if verbose and k % 100 == 0:
                print(f"  Re={self.Re:.0f} k={k}/{nx} x={xs[k]:.2f} Cf={Cf[k]:.5f}", flush=True)
        # first station Cf (from initial uniform profile is singular) -> copy next
        Cf[0] = Cf[1]
        P = P - P[-1]
        return dict(x=xs, U=Uall, V=Vall, nt=Nall, P_hat=2.0 * P, G=G_all, Cf=Cf,
                    mass_err=mass, y=self.yc, yf=self.yf, r=self.rc, Re=self.Re, nu=self.nu,
                    vol=self.vol)


def run_one(args):
    Re, N, nx, outdir = args
    t0 = time.time()
    p = Pipe(Re, N)
    res = p.march(nx=nx)
    path = outdir if outdir else C.ref_path(Re)
    np.savez_compressed(path, **res)
    return Re, time.time() - t0, res["Cf"][-1], res["mass_err"].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--nr", type=int, default=C.N_R)
    ap.add_argument("--nx", type=int, default=C.N_X)
    a = ap.parse_args()
    res = a.re if a.re else C.RE_ALL_SORTED
    jobs = [(r, a.nr, a.nx, None) for r in res]
    if a.workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for Re, dt, cf, me in pool.imap_unordered(run_one, jobs):
                print(f"Re={Re:.0f}: {dt:.0f}s  Cf(L)={cf:.5f}  max mass err={me:.1e}", flush=True)
    else:
        for j in jobs:
            Re, dt, cf, me = run_one(j)
            print(f"Re={Re:.0f}: {dt:.0f}s  Cf(L)={cf:.5f}  max mass err={me:.1e}", flush=True)


if __name__ == "__main__":
    main()
