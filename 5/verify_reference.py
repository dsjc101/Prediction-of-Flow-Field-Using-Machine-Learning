"""
verify_reference.py -- verification of the pipe SA marching solver (analogue of Tables 3-4 of the
turbulent channel report).  Needs data/ref_Re*.npz (python reference_solver.py).

Checks (all against things the solver was never fitted to):
  (a) mass conservation   max_x | 2 int U r dr - 1 |
  (b) fully developed Cf  vs  Prandtl smooth-pipe law  and  McKeon et al. (2005) Superpipe fit
  (c) log-layer slope     d U+/d ln y+  for 50 < y+ < 0.15 Re_tau   (ideal 1/kappa = 2.44; Superpipe ~2.4)
  (d) centreline ratio    U_c/U_b  vs  1 + 1.326 sqrt(lambda)   (White / Schlichting-Gersten approx.)
  (e) development length  L_dev = x after which Cf stays within 1 % of its final value; must be < 0.7 L
  (f) grid / step convergence at Re = 28 000 (three grids)
"""
import argparse, json, os
import numpy as np
import config as C
import reference_solver as R


def load(Re):
    return dict(np.load(C.ref_path(Re)))


def ldev(x, cf):
    bad = np.where(np.abs(cf / cf[-1] - 1.0) > 0.01)[0]
    return x[bad[-1] + 1] if len(bad) else 0.0


def log_slope(res, ret):
    yp = res["y"] * ret
    Up = res["U"][-1] / np.sqrt(res["Cf"][-1] / 2.0)
    m = (yp > 50) & (yp < 0.15 * ret)
    if m.sum() < 6:
        return np.nan
    return np.polyfit(np.log(yp[m]), Up[m], 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_grid", action="store_true")
    a = ap.parse_args()
    rows = []
    hdr = (f"{'Re':>8} {'Re_tau':>8} {'Re_tau Prandtl':>14} {'Cf':>9} {'Cf Prandtl':>10} {'dev%':>6} "
           f"{'dev% McKeon':>11} {'log slope':>9} {'Uc/Ub':>7} {'Uc/Ub corr':>10} {'L_dev':>7} {'L_dev/L':>7} {'mass err':>9}")
    print(hdr)
    for Re in C.RE_ALL_SORTED:
        if not os.path.exists(C.ref_path(Re)):
            print(f"{Re:>8}  (missing, run reference_solver.py)")
            continue
        r = load(Re)
        cf = r["Cf"][-1]
        ret = np.sqrt(cf / 2) * Re / 2
        cfp, _, retp = C.est_np(Re)
        cfm = C.mckeon_lambda(Re) / 4
        ls = log_slope(r, ret)
        uc = r["U"][-1][-1]
        uc_corr = 1 + 1.326 * np.sqrt(C.prandtl_lambda(Re))
        ld = ldev(r["x"], r["Cf"])
        row = dict(Re=Re, Re_tau=ret, Re_tau_prandtl=float(retp), Cf=cf, Cf_prandtl=float(cfp),
                   dev_pct=100 * (cf - cfp) / cfp, dev_mckeon_pct=100 * (cf - cfm) / cfm, log_slope=ls,
                   Uc_Ub=uc, Uc_Ub_corr=float(uc_corr), L_dev=ld, L_dev_over_L=ld / C.L,
                   mass_err=float(r["mass_err"].max()))
        rows.append(row)
        print(f"{Re:>8} {ret:8.1f} {retp:14.1f} {cf:9.5f} {cfp:10.5f} {row['dev_pct']:6.1f} {row['dev_mckeon_pct']:11.1f} "
              f"{ls:9.2f} {uc:7.4f} {uc_corr:10.4f} {ld:7.1f} {ld / C.L:7.2f} {row['mass_err']:9.1e}")
    bad = [r for r in rows if r["L_dev_over_L"] > 0.7]
    if bad:
        print("\nWARNING: development length exceeds 0.7 L for Re =", [b["Re"] for b in bad],
              "-> increase config.L (and regenerate data).")

    grid = []
    if not a.skip_grid:
        print("\nGrid convergence at Re = 28000 (N wall-normal cells, nx stations):")
        print(f"{'grid':>16} {'Cf(L)':>10} {'Uc(L)':>10} {'Cf(x=10R)':>11}")
        for N, nx in [(100, 200), (200, 400), (400, 800)]:
            p = R.Pipe(28000, N)
            res = p.march(nx=nx)
            i10 = np.argmin(np.abs(res["x"] - 10.0))
            g = dict(N=N, nx=nx, Cf_L=res["Cf"][-1], Uc_L=res["U"][-1][-1], Cf_10=res["Cf"][i10])
            grid.append(g)
            print(f"{'N=%d nx=%d' % (N, nx):>16} {g['Cf_L']:10.6f} {g['Uc_L']:10.5f} {g['Cf_10']:11.6f}")
        for k, nm in [("Cf_L", "Cf(L)"), ("Uc_L", "Uc(L)"), ("Cf_10", "Cf(10R)")]:
            print(f"   change mid->fine {nm}: {100 * abs(grid[2][k] - grid[1][k]) / abs(grid[2][k]):.3f} %")
        # step / outer-iteration independence
        p = R.Pipe(28000, 400)
        r1 = p.march(nx=800, max_dx=0.25, n_outer=3)
        r2 = p.march(nx=800, max_dx=0.10, n_outer=5)
        print(f"   sub-step 0.25/3 iter vs 0.10/5 iter: Cf(L) {r1['Cf'][-1]:.6f} vs {r2['Cf'][-1]:.6f} "
              f"({100 * abs(r1['Cf'][-1] - r2['Cf'][-1]) / r1['Cf'][-1]:.3f} %)")
    with open(os.path.join(C.OUT_DIR, "verify_reference.json"), "w") as f:
        json.dump(dict(table=rows, grid=grid), f, indent=1, default=float)
    print("\nsaved outputs/verify_reference.json")


if __name__ == "__main__":
    main()
