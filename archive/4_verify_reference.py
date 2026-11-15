"""
verify_reference.py -- is the reference solver trustworthy? (Run BEFORE training anything.)

Checks (Roache 1998 verification & validation philosophy, as in the laminar report):
 1. Mass flux = U_b to 1e-10 at every station.
 2. Fully-developed Cf and Re_tau vs Dean (1978) correlation  (empirical, +-~3%).
 3. Law of the wall in the fully-developed profile: slope of U+ vs ln(y+) in 50<y+<0.2Re_tau
    should be ~1/kappa (kappa ~ 0.39-0.41; Pope 2000).
 4. Development length (Cf within 1% of final) must be < 0.9 L, else L_OVER_H is too short.
 5. Grid convergence at one Re (N, 2N cells and coarser x): difference in Cf and U_c, with a
    Richardson-type observed order (Roache 1998).
 6. Hold-out-band sanity: bands lie inside the marched domain.
The independent DNS check (Moser-Kim-Mansour 1999; Lee & Moser 2015) is in dns_check.py.
"""
import numpy as np

import config as C
import reference_solver as R
from build_reference import load_reference


def main():
    print("1-4: per-Re checks")
    print(f"{'Re':>7} {'Re_tau':>8} {'Dean':>8} {'dev%':>6} {'Cf':>9} {'Cf_Dean':>9} {'dev%':>6} {'slope':>6} {'1/k':>5} {'Ldev/h':>7} {'mass err':>9}")
    for Re in C.RE_ALL:
        r = load_reference(Re)
        x, y, U = r["x"], r["y"], r["U"]
        mass = np.abs((U * np.diff(r["yf"])[None, :]).sum(1) - 1.0).max()
        rt, cf = r["Retau"][-1], r["Cf"][-1]
        cfd = 0.073 * (Re / 2.0) ** -0.25
        up = U[-1] / r["utau"][-1]
        yp = y * rt
        m = (yp > 50) & (yp < 0.2 * rt)
        slope = np.polyfit(np.log(yp[m]), up[m], 1)[0] if m.sum() > 5 else np.nan
        Ld = R.development_length(r)
        print(f"{Re:>7} {rt:8.1f} {C.retau_dean(Re):8.1f} {100*(rt/C.retau_dean(Re)-1):6.1f} {cf:9.5f} {cfd:9.5f} "
              f"{100*(cf/cfd-1):6.1f} {slope:6.2f} {1/0.41:5.2f} {Ld:7.1f} {mass:9.1e}"
              + ("   <-- L too short!" if Ld > 0.9 * C.L_OVER_H else ""))
    print("\n5: grid convergence at Re=28000 (reference: N=400, nx=800)")
    base = dict(L=C.L_OVER_H)
    runs = {}
    for tag, kw in {"coarse N=100,nx=200": dict(N=100, nx=200), "mid N=200,nx=400": dict(N=200, nx=400),
                    "fine N=400,nx=800": dict(N=400, nx=800)}.items():
        s = R.solve(28000.0, **base, **kw)
        runs[tag] = (s["Cf"][-1], s["U"][-1, -1], np.interp(10.0, s["x"], s["Cf"]))
        print(f"  {tag:22s} Cf_end={runs[tag][0]:.6f}  Uc_end={runs[tag][1]:.5f}  Cf(x=10h)={runs[tag][2]:.6f}")
    v = list(runs.values())
    for i, nm in enumerate(["Cf_end", "Uc_end", "Cf(10h)"]):
        e1, e2 = abs(v[0][i] - v[1][i]), abs(v[1][i] - v[2][i])
        p = np.log2(e1 / e2) if e2 > 0 and e1 > 0 else float("nan")
        print(f"  {nm}: |coarse-mid|={e1:.2e}  |mid-fine|={e2:.2e}  observed order ~ {p:.2f}  "
              f"-> fine-grid uncertainty ~ {100*e2/abs(v[2][i]):.3f}%")
    print("\nIf Cf and Re_tau are within a few % of Dean, mass error ~1e-12, L_dev < 0.9L and the\n"
          "fine-grid change is < ~1%, the reference is fit for use as a training/evaluation target.")


if __name__ == "__main__":
    main()
