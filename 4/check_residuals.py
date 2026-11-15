"""
check_residuals.py -- does the REFERENCE data satisfy the equations the PINN is asked to satisfy?

    python check_residuals.py --models outputs/abl_A_found.pt outputs/abl_B_data_only.pt outputs/abl_C_physics_x100.pt

Why: the ablation showed that raising the physics weight x100 cuts the PDE residual ~3x but makes u ~3x
WORSE. That means data and equations pull in different directions. Two possible causes:
  (1) MISMATCH: the reference (parabolic/boundary-layer: P=P(x) only, no streamwise diffusion) does not satisfy
      the full elliptic RANS+SA operator used in the PINN -> physics and data genuinely conflict;
  (2) OPTIMISATION: the reference does satisfy the operator (residual ~ 0) but the network cannot reach it.
This script computes the PINN's residuals ON THE REFERENCE FIELDS by finite differences (np.gradient, 2nd order,
non-uniform grids) and splits the x-momentum residual into the terms the boundary-layer approximation keeps and
the ones it drops. It then evaluates each model's autograd residual on the SAME grid points for comparison.
Residuals are scaled exactly as in physics.py (momentum, SA: / u_tau,est^2). Only VAL/TRAIN Re are allowed.
"""
import argparse
import os

import numpy as np
import torch

import config as C
import data as D
import physics as PH
import reference_solver as R
import train as T
from build_reference import load_reference

REGIONS = [("1<=x<10", 1.0, 10.0), ("10<=x<50", 10.0, 50.0), ("50<=x<L", 50.0, 1e9)]


def ref_terms(Re):
    r = load_reference(Re)
    x, yc = r["x"], r["y"]
    nu = 1.0 / r["Reh"]
    y = np.r_[0.0, yc]                                    # add the wall node (all fields 0 there)
    ext = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), a], axis=1)
    U, V, nt, nut = ext(r["U"]), ext(r["V"]), ext(r["nut_tilde"]), ext(r["nut"])
    P = 0.5 * r["P_hat"][:, None] * np.ones_like(U)       # kinematic pressure, P(x) only in the reference
    gx = lambda f: np.gradient(f, x, axis=0, edge_order=2)
    gy = lambda f: np.gradient(f, y, axis=1, edge_order=2)
    Ux, Uy, Vx, Vy = gx(U), gy(U), gx(V), gy(V)
    ne = nu + nut
    txx, txy, tyy = 2 * ne * Ux, ne * (Uy + Vx), 2 * ne * Vy
    Px, Py = gx(P), gy(P)
    ut2 = (0.5 * 0.073 * (Re / 2.0) ** -0.25)             # u_tau,est^2 (same scale as physics.py)
    conv = U * Ux + V * Uy
    xdiff_kept = gy(ne * Uy)                              # d/dy(nu_e U_y): kept by the BL approximation
    xdiff_dropped = gx(txx) + gy(ne * Vx)                 # streamwise-diffusion terms: dropped by it
    momx = conv + Px - (xdiff_kept + xdiff_dropped)
    # y-momentum: reference has P_y = 0; the PINN would need P_y = -(...) -> report that required P_y
    momy_no_p = U * Vx + V * Vy - (gx(txy) + gy(tyy))
    # SA with the PINN's vorticity Omega = |V_x - U_y|
    ntx, nty = gx(nt), gy(nt)
    d = np.maximum(y[None, :], 1e-8) * np.ones_like(U)
    _, St, destr = R.sa_functions(np.maximum(nt, 1e-30), nu, Uy - Vx, d)
    sa = (U * ntx + V * nty - (C.SA["cb1"] * St * nt - destr * nt ** 2
          + (gx((nu + nt) * ntx) + gy((nu + nt) * nty) + C.SA["cb2"] * (ntx ** 2 + nty ** 2)) / C.SA["sigma"]))
    out = dict(x=x, y=y, momx=momx / ut2, momy_noP=momy_no_p / ut2, sa=sa / ut2,
               conv=conv / ut2, Px=Px / ut2, kept=xdiff_kept / ut2, dropped=xdiff_dropped / ut2,
               yplus=y[None, :] * r["Retau"][-1] * np.ones_like(U), dropped_only=True)
    return out, r


def rms(a, m):
    return float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--re", type=float, nargs="*", default=[20000, 88000])
    a = ap.parse_args()
    device, dtype = T.get_device(), torch.float32
    models = [(os.path.basename(p).replace(".pt", ""), T.load_model(p, device)[0]) for p in a.models]
    for Re in a.re:
        assert Re in C.RE_TRAIN or Re in C.RE_VAL, "only TRAIN/VAL Re allowed here"
        t, r = ref_terms(Re)
        X, Y = np.meshgrid(t["x"], t["y"], indexing="ij")
        ok = (X >= 1.0)
        ok[:, :1] = False                    # wall node
        ok[:, -1:] = False                   # symmetry node (one-sided derivative)
        far = ok & (t["yplus"] >= 5.0)       # exclude the viscous sublayer where FD of the reference is least accurate
        print(f"\n=== Re={int(Re)}  (residuals in units of u_tau,est^2; FD on reference, y+>=5 only) ===")
        print(f"{'region':10s} {'conv':>8s} {'Px':>8s} {'y-diff(kept)':>13s} {'x-diff(dropped)':>16s} | {'NET mom_x':>10s} {'req. Py':>9s} {'SA':>9s}")
        for nm, lo, hi in REGIONS:
            m = far & (X >= lo) & (X < hi)
            print(f"{nm:10s} {rms(t['conv'], m):8.2f} {rms(t['Px'], m):8.2f} {rms(t['kept'], m):13.2f} "
                  f"{rms(t['dropped'], m):16.2f} | {rms(t['momx'], m):10.2f} {rms(t['momy_noP'], m):9.2f} {rms(t['sa'], m):9.2f}")
        if models:
            ix = np.where(t["x"] >= 1.0)[0][::4]
            iy = np.arange(1, len(t["y"]) - 1, 2)
            Xs, Ys = np.meshgrid(t["x"][ix], t["y"][iy], indexing="ij")
            yp = Ys * r["Retau"][-1]
            for name, m_ in models:
                res = {"momx": [], "sa": []}
                xx = torch.tensor(Xs.ravel()[:, None], device=device, dtype=dtype)
                yy = torch.tensor(Ys.ravel()[:, None], device=device, dtype=dtype)
                RR = torch.full_like(xx, float(Re))
                outs = []
                for i in range(0, xx.shape[0], 20000):
                    rr = PH.pde_residuals(m_, xx[i:i + 20000], yy[i:i + 20000], RR[i:i + 20000])
                    outs.append(torch.stack([rr["mom_x"], rr["sa"]], 1).detach().cpu().numpy()[:, :, 0])
                o = np.concatenate(outs)
                for nm, lo, hi in REGIONS:
                    mk = (Xs.ravel() >= lo) & (Xs.ravel() < hi) & (yp.ravel() >= 5.0)
                    res[nm] = (rms(o[:, 0], mk), rms(o[:, 1], mk))
                print(f"  PINN {name:22s} mom_x rms / SA rms:  " + "   ".join(f"{nm}: {res[nm][0]:7.2f}/{res[nm][1]:6.2f}" for nm, _, _ in REGIONS))
    print("""
HOW TO READ
 * If the reference's NET mom_x is ~ as large as the PINN's -> the reference does not satisfy the PINN's equations (MISMATCH);
   the large 'x-diff(dropped)' column says WHERE (inlet region). Physics and data then cannot both be satisfied.
 * If NET mom_x of the reference is much smaller than the PINN's (<~1) -> the equations are consistent with the data and
   the large PINN residual is an OPTIMISATION problem (capacity / iterations / loss scaling).
 * 'req. Py' is the pressure-gradient in y the full y-momentum equation would need (the reference has none).""")


if __name__ == "__main__":
    main()