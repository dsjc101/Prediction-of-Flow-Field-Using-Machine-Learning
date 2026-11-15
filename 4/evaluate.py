"""
evaluate.py -- metrics, overfitting diagnostics and figures.

    python evaluate.py --models outputs/final_s0.pt outputs/final_s1.pt outputs/final_s2.pt
    python evaluate.py --models ... --final        # ALSO scores the TEST Re (do this ONCE)

Metric (identical philosophy to Section 5 of the laminar report):
    relL2(q) = 100 * sqrt(mean_i (q_pred - q_ref)^2) / scale(Re),   scale = GLOBAL, Re-only
    scale: U_b for u, 0.05*u_tau,est for v, 0.5*U_b^2 for p, (nu~/nu) unit for chi.
    (v is ~0 in the developed region and every field is 0 at the wall, so a "percent of local
     value" metric would explode -- the laminar report already documented this failure.)

Overfitting diagnostics produced
--------------------------------
 1. TRAIN-Re vs VAL-Re vs TEST-Re error and the ratios (generalisation gap).
 2. x-hold-out bands at TRAIN Re (data never used for the data loss).
 3. PDE residual on a FRESH independent set of collocation points (seed-fixed, not the training
    stream) at every Re -- a PINN that memorised its collocation points would show a much larger
    value here than in the training log.
 4. Seed-to-seed spread (mean +- std over the models passed in).
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data as D
import physics as PH
import train as T


def fresh_residual(model, Re, device, dtype, n=6000, seed=777):
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    Rt = torch.full((n, 1), float(Re), device=device, dtype=dtype)
    x = (C.L_OVER_H * torch.rand(n, 1, generator=g) ** 2).clamp(min=0.05).to(device, dtype)
    yp = 0.5 * (1.0 / D.retau_t(torch.tensor(float(Re)))) * (D.retau_t(torch.tensor(float(Re))) / 0.5) ** torch.rand(n, 1, generator=g)
    y = yp.to(device, dtype).clamp(1e-6, 1.0)
    r = PH.pde_residuals(model, x, y, Rt)
    return {k: float(torch.sqrt((v ** 2).mean()).item()) for k, v in r.items()}


def wall_cf(model, es_x, Re, device, dtype):
    """Cf(x) = 2 nu dU/dy|wall from the network (hard no-slip -> evaluate at tiny y)."""
    x = es_x.clone().requires_grad_(True)
    y = torch.zeros_like(x).requires_grad_(True)
    Rt = torch.full_like(x, float(Re))
    U, _, _, _ = model(x, y, Rt)
    Uy, = torch.autograd.grad(U.sum(), y)
    return (2.0 * (4.0 / Re) * Uy).detach().cpu().numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--final", action="store_true", help="include TEST Reynolds numbers (do once)")
    ap.add_argument("--tag", default="eval")
    a = ap.parse_args()
    device, dtype = T.get_device(), torch.float32
    models = [T.load_model(p, device)[0] for p in a.models]

    roles = [("TRAIN", C.RE_TRAIN), ("VAL", C.RE_VAL)]
    if a.final:
        roles += [("TEST-interp", C.RE_TEST_INTERP), ("TEST-extrap", C.RE_TEST_EXTRAP)]
    rows = []
    for role, res in roles:
        for Re in res:
            es = D.EvalSet(Re, device, dtype, stride_x=2, stride_y=1, final=a.final)
            per = [D.score_set(m, es) for m in models]
            fr = [fresh_residual(m, Re, device, dtype) for m in models]
            for region in per[0]:
                for f in ("u", "v", "p", "chi", "cf"):
                    vals = [p[region][f] for p in per]
                    rows.append(dict(role=role, Re=Re, region=region, field=f,
                                     mean=np.mean(vals), std=np.std(vals)))
            rows.append(dict(role=role, Re=Re, region="fresh-PDE-residual", field="mom_x",
                             mean=np.mean([f["mom_x"] for f in fr]), std=np.std([f["mom_x"] for f in fr])))
            rows.append(dict(role=role, Re=Re, region="fresh-PDE-residual", field="sa",
                             mean=np.mean([f["sa"] for f in fr]), std=np.std([f["sa"] for f in fr])))

    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(C.OUT_DIR, f"{a.tag}_metrics.csv")
    df.to_csv(csv, index=False)

    lines = ["=" * 78, f"EVALUATION ({len(models)} model(s)); relL2 %, mean +- std over seeds", "=" * 78]
    for role, _ in roles:
        sub = df[(df.role == role) & (df.region == "all") & (df.field.isin(["u", "v", "p", "cf"]))]
        piv = sub.pivot_table(index="Re", columns="field", values="mean")
        pivs = sub.pivot_table(index="Re", columns="field", values="std")
        lines.append(f"\n[{role}] whole-domain")
        for Re in piv.index:
            lines.append(f"  Re={int(Re):>7}: " + "  ".join(
                f"{f}={piv.loc[Re, f]:6.2f}+-{pivs.loc[Re, f]:4.2f}" for f in ("u", "v", "p", "cf")))
    # gap summary
    def role_mean(role, region="all"):
        s = df[(df.role.str.startswith(role)) & (df.region == region) & (df.field.isin(["u", "v", "p", "cf"]))]
        return {f: s[s.field == f]["mean"].mean() for f in ("u", "v", "p", "cf")}
    tr, va = role_mean("TRAIN"), role_mean("VAL")
    lines.append("\nGENERALISATION GAP (mean over Re of each role)")
    for f in ("u", "v", "p", "cf"):
        lines.append(f"  {f}: TRAIN {tr[f]:.2f}%  VAL {va[f]:.2f}%  ratio VAL/TRAIN = {va[f]/tr[f]:.2f}")
    if a.final:
        for nm in ("TEST-interp", "TEST-extrap"):
            te = role_mean(nm)
            lines.append(f"  {nm}: " + "  ".join(f"{f} {te[f]:.2f}% (x{te[f]/tr[f]:.2f} of TRAIN)" for f in ("u", "v", "p", "cf")))
    ho = df[(df.role == "TRAIN") & (df.region == "x-holdout-bands") & (df.field.isin(["u", "v", "p", "cf"]))]
    al = df[(df.role == "TRAIN") & (df.region == "all") & (df.field.isin(["u", "v", "p", "cf"]))]
    lines.append("\nX-HOLD-OUT BANDS at TRAIN Re (never in the data loss): "
                 + "  ".join(f"{f}: {ho[ho.field == f]['mean'].mean():.2f}% (all-x {al[al.field == f]['mean'].mean():.2f}%)" for f in ("u", "v", "p", "cf")))
    fr = df[df.region == "fresh-PDE-residual"]
    lines.append("FRESH-point scaled PDE residual (rms): " + "  ".join(
        f"{r}: " + ", ".join(f"{int(x.Re)}:{x['mean']:.2e}" for _, x in fr[(fr.role == r)].iterrows() if x.field == 'mom_x')
        for r in fr.role.unique()))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(C.OUT_DIR, f"{a.tag}_summary.txt"), "w").write(txt + "\n")

    # ------------------------------------------------------------------ figures
    m0 = models[0]
    show = [(C.RE_TRAIN[1], "TRAIN"), (C.RE_VAL[0], "VAL")]
    if a.final:
        show += [(C.RE_TEST_INTERP[0], "TEST"), (C.RE_TEST_EXTRAP[-1], "TEST-extrap")]
    fig, axs = plt.subplots(len(show), 4, figsize=(17, 3.6 * len(show)), squeeze=False)
    for i, (Re, role) in enumerate(show):
        es = D.EvalSet(Re, device, dtype, stride_x=1, stride_y=1, final=a.final, full=True)
        P = D.predict(m0, es).cpu().numpy()
        ny = len(es.iy)
        xs = es.ref["x"][es.ix]
        for j, frac in enumerate((0.05, 0.25, 0.9)):
            k = int(np.argmin(np.abs(xs - frac * C.L_OVER_H)))
            ax = axs[i, j]
            sl = slice(k * ny, (k + 1) * ny)
            ax.plot(es.ref["y"], es.u.cpu().numpy()[sl], "k", label="reference (SA-RANS)")
            ax.plot(es.ref["y"], P[sl, 0], "C1--", label="PINN")
            ax.set_title(f"{role} Re={Re}  U/U_b at x={xs[k]:.1f}h"); ax.set_xlabel("y/h")
            if i == 0 and j == 0:
                ax.legend()
        ax = axs[i, 3]
        xr = es.ref["x"]
        cf_p = wall_cf(m0, torch.tensor(xr[xr > 0.2], device=device, dtype=dtype)[:, None], Re, device, dtype)
        ax.semilogx(xr[xr > 0.2], es.ref["Cf"][xr > 0.2], "k", label="reference")
        ax.semilogx(xr[xr > 0.2], cf_p, "C1--", label="PINN")
        ax.set_title(f"{role} Re={Re}  Cf(x)"); ax.set_xlabel("x/h"); ax.set_ylim(0, 3 * es.ref["Cf"][-1])
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_DIR, f"{a.tag}_profiles_cf.png"), dpi=130)
    plt.close()

    fig, axs = plt.subplots(1, len(show), figsize=(4.8 * len(show), 3.8), squeeze=False)
    for i, (Re, role) in enumerate(show):
        es = D.EvalSet(Re, device, dtype, stride_x=2, stride_y=1, final=a.final, full=True)
        P = D.predict(m0, es).cpu().numpy()
        err = np.abs(P[:, 0] - es.u.cpu().numpy()).reshape(es.shape)
        xs = es.ref["x"][es.ix]
        im = axs[0, i].pcolormesh(xs, es.ref["y"][es.iy], err.T, shading="auto", vmin=0, vmax=0.05)
        axs[0, i].set_xscale("symlog", linthresh=1.0)
        axs[0, i].set_title(f"{role} Re={Re}: |dU|/U_b"); axs[0, i].set_xlabel("x/h"); axs[0, i].set_ylabel("y/h")
        plt.colorbar(im, ax=axs[0, i])
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_DIR, f"{a.tag}_u_error_maps.png"), dpi=130)
    plt.close()
    print("wrote", csv, "and figures in", C.OUT_DIR)


if __name__ == "__main__":
    main()