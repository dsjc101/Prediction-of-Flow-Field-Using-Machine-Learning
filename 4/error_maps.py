"""
error_maps.py -- error maps over the whole domain for all 4 outputs (U, V, P, nu~), one figure per Re,
plus a sign analysis: where the REFERENCE is negative, is the MODEL negative too?

    python error_maps.py --models outputs_sparse/final_s0.pt outputs_sparse/final_s1.pt outputs_sparse/final_s2.pt --sparse
    python error_maps.py --models outputs/final_s0.pt                       # dense model, first model only
    python error_maps.py --models outputs/final_s*.pt --ensemble            # average the seeds first
    python error_maps.py --models ... --re 20000 88000 --signed --ylog --mirror --final

Options
  --sparse   read/write outputs_sparse/        --final   also allow the TEST Reynolds numbers
  --ensemble average predictions of all given models (default: first model only, like the report figures)
  --signed   plot (pred - ref) with a diverging colour map instead of |pred - ref|
  --ylog     logarithmic wall-normal axis (shows the viscous sublayer)     --mirror  show the full channel y in [-1, 1]
  --pct P    colour scale clipped at the P-th percentile of the error (default 99; the true max is printed in the title)
  --tau T    "significant" sign threshold = T x max|ref| of that field (default 0.02), so sign flips of tiny values
             near zero are not counted

Fields plotted (same normalisation as every table in the report):
  U/U_b ; V^ = V/(0.05 u_tau,est) ; P^ = p/(0.5 U_b^2) ; chi = nu~/nu (latent: never supervised)
Outputs (in <outputs>/error_maps/): errmap_Re<Re>.png, signmap_Re<Re>.png (only if any negatives), sign_report.txt/.csv
"""
import argparse
import os
import sys

if "--sparse" in sys.argv:
    os.environ["PINN_SPARSE"] = "1"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

import config as C
import data as D
import train as T

NAMES = [("U", "U/U_b"), ("V", "V̂ = V/(0.05 u_τ,est)"), ("P", "P̂ = p/(½U_b²)"), ("chi", "χ = ν̃/ν")]


def predict_all(models, es, ensemble):
    ms = models if ensemble else models[:1]
    return torch.stack([D.predict(m, es) for m in ms]).mean(0).cpu().numpy()


def fields(es, P):
    ref = np.stack([es.u.cpu().numpy(), es.v.cpu().numpy(), es.p.cpu().numpy(), es.chi.cpu().numpy()], 1)
    return ref, P


def sign_stats(ref, pred, tau_frac):
    tau = tau_frac * np.abs(ref).max()
    neg, pos = ref < -tau, ref > tau
    sig = neg | pos
    s = dict(tau=tau, ref_min=ref.min(), ref_max=ref.max(), pred_min=pred.min(), pred_max=pred.max(),
             frac_neg_ref=neg.mean(), frac_neg_pred=(pred < -tau).mean())
    s["neg_recall"] = float((pred[neg] < 0).mean()) if neg.any() else float("nan")        # ref negative -> model negative
    s["false_neg"] = float((pred[pos] < 0).mean()) if pos.any() else float("nan")         # ref positive -> model negative
    s["agree_sig"] = float((np.sign(pred[sig]) == np.sign(ref[sig])).mean()) if sig.any() else float("nan")
    return s, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--re", type=float, nargs="*", default=None)
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--ylog", action="store_true")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--tau", type=float, default=0.02)
    a = ap.parse_args()

    device, dtype = T.get_device(), torch.float32
    models = [T.load_model(p, device)[0] for p in a.models]
    res = a.re or (C.RE_TRAIN + C.RE_VAL + (C.RE_TEST if a.final else []))
    out = os.path.join(C.OUT_DIR, "error_maps")
    os.makedirs(out, exist_ok=True)
    lines, rows = [], []
    hdr = (f"SIGN ANALYSIS ({'ensemble of %d models' % len(models) if a.ensemble else 'first model only'}; "
           f"significant = |ref| > {a.tau} x max|ref|)\n"
           "neg_recall = of the points where the REFERENCE is significantly negative, fraction where the MODEL is negative;\n"
           "false_neg  = of the points where the reference is significantly positive, fraction where the model is negative;\n"
           "agree_sig  = fraction of significant points with the same sign.\n")
    lines.append(hdr)
    for Re in res:
        es = D.EvalSet(Re, device, dtype, final=a.final, full=True)
        pred = predict_all(models, es, a.ensemble)
        ref, pred = fields(es, pred)
        x, y = es.ref["x"][es.ix], es.ref["y"][es.iy]
        nx, ny = es.shape
        ytxt = "y/h"
        if a.mirror:
            y = np.concatenate([-y[::-1], y])
            ytxt = "y/h (mirrored; V is odd in y)"

        def grid(v, k):
            g = v[:, k].reshape(nx, ny)
            if a.mirror:
                sgn = -1.0 if k == 1 else 1.0
                g = np.concatenate([sgn * g[:, ::-1], g], axis=1)
            return g

        # ---------------- error maps (4 panels)
        fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
        for k, (key, lab) in enumerate(NAMES):
            ax = axs[k // 2, k % 2]
            
            # MULTIPLY BY 100 TO CONVERT TO PERCENTAGES
            e = (grid(pred, k) - grid(ref, k)) * 100.0
            
            ee = e if a.signed else np.abs(e)
            lim = np.percentile(np.abs(e), a.pct)
            kw = dict(norm=TwoSlopeNorm(0.0, -lim, lim), cmap="RdBu_r") if a.signed else dict(vmin=0, vmax=lim, cmap="viridis")
            im = ax.pcolormesh(x, y, ee.T, shading="auto", **kw)
            
            # RESTORED: symlog x-axis
            ax.set_xscale("symlog", linthresh=1.0)
            
            if a.ylog:
                ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("x/h (linear below 1, log above)")
            ax.set_ylabel(ytxt)
            
            # UPDATED TITLE TO INCLUDE (%)
            kind = "pred − ref (%)" if a.signed else "|pred − ref| (%)"
            ax.set_title(f"{lab}: {kind}   max |err| = {np.abs(e).max():.3g}, colour clipped at p{a.pct:g} = {lim:.3g}", fontsize=9)
            plt.colorbar(im, ax=ax)
            
        role = "TRAIN" if Re in C.RE_TRAIN else ("VAL" if Re in C.RE_VAL else "TEST")
        fig.suptitle(f"Re = {int(Re)} ({role}) — error maps over the whole domain", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"errmap_Re{int(Re)}.png"), dpi=130)
        plt.close()

        # ---------------- sign analysis
        lines.append(f"\nRe = {int(Re)} ({role})")
        lines.append(f"{'field':8s} {'ref min':>9s} {'ref max':>9s} {'model min':>10s} {'model max':>10s} {'% ref<-tau':>10s} {'% model<-tau':>12s} {'neg_recall':>10s} {'false_neg':>10s} {'agree_sig':>10s}")
        need_map = []
        for k, (key, lab) in enumerate(NAMES):
            s, tau = sign_stats(ref[:, k], pred[:, k], a.tau)
            lines.append(f"{key:8s} {s['ref_min']:9.3g} {s['ref_max']:9.3g} {s['pred_min']:10.3g} {s['pred_max']:10.3g} "
                         f"{100*s['frac_neg_ref']:10.2f} {100*s['frac_neg_pred']:12.2f} {s['neg_recall']:10.3f} {s['false_neg']:10.3f} {s['agree_sig']:10.3f}")
            rows.append(dict(Re=Re, field=key, **s))
            if s["frac_neg_ref"] > 0 or s["frac_neg_pred"] > 0:
                need_map.append((k, key, lab, tau))
        if need_map:
            fig, axs = plt.subplots(1, len(need_map), figsize=(6.2 * len(need_map), 4.4), squeeze=False)
            cm = ListedColormap(["#d9d9d9", "#fdf3c4", "#2ca25f", "#e34a33", "#fd8d3c"])
            for ax, (k, key, lab, tau) in zip(axs[0], need_map):
                r, p_ = grid(ref, k), grid(pred, k)
                cat = np.zeros(r.shape, int)
                cat[(r > tau) & (p_ >= 0)] = 1      # both non-negative
                cat[(r < -tau) & (p_ < 0)] = 2      # negative correctly predicted
                cat[(r < -tau) & (p_ >= 0)] = 3     # negative MISSED
                cat[(r > tau) & (p_ < 0)] = 4       # false negative
                im = ax.pcolormesh(x, y, cat.T, shading="auto", cmap=cm, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
                
                # RESTORED: symlog x-axis for sign maps
                ax.set_xscale("symlog", linthresh=1.0)
                
                ax.set_xlabel("x/h (linear below 1, log above)"); ax.set_ylabel(ytxt)
                ax.set_title(f"{lab}: sign agreement (|ref| > {tau:.3g})", fontsize=9)
            cb = plt.colorbar(im, ax=axs[0].tolist(), ticks=range(5), shrink=0.9)
            cb.ax.set_yticklabels(["|ref| ≈ 0", "both ≥ 0", "ref<0, model<0 (correct)", "ref<0, model≥0 (MISSED)", "ref>0, model<0 (false)"])
            plt.savefig(os.path.join(out, f"signmap_Re{int(Re)}.png"), dpi=130, bbox_inches="tight")
            plt.close()
        else:
            lines.append("   -> neither reference nor model has significantly negative values for any field (no sign map drawn)")
        print(f"Re={int(Re)} done", flush=True)

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "sign_report.txt"), "w").write(txt + "\n")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out, "sign_report.csv"), index=False)
    print("\nfigures + reports in", out)


if __name__ == "__main__":
    main()