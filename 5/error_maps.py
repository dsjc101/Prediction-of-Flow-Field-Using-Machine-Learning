"""
error_maps.py -- FULL-DOMAIN error analysis of trained PINNs against the SA reference field.

  python error_maps.py --regime dense  --tag final                 (TRAIN + VAL)
  python error_maps.py --regime sparse --tag final --final         (TEST Re only; once per model)
  python error_maps.py --regime dense  --tag final --roles train val --residuals   (adds PDE-residual maps)

Every reference cell (all stations x >= 0.05 R, all 400 wall-normal cells) is used -- no sub-sampling for
the error fields.  Residual maps use every 4th station / 4th cell (autograd is costly) and r >= 0.02.

Scales (global, Re-only -- identical to evaluate.py so the maps and the tables agree):
   U       : error / U_b
   V_hat   : error / (0.05 u_tau,est)   (V_hat = V / v_s)
   P_hat   : error / (0.5 U_b^2) units (P_hat is already P/(0.5 U_b^2)),  P pinned to 0 at x = L
   nu_tilde: error / u_tau,est   (R = 1)
   PDE residuals: in u_tau,est units (continuity: u_tau, momentum & SA: u_tau^2)

Outputs (outputs/):
   emap_signed_{regime}_{Re}.png    signed error maps (x log scale on top row, x linear below)
   emap_logabs_{regime}_{Re}.png    log10 |error| maps
   emap_resid_{regime}_{Re}.png     log10 |PDE residual| maps (with --residuals)
   emap_lines_{regime}_{roles}.png  rms error vs x, rms error vs y, Cf error vs x, CDF of |U error|
   emap_blocks_{regime}_{roles}.png + emap_stats_{regime}_{roles}.txt   where the error lives (x-y blocks)
"""
import argparse, os
import numpy as np
import torch
import config as C
import pinn as P
import data as D
import train as T
import evaluate as E

X_BINS = [0.05, 1.0, 3.0, 10.0, 30.0, 100.0, 160.01]
Y_BINS = [0.0, 0.02, 0.1, 0.3, 0.6, 1.0001]


def scales(Re):
    cf, ut, ret = C.est_np(Re)
    return dict(U=1.0, Vh=1.0, P=1.0, nt=float(ut), ut=float(ut))


def field_errors(model, Re, dev="cpu"):
    """returns x, y, dict of signed error arrays shape (nx, ny) in the scales above, plus the reference arrays."""
    d = D.load_ref(Re)
    ix = np.where(d["x"] >= C.X_DATA_MIN)[0]
    iy = np.arange(len(d["r"]))
    pts = D._pack(Re, ix, iy, dev)
    with torch.no_grad():
        o = P.predict(model, pts["x"], pts["r"], pts["Re"])
    nx, ny = len(ix), len(iy)
    vs = float(D.vs_np(Re))
    sc = scales(Re)
    ref = dict(U=pts["U"], Vh=pts["V"] / vs, P=pts["P"], nt=pts["nt"])
    pred = dict(U=o[:, 0], Vh=o[:, 1] / vs, P=o[:, 2], nt=o[:, 3])
    err = {k: ((pred[k] - ref[k]) / sc[k]).reshape(nx, ny).cpu().numpy() for k in ref}
    refm = {k: v.reshape(nx, ny).cpu().numpy() for k, v in ref.items()}
    return d["x"][ix], d["y"][iy], err, refm


def residual_fields(model, Re, dev="cpu", stride_x=4, stride_y=4):
    d = D.load_ref(Re)
    ix = np.where(d["x"] >= C.X_DATA_MIN)[0][::stride_x]
    iy = np.where(d["r"] >= C.R_AXIS_MIN)[0][::stride_y]
    X, R_ = np.meshgrid(d["x"][ix], d["r"][iy], indexing="ij")
    x = torch.tensor(X.ravel(), dtype=torch.float32, device=dev)
    r = torch.tensor(R_.ravel(), dtype=torch.float32, device=dev)
    Rr = torch.full_like(x, float(Re))
    out = {k: [] for k in ("cont", "momx", "momr", "sa")}
    for i in range(0, x.numel(), 4000):
        res = P.rans_sa_residuals(model, x[i:i + 4000], r[i:i + 4000], Rr[i:i + 4000])
        for k in out:
            out[k].append(res[k].detach().cpu().numpy())
    return d["x"][ix], d["y"][iy], {k: np.concatenate(v).reshape(len(ix), len(iy)) for k, v in out.items()}


NAMES = dict(U="U error / U_b", Vh="V_hat error", P="P_hat error", nt="nu_tilde error / u_tau")


def _sym_lim(a, q=99.0):
    return max(float(np.percentile(np.abs(a), q)), 1e-12)


def fig_signed(x, y, err, Re, tag, regime):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["U", "Vh", "P", "nt"]
    fig, axs = plt.subplots(2, 4, figsize=(18, 6.4), squeeze=False)
    for j, k in enumerate(keys):
        lim = _sym_lim(err[k])
        for i, xs in enumerate(["log", "lin"]):
            ax = axs[i][j]
            im = ax.pcolormesh(x, y, err[k].T, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            if xs == "log":
                ax.set_xscale("log")
            ax.set_xlabel("x/R" + (" (log)" if xs == "log" else " (linear)"))
            ax.set_ylabel("y/R  (0 = wall, 1 = axis)")
            ax.set_title(f"{NAMES[k]}   [colour limit ±{lim:.3g} = 99th pct]", fontsize=9)
            fig.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle(f"{regime.upper()} PINN - signed error (PINN - SA ref), Re = {Re}  ({C.ROLE_NOTE[Re]})", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_signed_{regime}_{Re}.png"), dpi=120)
    plt.close(fig)


def fig_logabs(x, y, err, Re, regime):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["U", "Vh", "P", "nt"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6), squeeze=False)
    for ax, k in zip(axs[0], keys):
        a = np.log10(np.abs(err[k]) + 1e-12)
        im = ax.pcolormesh(x, y, a.T, shading="auto", cmap="magma", vmin=-4, vmax=max(0.0, float(np.percentile(a, 99.9))))
        ax.set_xscale("log"); ax.set_xlabel("x/R (log)"); ax.set_ylabel("y/R")
        ax.set_title("log10 |" + NAMES[k] + "|", fontsize=9)
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle(f"{regime.upper()} - Re = {Re} ({C.ROLE_NOTE[Re]})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_logabs_{regime}_{Re}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_resid(x, y, res, Re, regime):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = dict(cont="continuity", momx="x-momentum", momr="r-momentum", sa="SA transport")
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6), squeeze=False)
    for ax, k in zip(axs[0], names):
        a = np.log10(np.abs(res[k]) + 1e-8)
        im = ax.pcolormesh(x, y, a.T, shading="auto", cmap="magma", vmin=-3, vmax=max(0.0, float(np.percentile(a, 99.9))))
        ax.set_xscale("log"); ax.set_xlabel("x/R (log)"); ax.set_ylabel("y/R")
        ax.set_title("log10 |" + names[k] + " residual| (u_tau units)", fontsize=9)
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle(f"{regime.upper()} - PDE residual, Re = {Re} ({C.ROLE_NOTE[Re]})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_resid_{regime}_{Re}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def block_stats(x, y, eU):
    """rms of the U error and share of total squared error per (x-bin, y-bin) block."""
    tot = np.sum(eU ** 2)
    rms = np.zeros((len(Y_BINS) - 1, len(X_BINS) - 1)); share = np.zeros_like(rms)
    for i in range(len(Y_BINS) - 1):
        my = (y >= Y_BINS[i]) & (y < Y_BINS[i + 1])
        for j in range(len(X_BINS) - 1):
            mx = (x >= X_BINS[j]) & (x < X_BINS[j + 1])
            sub = eU[np.ix_(mx, my)]
            rms[i, j] = 100 * np.sqrt(np.mean(sub ** 2)) if sub.size else np.nan
            share[i, j] = 100 * np.sum(sub ** 2) / tot if sub.size else 0.0
    return rms, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["dense", "sparse"], required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--roles", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="REQUIRED for TEST Re")
    ap.add_argument("--residuals", action="store_true", help="also compute PDE-residual maps (slower)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if "test" in a.roles and not a.final:
        raise SystemExit("TEST Reynolds numbers need the explicit --final flag.")
    if a.final:
        a.roles = ["test"]
    files = E.find_models(a.regime, a.tag)
    if not files:
        raise SystemExit(f"no models for {a.regime}/{a.tag}")
    re_list = []
    for r in a.roles:
        re_list += {"train": C.RE_TRAIN, "val": C.RE_VAL, "test": C.RE_TEST}[r]
    models = [T.load_model(f, a.device)[0].eval() for f in files]
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rtag = "+".join(a.roles)
    lines_txt = [f"=== {a.regime.upper()} full-domain error statistics ({len(models)} seed(s); maps use seed 0) ===",
                 "errors in global Re-only scales (see header of error_maps.py); percentages = 100 x error"]
    store = {}
    for Re in re_list:
        per_seed = []
        for k, m in enumerate(models):
            x, y, err, refm = field_errors(m, Re, a.device)
            per_seed.append(err)
            if k == 0:
                err0 = err
        store[Re] = (x, y, per_seed)
        fig_signed(x, y, err0, Re, rtag, a.regime)
        fig_logabs(x, y, err0, Re, a.regime)
        if a.residuals:
            xr, yr, res = residual_fields(models[0], Re, a.device)
            fig_resid(xr, yr, res, Re, a.regime)
        # statistics
        lines_txt.append(f"\n--- Re = {Re} ({C.ROLE_NOTE[Re]}) ---")
        for q in ("U", "Vh", "P", "nt"):
            e = err0[q]
            idx = np.unravel_index(np.argmax(np.abs(e)), e.shape)
            lines_txt.append(f"{q:>3}: rms {100*np.sqrt(np.mean(e**2)):8.3f}%   mean(signed) {100*np.mean(e):8.3f}%   "
                             f"p95|e| {100*np.percentile(np.abs(e),95):8.3f}%   p99|e| {100*np.percentile(np.abs(e),99):8.3f}%   "
                             f"max|e| {100*np.abs(e[idx]):8.3f}% at x={x[idx[0]]:.2f}R, y={y[idx[1]]:.4f}R")
        rms, share = block_stats(x, y, err0["U"])
        lines_txt.append("U-error rms [% of U_b] per block (rows: y bins from wall, cols: x bins):")
        lines_txt.append("   x bins: " + "  ".join(f"[{X_BINS[j]:g},{X_BINS[j+1]:.0f})" for j in range(len(X_BINS) - 1)))
        for i in range(len(Y_BINS) - 1):
            lines_txt.append(f"   y[{Y_BINS[i]:g},{min(Y_BINS[i+1],1):g}): " + "  ".join(f"{rms[i,j]:9.3f}" for j in range(rms.shape[1])))
        lines_txt.append("share of total squared U error [%] per block:")
        for i in range(len(Y_BINS) - 1):
            lines_txt.append(f"   y[{Y_BINS[i]:g},{min(Y_BINS[i+1],1):g}): " + "  ".join(f"{share[i,j]:9.2f}" for j in range(share.shape[1])))
        print(f"  Re={Re} done", flush=True)

    # ---- line figures over all Re -------------------------------------------------
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(re_list)))
    fig, axs = plt.subplots(2, 4, figsize=(19, 8))
    for c, Re in zip(cols, re_list):
        x, y, per_seed = store[Re]
        for q, ax in zip(("U", "Vh", "P", "nt"), axs[0]):
            rx = np.array([100 * np.sqrt(np.mean(s[q] ** 2, axis=1)) for s in per_seed])
            ax.semilogx(x, rx.mean(0), color=c, label=f"{Re}")
            ax.fill_between(x, rx.min(0), rx.max(0), color=c, alpha=0.15)
            ax.set_xlabel("x/R"); ax.set_title(f"{NAMES[q]}: rms over r, vs x  [%]", fontsize=9)
        for q, ax in zip(("U", "Vh", "P", "nt"), axs[1]):
            ry = np.array([100 * np.sqrt(np.mean(s[q] ** 2, axis=0)) for s in per_seed])
            ax.semilogx(np.maximum(y, 1e-4), ry.mean(0), color=c)
            ax.fill_between(np.maximum(y, 1e-4), ry.min(0), ry.max(0), color=c, alpha=0.15)
            ax.set_xlabel("y/R (log)"); ax.set_title(f"{NAMES[q]}: rms over x, vs y  [%]", fontsize=9)
    axs[0][0].legend(title="Re", fontsize=7)
    fig.suptitle(f"{a.regime.upper()} - where the error lives (mean over seeds, band = min-max over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_lines_{a.regime}_{rtag}.png"), dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for c, Re in zip(cols, re_list):
        x, y, per_seed = store[Re]
        e = np.abs(per_seed[0]["U"]).ravel() * 100
        s = np.sort(e)
        axs[1].semilogx(np.maximum(s, 1e-4), np.linspace(0, 1, len(s)), color=c, label=f"{Re}")
        d = D.load_ref(Re)
        m = d["x"] >= C.CF_X_MIN
        xt = torch.tensor(d["x"][m], dtype=torch.float32)
        for k, mod in enumerate(models):
            cfn = P.wall_shear_cf(mod, xt, torch.full_like(xt, float(Re))).detach().numpy()
            axs[0].semilogx(d["x"][m], 100 * (cfn - d["Cf"][m]) / d["Cf"][m], color=c, alpha=0.8 if k == 0 else 0.25)
    axs[0].axhline(0, color="k", lw=0.5); axs[0].set_xlabel("x/R"); axs[0].set_ylabel("Cf error [%] (signed, relative)")
    axs[0].set_title("wall-friction error along the pipe (each seed)")
    axs[1].set_xlabel("|U error| [% of U_b]"); axs[1].set_ylabel("fraction of points below"); axs[1].legend(title="Re", fontsize=7)
    axs[1].set_title("cumulative distribution of |U error| (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_cf_cdf_{a.regime}_{rtag}.png"), dpi=120)
    plt.close(fig)

    # ---- block heat-map averaged over Re ------------------------------------------
    sh = np.mean([block_stats(store[Re][0], store[Re][1], store[Re][2][0]["U"])[1] for Re in re_list], axis=0)
    rm = np.mean([block_stats(store[Re][0], store[Re][1], store[Re][2][0]["U"])[0] for Re in re_list], axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for ax, A, ttl in [(axs[0], rm, "rms U error [% of U_b]"), (axs[1], sh, "share of total squared U error [%]")]:
        im = ax.imshow(A, origin="lower", cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(X_BINS) - 1)); ax.set_xticklabels([f"{X_BINS[j]:g}-{X_BINS[j+1]:.0f}" for j in range(len(X_BINS) - 1)])
        ax.set_yticks(range(len(Y_BINS) - 1)); ax.set_yticklabels([f"{Y_BINS[i]:g}-{min(Y_BINS[i+1],1):g}" for i in range(len(Y_BINS) - 1)])
        ax.set_xlabel("x/R bin"); ax.set_ylabel("y/R bin (wall -> axis)"); ax.set_title(ttl)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i,j]:.2f}" if A[i, j] < 10 else f"{A[i,j]:.0f}", ha="center", va="center", fontsize=8,
                        color="w" if A[i, j] > 0.6 * np.nanmax(A) else "k")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"{a.regime.upper()} - block decomposition of the U error (mean over the Re listed)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"emap_blocks_{a.regime}_{rtag}.png"), dpi=120)
    plt.close(fig)

    txt = "\n".join(lines_txt)
    with open(os.path.join(C.OUT_DIR, f"emap_stats_{a.regime}_{rtag}.txt"), "w") as f:
        f.write(txt)
    print(txt)
    print(f"\nsaved outputs/emap_*_{a.regime}*.png and emap_stats_{a.regime}_{rtag}.txt")


if __name__ == "__main__":
    main()