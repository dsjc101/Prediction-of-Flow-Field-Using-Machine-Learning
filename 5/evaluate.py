"""
evaluate.py -- error tables and figures for trained models (mean +- std over seeds).

  python evaluate.py --regime dense  --tag final                 (TRAIN + VAL tables; safe to run any time)
  python evaluate.py --regime sparse --tag final --final         (TEST Re ONLY -- run once per model!)

relL2(q) = 100 * sqrt(mean_i (q_pred - q_ref)^2) / scale(Re),  scale = U_b for U, 0.5 U_b^2 for P_hat,
0.05 u_tau,est for V_hat (global Re-only scales, never local reference values).  Cf error is RELATIVE, x >= 1 R.
"""
import argparse, glob, json, os
import numpy as np
import torch
import config as C
import pinn as P
import data as D
import train as T


def rl2(e, mask=None):
    e = e if mask is None else e[mask]
    return float(100 * torch.sqrt(torch.mean(e ** 2))) if e.numel() else float("nan")


def region_masks(x):
    m = {"all": torch.ones_like(x, dtype=torch.bool),
         "x<3R": x < 3.0, "inlet x<10R": x < 10.0, "developed x>=0.9L": x >= 0.9 * C.L}
    b = torch.zeros_like(x, dtype=torch.bool)
    for a, c in C.HOLDOUT_BANDS:
        b |= (x >= a) & (x <= c)
    m["hold-out bands"] = b
    return m


def metrics_for(model, Re, dev="cpu"):
    model.eval()
    pts, cf, ix, iy = D.eval_set(Re, dev)
    o = P.predict(model, pts["x"], pts["r"], pts["Re"])
    vs = float(D.vs_np(Re))
    e = dict(U=o[:, 0] - pts["U"], Vh=(o[:, 1] - pts["V"]) / vs, P=o[:, 2] - pts["P"])
    out = {}
    for name, m in region_masks(pts["x"]).items():
        for q in ("U", "Vh", "P"):
            out[f"{q}|{name}"] = rl2(e[q], m)
    cfn = P.wall_shear_cf(model, cf["x"], cf["Re"])
    ec = (cfn - cf["cf_ref"]) / cf["cf_ref"]
    for name, m in region_masks(cf["x"]).items():
        out[f"Cf|{name}"] = rl2(ec, m)
    for q in ("U", "Vh", "P", "Cf"):
        out[q] = out[f"{q}|all"]
    # fresh-point residual (independent of the training stream)
    xf, rf, Ref = D.fresh_residual_points(Re, dev=dev)
    res = P.rans_sa_residuals(model, xf, rf, Ref)
    for k, v in res.items():
        out[f"res_{k}"] = float(torch.sqrt(torch.mean(v.detach() ** 2)))
    return out


def aggregate(per_seed):
    """list of dict -> dict of (mean, std)"""
    keys = per_seed[0].keys()
    return {k: (float(np.mean([d[k] for d in per_seed])), float(np.std([d[k] for d in per_seed]))) for k in keys}


def fmt(ms, nd=2):
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def find_models(regime, tag, seeds=None):
    files = sorted(glob.glob(os.path.join(T.model_dir(regime), f"{tag}_seed*.pt")))
    if seeds is not None:
        files = [f for f in files if int(f.split("seed")[-1].split(".")[0]) in seeds]
    return files


def evaluate_models(files, re_list, dev="cpu", verbose=True):
    res = {Re: [] for Re in re_list}
    for f in files:
        m, _ = T.load_model(f, dev)
        for Re in re_list:
            res[Re].append(metrics_for(m, Re, dev))
        if verbose:
            print(f"  evaluated {os.path.basename(f)}", flush=True)
    return {Re: aggregate(v) for Re, v in res.items()}


def print_tables(agg, regime, roles):
    lines = []
    lines.append(f"\n=== {regime.upper()} model: relL2 %  (mean ± std over seeds) ===")
    lines.append(f"{'role':<18}{'Re':>8}  {'U':>14}{'V_hat':>16}{'P_hat':>14}{'Cf':>14}")
    for Re in sorted(agg):
        a = agg[Re]
        lines.append(f"{C.ROLE_NOTE[Re]:<18}{Re:>8}  {fmt(a['U']):>14}{fmt(a['Vh'], 1):>16}{fmt(a['P']):>14}{fmt(a['Cf']):>14}")
    # generalisation ratios
    lines.append(f"\n--- region breakdown (mean over the Re listed above), relL2 %: U / V_hat / P_hat / Cf ---")
    for name in ["all", "x<3R", "inlet x<10R", "developed x>=0.9L", "hold-out bands"]:
        v = [np.mean([agg[Re][f"{q}|{name}"][0] for Re in agg]) for q in ("U", "Vh", "P", "Cf")]
        lines.append(f"{name:<22}{v[0]:9.2f}{v[1]:10.1f}{v[2]:9.2f}{v[3]:9.2f}")
    lines.append(f"\n--- fresh-point PDE residual rms (u_tau units): continuity / x-mom / r-mom / SA ---")
    for Re in sorted(agg):
        a = agg[Re]
        lines.append(f"{C.ROLE_NOTE[Re]:<18}{Re:>8}  {a['res_cont'][0]:9.2f}{a['res_momx'][0]:9.2f}{a['res_momr'][0]:9.2f}{a['res_sa'][0]:9.2f}")
    txt = "\n".join(lines)
    print(txt)
    return txt


def gen_ratios(agg_all):
    """mean error per role / mean TRAIN error"""
    out = []
    train = [Re for Re in agg_all if C.ROLE[Re] == "TRAIN"]
    if not train:
        return ""
    out.append("\n--- generalisation ratios (role mean / TRAIN mean) for U, P_hat, Cf ---")
    for q in ("U", "P", "Cf"):
        base = np.mean([agg_all[Re][q][0] for Re in train])
        s = f"{q:>3}: TRAIN {base:.2f}"
        for role, sel in [("VAL", lambda r: C.ROLE[r] == "VAL"), ("TEST interp", lambda r: C.ROLE_NOTE[r] == "TEST interp"),
                          ("TEST extrap", lambda r: C.ROLE_NOTE[r].startswith("TEST extrap"))]:
            rs = [Re for Re in agg_all if sel(Re)]
            if rs:
                v = np.mean([agg_all[Re][q][0] for Re in rs])
                s += f" | {role} {v:.2f} ({v / base:.2f})"
        out.append(s)
    txt = "\n".join(out)
    print(txt)
    return txt


# ----------------------------------------------------------------------------------
# figures (seed-0 model)
# ----------------------------------------------------------------------------------
def figures(model, re_list, tag, dev="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = []
    for want in [20_000, 14_000, 28_000, 120_000, 10_000, 56_000, 96_000]:
        if want in re_list and len(pick) < 4:
            pick.append(want)
    for Re in re_list:
        if len(pick) < 4 and Re not in pick:
            pick.append(Re)
    if not pick:
        return
    fig, axs = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.4), squeeze=False)
    for ax, Re in zip(axs[0], pick):
        pts, cf, ix, iy = D.eval_set(Re, dev)
        o = P.predict(model, pts["x"], pts["r"], pts["Re"])
        err = (o[:, 0] - pts["U"]).abs().reshape(len(ix), len(iy)).cpu().numpy()
        d = D.load_ref(Re)
        im = ax.pcolormesh(d["x"][ix], d["y"][iy], err.T, shading="auto", vmin=0, vmax=0.05, cmap="viridis")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(f"{C.ROLE_NOTE[Re]}  Re={Re}")
        ax.set_xlabel("x/R"); ax.set_ylabel("y/R  (0 = wall, 1 = axis)")
    fig.colorbar(im, ax=axs[0].tolist(), label="|U_PINN - U_ref|/U_b (clipped 0.05)")
    fig.savefig(os.path.join(C.OUT_DIR, f"figure_u_error_maps_{tag}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(len(pick), 4, figsize=(15, 3.2 * len(pick)), squeeze=False)
    for row, Re in zip(axs, pick):
        d = D.load_ref(Re)
        x_ref = d["x"]
        for c, fr in enumerate([0.05, 0.25, 0.9]):
            k = int(np.argmin(np.abs(x_ref - fr * C.L)))
            r_ = torch.tensor(d["r"], dtype=torch.float32, device=dev)
            xx = torch.full_like(r_, float(x_ref[k])); RR = torch.full_like(r_, float(Re))
            o = P.predict(model, xx, r_, RR)[:, 0].cpu().numpy()
            row[c].plot(d["U"][k], d["y"], "k-", label="SA reference")
            row[c].plot(o, d["y"], "--", color="tab:orange", label="PINN")
            row[c].set_title(f"Re={Re} ({C.ROLE_NOTE[Re]})  x={x_ref[k]:.1f}R"); row[c].set_xlabel("U/U_b"); row[c].set_ylabel("y/R")
        m = x_ref >= 0.2
        xt = torch.tensor(x_ref[m], dtype=torch.float32, device=dev)
        cfn = P.wall_shear_cf(model, xt, torch.full_like(xt, float(Re))).cpu().numpy()
        row[3].semilogx(x_ref[m], d["Cf"][m], "k-"); row[3].semilogx(x_ref[m], cfn, "--", color="tab:orange")
        row[3].set_xlabel("x/R"); row[3].set_ylabel("Cf"); row[3].set_title("wall friction")
    axs[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, f"figure_profiles_cf_{tag}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["dense", "sparse"], required=True)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--roles", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="REQUIRED to evaluate TEST Re (do this once per model)")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if "test" in a.roles and not a.final:
        raise SystemExit("TEST Reynolds numbers need the explicit --final flag (evaluate once per model).")
    if a.final:
        a.roles = ["test"]
    files = find_models(a.regime, a.tag, a.seeds)
    if not files:
        raise SystemExit(f"no models for {a.regime}/{a.tag} in {T.model_dir(a.regime)}")
    re_list = []
    for r in a.roles:
        re_list += {"train": C.RE_TRAIN, "val": C.RE_VAL, "test": C.RE_TEST}[r]
    print(f"{len(files)} model(s); Re = {re_list}")
    agg = evaluate_models(files, re_list, a.device)
    txt = print_tables(agg, a.regime, a.roles)
    txt += gen_ratios(agg) or ""
    name = f"eval_{a.regime}_{a.tag}_{'+'.join(a.roles)}"
    with open(os.path.join(C.OUT_DIR, name + ".json"), "w") as f:
        json.dump({str(k): v for k, v in agg.items()}, f, indent=1)
    with open(os.path.join(C.OUT_DIR, name + ".txt"), "w") as f:
        f.write(txt)
    if not a.no_figures:
        m0, _ = T.load_model(files[0], a.device)
        figures(m0, re_list, f"{a.regime}_{a.tag}_{'+'.join(a.roles)}", a.device)
    print(f"\nsaved outputs/{name}.json/.txt (+ figures)")


if __name__ == "__main__":
    main()
