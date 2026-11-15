"""
evaluate_global.py -- Entire-(x,r)-domain accuracy for ALL THREE outputs
(u, v, p): (1) a single global relative/normalized-error number per Re per
field (whole-field norm ratio), (2) a spatial map of LOCAL pointwise
error/deviation over the whole (x,r) plane, per field.

------------------------------------------------------------------------
METRIC DEFINITIONS -- see evaluate.py's module docstring for the full
derivation of relative_l2 vs normalized_l2 and why each field uses which.
Recap, applied at the WHOLE-DOMAIN scale (flattened over the entire
500 x-stations by 81 r-points reference grid, not one station):

  u : GLOBAL  = self-normalized relative L2 over the whole field
                (relative_l2) -- u's own field norm never vanishes.
      LOCAL   = pointwise |u_pred-u_ref| normalized by U_mean(Re) (fixed
                scale, not the local u_ref value) -- see "SECOND FIX"
                note below for why the local value cannot be used as its
                own denominator (it is allowed to be exactly zero at the
                wall/inlet-corner).

  v : GLOBAL  = scale-normalized L2 by U_mean(Re) -- NOT self-normalized.
                v -> 0 as the flow approaches full development, and for
                most Re in this sweep (especially the low-Re cases,
                where Le(Re) << L) MOST of the domain by area is already
                past that point, so ||v_ref||_2 over the WHOLE field can
                be just as small as it is locally -- self-normalizing
                here reproduces the same blown-up-percentage artifact
                evaluate.py's docstring documents for the per-station
                developing-region check (100-260% "errors" that were
                pure normalization artifacts once checked against the
                fully-developed station, where v's true error is ~1-4%).
      LOCAL   = pointwise |v_pred-v_ref| normalized by the SAME U_mean(Re)
                scale (consistent with the global number above, and with
                u/v being the same kind of quantity, a velocity
                component).

  p : GLOBAL and LOCAL both use p_scale(Re) = model.p_scale(Re) =
      0.5*U_mean(Re)^2 (Reynolds pressure-coefficient convention,
      Cp=p/(0.5 U^2)) instead of self-normalizing, because the reference
      p field is p(x) only (the parabolized Hornbeck solver's own
      assumption -- see entrance_solver.py's docstring) broadcast across
      r, and it passes through the outlet gauge p(x=L)=0 by construction,
      so a self-normalized ratio would be undefined/unstable near the
      outlet for the same reason given below for the LOCAL u/v map.

------------------------------------------------------------------------
SECOND FIX: eps-regularizing the LOCAL denominator (an earlier version of
this file) was the wrong fix and still produced ~5000% "max errors."
------------------------------------------------------------------------
Why: at the inlet corner (x=0, r=R), entrance_solver.py's reference data
has a genuine step discontinuity -- every point along the inlet row is
set to U0 EXCEPT the single last grid point, forced to exactly 0
(no-slip). A continuous network cannot reproduce a discontinuity, so
right at that one corner pixel it predicts something close to U0 where
the reference is exactly 0. The ABSOLUTE error there is genuinely
~U0 -- not tiny -- so flooring the denominator at eps=2% of U0 just
turns "true error ~U0" into "~U0 / (0.02*U0) = 50x = 5000%". A bigger
eps would not fix this: the problem was never really the floor, it was
dividing by a LOCAL reference value at all, when that local value is
allowed to be exactly zero by construction.

FIX: stop normalizing pointwise error by the LOCAL reference value.
Normalize by a fixed, Re-characteristic scale instead -- U_mean(Re) for
u and v, p_scale(Re) for p -- a single, always-nonzero, well-defined
scale, so no eps/floor hack is needed at all:

    err_pct(x,r) = 100 * |field_pred(x,r) - field_ref(x,r)| / scale(Re)

This is standard practice for reporting pointwise error in wall-bounded
flows specifically because it stays well-defined at the wall/no-slip
line (or, for p, at the outlet gauge) where the local field itself is
zero (see e.g. the discussion of normalizing error/residual by a
characteristic problem scale rather than a local, possibly-vanishing
quantity in Roache, P.J. (1998), Verification and Validation in
Computational Science and Engineering, Hermosa Publishers). Under this
metric the inlet-corner discontinuity now reads as what it actually is:
an absolute mismatch of about one U_mean, i.e. ~100%, not ~5000%.

THE GLOBAL (whole-field) u NUMBER WAS NEVER AFFECTED BY EITHER BUG: it's
a ratio of two full-field norms computed before any division happens
pointwise, so it stayed reliable (~2-3%) through both versions of this
file -- see relative_l2 below. v's global number is the same kind of
ratio (also unaffected); p's global number uses p_scale(Re) throughout
(never self-normalized), for the reason given above.
------------------------------------------------------------------------
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import PipePINN, u_ref, p_scale

FIELDS = ["u", "v", "p"]


def relative_l2(pred, exact):
    """100 * ||pred-exact||_2 / ||exact||_2, whole flattened field."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (np.sqrt(np.mean(exact ** 2)) + 1e-12)


def normalized_l2(pred, exact, scale):
    """100 * ||pred-exact||_2 / scale, scale a fixed Re-characteristic constant."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (abs(scale) + 1e-12)


def plot_entire_domain_errors():
    print("Initializing Entire-Domain Error Evaluation (axisymmetric pipe, u/v/p)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PipePINN(n_hidden=4, width=80).to(device)
    checkpoint_path = "outputs/pipe_pinn_hybrid.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} missing. Run train_hybrid.py first.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_Re = sorted(cfg.ALL_RE_FOR_EVAL)
    train_Re = cfg.RE_TRAIN

    data_path = "outputs/reference_data.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError("reference_data.npz missing. Run generate_data.py first.")

    ref_data = np.load(data_path)

    role_of = {**{r: "TRAIN" for r in train_Re}, 300.0: "INTERP (unseen)", 600.0: "EXTRAP (unseen)"}
    colors = {"TRAIN": "tab:blue", "INTERP (unseen)": "tab:green", "EXTRAP (unseen)": "tab:red"}

    # global_errs[field][Re] = whole-field error number; local_stats[field][Re] = (max_err, loc, area_good)
    global_errs = {f: {} for f in FIELDS}
    local_stats = {f: {} for f in FIELDS}

    figs_spatial = {}
    axes_spatial = {}
    for f in FIELDS:
        fig, axes = plt.subplots(len(all_Re), 1, figsize=(10, 2.5 * len(all_Re)))
        if len(all_Re) == 1:
            axes = [axes]
        scale_note = "U_mean(Re)" if f in ("u", "v") else "p_scale(Re)=0.5*U_mean(Re)^2"
        fig.suptitle(f"Spatial Error Distribution (x-r plane), field={f}: "
                     f"|{f}_pred - {f}_ref| as % of {scale_note}", fontsize=12, y=0.995)
        figs_spatial[f] = fig
        axes_spatial[f] = axes

    for i, Re in enumerate(all_Re):
        x_ref = ref_data[f"x_{Re}"]
        r_ref = ref_data[f"r_{Re}"]
        U_exact = ref_data[f"U_{Re}"]
        V_exact = ref_data[f"V_{Re}"]
        P_exact_1d = ref_data[f"P_{Re}"] - ref_data[f"P_{Re}"][-1]  # outlet gauge, matches
                                                                     # train_hybrid.py's data-loss target
        P_exact = np.broadcast_to(P_exact_1d[:, None], U_exact.shape)  # p=p(x) only (parabolized ref.)

        X, Rg = np.meshgrid(x_ref, r_ref, indexing='ij')
        x_flat = X.flatten()
        r_flat = Rg.flatten()

        x_tensor = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, 1).to(device)
        r_tensor = torch.tensor(r_flat, dtype=torch.float32).reshape(-1, 1).to(device)
        re_tensor = torch.full_like(x_tensor, float(Re)).to(device)

        with torch.no_grad():
            u, v, p = model(x_tensor, r_tensor, re_tensor)
            u_pred = u.cpu().numpy().reshape(X.shape)
            v_pred = v.cpu().numpy().reshape(X.shape)
            p_pred = p.cpu().numpy().reshape(X.shape)

        preds = dict(u=u_pred, v=v_pred, p=p_pred)
        exacts = dict(u=U_exact, v=V_exact, p=P_exact)
        Umean = u_ref(Re)
        pscl = p_scale(Re)
        scales = dict(u=Umean, v=Umean, p=pscl)

        role = role_of[Re]
        weights = Rg  # area weight ~ r dr d(theta) for an annulus at radius r

        for f in FIELDS:
            pred_f, exact_f, scale_f = preds[f], exacts[f], scales[f]

            # ---- (1) GLOBAL: single ratio of two full-field norms ----
            # u self-normalizes (never near-zero anywhere in this problem).
            # v and p do NOT self-normalize: for v, most of the domain is
            # close to fully developed (v->0) for every Re, especially low
            # Re where Le << L, so ||v_ref|| over the WHOLE field can be
            # tiny too -- an earlier version of this file self-normalized
            # v and got the same kind of blown-up numbers (100-260%) that
            # evaluate.py's docstring documents for the per-station check;
            # p never self-normalizes for the reason given in the module
            # docstring (gauge passes through 0 at the outlet).
            if f == "u":
                g_err = relative_l2(pred_f, exact_f)
            else:
                g_err = normalized_l2(pred_f, exact_f, scale_f)
            global_errs[f][Re] = g_err

            # ---- (2) LOCAL (pointwise) error, normalized by the fixed scale ----
            err_map_pct = 100.0 * np.abs(pred_f - exact_f) / (abs(scale_f) + 1e-12)
            max_err = err_map_pct.max()
            max_idx = np.unravel_index(np.argmax(err_map_pct), err_map_pct.shape)
            max_loc_x, max_loc_r = X[max_idx], Rg[max_idx]
            good = err_map_pct < 5.0
            area_good_frac = 100.0 * np.sum(weights[good]) / np.sum(weights)
            local_stats[f][Re] = (max_err, max_loc_x, max_loc_r, area_good_frac)

            print(f"Re={Re:<4.0f} ({role:<6}) [{f}] | whole-field error={g_err:.3f}%  | "
                  f"max local |err|/scale={max_err:.1f}% at x={max_loc_x:.3f} m, r={max_loc_r:.4f} m "
                  f"(R={cfg.R} m)  | area with |err|/scale<5%: {area_good_frac:.1f}%")

            axg = axes_spatial[f][i]
            c = axg.contourf(X, Rg, err_map_pct, levels=np.linspace(0, 20, 40), cmap='magma', extend='max')
            axg.set_title(f"Re = {Re} ({role}) - global={g_err:.2f}%  "
                          f"- max|err|/scale={max_err:.0f}% @ x={max_loc_x:.2f} m  "
                          f"- {area_good_frac:.0f}% of area <5%", fontsize=10)
            axg.set_ylabel("r [m]")
            figs_spatial[f].colorbar(c, ax=axg, label=f"|{f} err| / scale [%]")

    for f in FIELDS:
        axes_spatial[f][-1].set_xlabel("x [m]")
        figs_spatial[f].tight_layout()
        spatial_path = f"outputs/entire_domain_heatmaps_{f}.png"
        figs_spatial[f].savefig(spatial_path, dpi=300)
        plt.close(figs_spatial[f])
        print(f"Saved spatial error heatmaps ({f}) to {spatial_path}")

    # ---- metrics_global.csv, all 3 fields ----
    with open("outputs/metrics_global.csv", "w") as f_csv:
        f_csv.write("Re,tag,u_global_relL2_pct,v_global_normL2_pct,p_global_normL2_pct,"
                    "u_area_frac_err_lt_5pct,v_area_frac_err_lt_5pct,p_area_frac_err_lt_5pct\n")
        for Re in all_Re:
            f_csv.write(f"{Re},{role_of[Re]},"
                        f"{global_errs['u'][Re]:.4f},{global_errs['v'][Re]:.4f},{global_errs['p'][Re]:.4f},"
                        f"{local_stats['u'][Re][3]:.2f},{local_stats['v'][Re][3]:.2f},{local_stats['p'][Re][3]:.2f}\n")
    print("Saved outputs/metrics_global.csv")

    # ---- global-error-vs-Re scatter, one subplot per field ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for col, f in enumerate(FIELDS):
        ax = axes[col]
        for Re in all_Re:
            ax.scatter([Re], [global_errs[f][Re]], color=colors[role_of[Re]], s=120,
                       label=role_of[Re], zorder=3)
        xs_sorted = sorted(all_Re)
        ax.plot(xs_sorted, [global_errs[f][Re] for Re in xs_sorted], linestyle='--', color='gray', zorder=1)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('Reynolds Number (Re)', fontsize=11)
        metric_name = ("Relative L2" if f == "u" else
                       "Normalized L2 (by U_mean)" if f == "v" else
                       "Normalized L2 (by p_scale)")
        ax.set_ylabel(f'Global {f} {metric_name} Error [%]', fontsize=11)
        ax.set_title(f'Entire-domain accuracy vs. Re: {f}', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.suptitle('Entire Domain Accuracy Across Reynolds Sweep (Axisymmetric Pipe) -- u, v, p', fontsize=13)
    fig.tight_layout()
    scatter_path = "outputs/entire_domain_error_vs_Re.png"
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)
    print(f"Saved global error scatter plot to {scatter_path}")


if __name__ == "__main__":
    main_dir = "outputs"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    plot_entire_domain_errors()