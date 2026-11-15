"""
evaluate_global.py -- Whole-domain error checks (global error-vs-Re curve
+ spatial error heatmaps) for ALL THREE predicted fields (u, v, p).

Uses the SAME error metric as evaluate.py (see that module's docstring
for the full derivation and citation trail -- Roache, P.J. (1998),
Verification and Validation in Computational Science and Engineering,
Hermosa Publishers): a global, solution-independent reference scale per
Re, never the local/exact field value, so the metric stays well-defined
even where a field is legitimately near zero (v everywhere once
fully-developed; any field right at the walls).

    relL2(q)  = 100 * sqrt(mean_(x,y) (q_pred - q_exact)^2) / scale(Re)
    scale(Re) = U0(Re)          for q in {u, v}
    scale(Re) = 0.5*U0(Re)^2    for q = p

POINTS CONSIDERED: every (x, y) grid point of the Bodoia-Osterle
solver's own output grid for that Re (nx streamwise stations x ny
wall-normal points -- the full field, not a subsample), so "global"
here is literal: one number summarizing agreement over the entire
simulated domain, using the same relL2 formula evaluate.py uses
station-by-station.

Produces:
    outputs/entire_domain_error_vs_Re_{u,v,p}.png  -- one global relL2 (%)
                                                       per Re, per field
    outputs/entire_domain_heatmap_{u,v,p}.png       -- spatial error,
                                                       normalized by
                                                       scale(Re), shared
                                                       colorbar across all
                                                       Re so panels are
                                                       directly comparable
    outputs/metrics_full_domain.csv                 -- all three fields
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import ChannelPINN
from evaluate import ALL_RE, TAGS, COLORS, FIELDS, OUT, field_scale, load_reference

DEVICE = "cpu"


def load_model():
    model = ChannelPINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(f"{OUT}/channel_pinn_hybrid.pt", map_location=DEVICE))
    model.eval()
    return model


def predict_grid(model, X, Y, Re_val):
    """Returns (U, V, P) physical fields, each reshaped to X.shape."""
    x = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1)
    Re = torch.full_like(x, float(Re_val))
    with torch.no_grad():
        u, v, p = model(x, y, Re)
    return (u.numpy().reshape(X.shape), v.numpy().reshape(X.shape), p.numpy().reshape(X.shape))


def global_relative_l2(pred, exact, scale):
    """Whole-field version of evaluate.relL2 -- same formula, mean taken
    over the entire (x, y) grid instead of one streamwise station."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (scale + 1e-12)


def main():
    model = load_model()
    ref = np.load(f"{OUT}/reference_data.npz")

    global_rows = {f: [] for f in FIELDS}       # (Re, tag, relL2)
    local_err_fields = {f: {} for f in FIELDS}  # Re -> (x, y, err_field)

    for Re in ALL_RE:
        x_ref, y_ref, U_ref, V_ref, P_ref = load_reference(ref, Re)
        X, Y = np.meshgrid(x_ref, y_ref, indexing="ij")
        U_pred, V_pred, P_pred = predict_grid(model, X, Y, Re)
        P_ref_grid = np.broadcast_to(P_ref[:, None], X.shape)

        exact = {"u": U_ref, "v": V_ref, "p": P_ref_grid}
        pred = {"u": U_pred, "v": V_pred, "p": P_pred}

        for f in FIELDS:
            scale = field_scale(f, Re)
            err_global = global_relative_l2(pred[f], exact[f], scale)
            global_rows[f].append((Re, TAGS[Re], err_global))
            # pointwise field, normalized by the SAME global per-Re scale
            # (not the local field value -- see module docstring)
            local_err_fields[f][Re] = (x_ref, y_ref, 100.0 * np.abs(pred[f] - exact[f]) / (scale + 1e-12))

    # ---- global error-vs-Re scatter, one figure per field -------------
    for f in FIELDS:
        plt.figure(figsize=(6, 4.5))
        for Re, tag, err in global_rows[f]:
            plt.scatter([Re], [err], c=COLORS[tag], label=tag, s=70, zorder=3)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # Linear scale as required by report guidelines (not log)
        plt.xscale("linear")
        plt.xlabel("Re")
        plt.ylabel(f"Global {f} relative L2 error [%]  (whole-field norm)")
        plt.title(f"Whole-domain accuracy across the Re sweep -- field: {f}")
        plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(f"{OUT}/entire_domain_error_vs_Re_{f}.png", dpi=150)
        plt.close()

    # ---- spatial heatmaps, one figure per field, shared colorbar scale --
    for f in FIELDS:
        all_local_max = max(np.percentile(v[2], 99) for v in local_err_fields[f].values())
        fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(9, 2.6 * len(ALL_RE)), sharex=True)
        for ax, Re in zip(axes, ALL_RE):
            x_ref, y_ref, err_field = local_err_fields[f][Re]
            im = ax.contourf(x_ref, y_ref, err_field.T, levels=20, vmin=0, vmax=all_local_max, cmap="inferno")
            ax.set_title(f"Re = {Re:.0f} ({TAGS[Re]}) -- max: {err_field.max():.2f}%", fontsize=10)
            ax.set_ylabel("y [m]")
            fig.colorbar(im, ax=ax, label=f"|{f}_pred - {f}_ref| / scale(Re) [%]")
        axes[-1].set_xlabel("x [m]")
        fig.suptitle(f"Spatial error, field {f}, normalized by the global reference scale(Re) "
                     "(not the local field value -- see module docstring)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/entire_domain_heatmap_{f}.png", dpi=140)
        plt.close(fig)

    with open(f"{OUT}/metrics_full_domain.csv", "w") as fh:
        fh.write("field,Re,tag,global_relL2_pct\n")
        for f in FIELDS:
            for Re, tag, err in global_rows[f]:
                fh.write(f"{f},{Re},{tag},{err:.4f}\n")

    print("Global whole-domain relative L2 error by Re, by field:")
    for f in FIELDS:
        print(f"-- field: {f} --")
        for Re, tag, err in global_rows[f]:
            print(f"  Re={Re:6.0f} [{tag:16s}] relL2={err:.3f}%")
    print(f"\nSaved figures + metrics to {OUT}/")


if __name__ == "__main__":
    main()