"""
evaluate.py -- Evaluates the merged hybrid PINN's THREE predicted fields
(streamwise velocity u, cross-stream velocity v, kinematic pressure p)
against:

  1. The Bodoia & Osterle (1961) developing-flow reference
     (entrance_solver.py), at four stations spanning each Re's own
     development region (x = 5%, 25%, 50%, 90% of Le(Re)) -- for every Re
     including the two held-out ones (Re=300 interpolation, Re=600
     extrapolation).
  2. The exact fully-developed closed-form solution, far downstream of
     each Re's development length:
        u : Poiseuille profile,      config.poiseuille_profile  (White 2006)
        v : identically zero,        config.poiseuille_v        (White 2006)
        p : linear pressure drop,    config.poiseuille_pressure (White 2006)
     -- an independent theory-level check, not derived from the same
     numerical solver as (1).
  3. A "not memorization" generalization check (as before): TRAIN Re
     (direct data supervision everywhere) vs. INTERP/EXTRAP Re (no data
     supervision anywhere -- only the PDE residual and BCs ever touched
     those two Re values), now computed for all three fields.

================================================================================
ERROR METRIC -- exact definition, and why
================================================================================
For a field q in {u, v, p} at a chosen streamwise station x (or over the
whole 2-D grid, for evaluate_global.py), the error is

    relL2(q)  =  100 * sqrt( mean_i (q_pred_i - q_exact_i)^2 )  /  scale(Re)

where the mean is taken over EVERY wall-normal grid point i at that
station (all ny points spanning y in [-h, h] from the Bodoia-Osterle
solver's own grid -- i.e. the full cross-channel profile, not a single
sample point, so the metric reflects the whole profile SHAPE, not one
lucky/unlucky point), and

    scale(Re) = U0(Re) = config.inlet_velocity(Re)        for q in {u, v}
    scale(Re) = q(Re)  = model.p_scale(Re) = 0.5*U0(Re)^2  for q = p

scale(Re) is a GLOBAL reference derived only from the Reynolds-number
definition -- never from the local/exact field being compared against.
This matters because two of the three fields legitimately pass through
(or sit at) zero within the domain: v is exactly zero in the
fully-developed region (config.poiseuille_v), and the wall-normal
profile of any field is zero at the two no-slip walls. Normalizing by
the LOCAL or per-station RMS of the exact field (as e.g. a naive
"percent of exact" metric would) blows up or is undefined whenever that
local/station value is near zero -- exactly the pitfall documented in
evaluate_global.py (Roache, P.J. (1998), Verification and Validation in
Computational Science and Engineering, Hermosa Publishers) and in this
project's own earlier "5087015.6%" failure mode. Normalizing by a fixed,
solution-independent GLOBAL scale per Re avoids that failure for all
three fields uniformly, and is the same choice evaluate_global.py and
model.py's own output-normalization already make -- so one metric
definition is used everywhere in this project: developing-region,
fully-developed, and whole-domain alike.

WHY THESE STATIONS: the 4 fractions of Le(Re) (5/25/50/90%) sample,
respectively, the sharp near-inlet development (5%), early development
(25%), the region where profile curvature is changing fastest (50%),
and the approach to fully-developed (90%) -- i.e. they deliberately
span the whole development history, not just its endpoints, per the
same rationale train_hybrid.py's inlet-clustered sampling gives for why
the near-inlet region needs dedicated attention (Dennis & Smith 1973;
Tannehill, Anderson & Pletcher 1997, Sec. 5.6). The x=3*Le station used
for the "fully developed" check is deliberately taken well past Le
itself (Le is only the point where the centerline reaches within 1% of
the exact asymptote, by construction of the Chen (1973) correlation), so
comparing against the CLOSED-FORM asymptote there is a genuine
independent check, not a restatement of the correlation used to place
the station.
================================================================================
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import ChannelPINN, p_scale as _p_scale

DEVICE = "cpu"
ALL_RE = cfg.ALL_RE_FOR_EVAL
TAGS = {Re: ("TRAIN" if Re in cfg.RE_TRAIN else
             ("INTERP (unseen)" if Re == 300.0 else "EXTRAP (unseen)"))
        for Re in ALL_RE}
COLORS = {"TRAIN": "tab:blue", "INTERP (unseen)": "tab:green", "EXTRAP (unseen)": "tab:red"}
FIELDS = ["u", "v", "p"]
OUT = "outputs"


def load_model(ckpt=f"{OUT}/channel_pinn_hybrid.pt"):
    model = ChannelPINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def predict(model, x_val, y_val, Re_val):
    """Returns (u, v, p) in PHYSICAL units at the given points."""
    x = torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    Re = torch.full_like(x, float(Re_val))
    with torch.no_grad():
        u, v, p = model(x, y, Re)
    return u.numpy().flatten(), v.numpy().flatten(), p.numpy().flatten()


def field_scale(field, Re):
    """Global, solution-independent reference scale for `field` at this
    Re -- see the ERROR METRIC block in the module docstring. Never
    derived from the exact/reference field itself."""
    if field == "p":
        return float(_p_scale(torch.tensor(float(Re))))
    return cfg.inlet_velocity(Re)  # u, v


def relL2(pred, exact, scale):
    """The single error formula used everywhere in this file -- see the
    module docstring's ERROR METRIC section for the full justification."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (scale + 1e-12)


def load_reference(ref, Re):
    """Bodoia-Osterle reference arrays for this Re, with pressure shifted
    to the SAME outlet-zero gauge (p(x=L)=0) the PINN's outlet BC and
    train_hybrid._build_interpolators() both use -- comparing the model's
    p output against the solver's raw (inlet-zero) P without this shift
    would add a constant, Re-dependent offset to every pressure error."""
    x = ref[f"x_{Re}"]; y = ref[f"y_{Re}"]
    U = ref[f"U_{Re}"]; V = ref[f"V_{Re}"]
    P = ref[f"P_{Re}"]
    P_shifted = P - P[-1]
    return x, y, U, V, P_shifted


def main():
    model = load_model()
    ref = np.load(f"{OUT}/reference_data.npz")

    stations_frac = [0.05, 0.25, 0.5, 0.9]  # fractions of each Re's own Le
    # dev_rows[field] : list of (Re, tag, [err@5%,err@25%,err@50%,err@90%], mean)
    dev_rows = {f: [] for f in FIELDS}
    fd_rows = {f: [] for f in FIELDS}

    # ---- 1. Developing-region check (u, v, p) at 4 stations per Re -------
    fig, axes = plt.subplots(1, len(ALL_RE), figsize=(4.2 * len(ALL_RE), 4), sharey=True)
    for i, Re in enumerate(ALL_RE):
        x_ref, y_ref, U_ref, V_ref, P_ref = load_reference(ref, Re)
        Le = cfg.entrance_length(Re)
        sc_u, sc_v, sc_p = (field_scale("u", Re), field_scale("v", Re), field_scale("p", Re))

        # profile plot at the 50%-Le station (u only, for a quick visual sanity check)
        x_mid = min(0.5 * Le, cfg.L * 0.95)
        idx_mid = np.argmin(np.abs(x_ref - x_mid))
        u_pred_mid, _, _ = predict(model, np.full_like(y_ref, x_ref[idx_mid]), y_ref, Re)
        err_mid = relL2(u_pred_mid, U_ref[idx_mid, :], sc_u)
        ax = axes[i]
        ax.plot(U_ref[idx_mid, :], y_ref, "k-", lw=2, label="Bodoia-Osterle (ref.)")
        ax.plot(u_pred_mid, y_ref, "o", ms=3, color="tab:orange", label="PINN")
        ax.set_title(f"Re={Re:.0f} [{TAGS[Re]}]\nx={x_ref[idx_mid]:.3f} m (u relL2={err_mid:.1f}%)", fontsize=10)
        ax.set_xlabel("u [m/s]")
        if i == 0:
            ax.set_ylabel("y [m]"); ax.legend(fontsize=8)

        errs = {f: [] for f in FIELDS}
        for frac in stations_frac:
            xs = min(frac * Le, cfg.L * 0.98)
            j = np.argmin(np.abs(x_ref - xs))
            up, vp, pp = predict(model, np.full_like(y_ref, x_ref[j]), y_ref, Re)
            errs["u"].append(relL2(up, U_ref[j, :], sc_u))
            errs["v"].append(relL2(vp, V_ref[j, :], sc_v))
            errs["p"].append(relL2(pp, np.full_like(y_ref, P_ref[j]), sc_p))
        for f in FIELDS:
            dev_rows[f].append((Re, TAGS[Re], errs[f], float(np.mean(errs[f]))))

        # ---- 2. Fully-developed check (u, v, p) vs. exact closed form ----
        x_fd = min(3.0 * Le, cfg.L * 0.98)
        y_fd = np.linspace(-cfg.h, cfg.h, 100)
        u_pred_fd, v_pred_fd, p_pred_fd = predict(model, np.full_like(y_fd, x_fd), y_fd, Re)
        u_exact_fd = cfg.poiseuille_profile(y_fd, Re)
        v_exact_fd = cfg.poiseuille_v(y_fd, Re)
        p_exact_fd = np.full_like(y_fd, cfg.poiseuille_pressure(x_fd, Re))
        fd_rows["u"].append((Re, TAGS[Re], relL2(u_pred_fd, u_exact_fd, sc_u)))
        fd_rows["v"].append((Re, TAGS[Re], relL2(v_pred_fd, v_exact_fd, sc_v)))
        fd_rows["p"].append((Re, TAGS[Re], relL2(p_pred_fd, p_exact_fd, sc_p)))

    fig.tight_layout()
    fig.savefig(f"{OUT}/hybrid_profile_comparison.png", dpi=140)
    plt.close(fig)

    # ---- write per-field developing-region CSV -----------------------
    with open(f"{OUT}/metrics_developing.csv", "w") as fh:
        fh.write("field,Re,tag,relL2_5pct,relL2_25pct,relL2_50pct,relL2_90pct,relL2_mean\n")
        for f in FIELDS:
            for Re, tag, errs, mean in dev_rows[f]:
                fh.write(f"{f},{Re},{tag}," + ",".join(f"{e:.3f}" for e in errs) + f",{mean:.3f}\n")

    with open(f"{OUT}/metrics_fully_developed.csv", "w") as fh:
        fh.write("field,Re,tag,relL2_pct\n")
        for f in FIELDS:
            for Re, tag, err in fd_rows[f]:
                fh.write(f"{f},{Re},{tag},{err:.4f}\n")

    # ---- developing-region error-vs-Re plots, one figure per field, ----
    # ---- one subplot per station fraction -------------------------------
    for f in FIELDS:
        fig, axes = plt.subplots(1, len(stations_frac), figsize=(4.2 * len(stations_frac), 4), sharey=True)
        for k, frac in enumerate(stations_frac):
            ax = axes[k]
            for Re, tag, errs, mean in dev_rows[f]:
                ax.scatter([Re], [errs[k]], c=COLORS[tag], label=tag, s=70, zorder=3)
            # Linear scale as required by report guidelines (not log)
            ax.set_xscale("linear")
            ax.set_title(f"{int(frac*100)}% of Le(Re)")
            ax.set_xlabel("Re")
            ax.grid(alpha=0.3, which="both")
            if k == 0:
                ax.set_ylabel(f"{f} relative L2 error [%]")
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        fig.suptitle(f"Developing-region error vs. Re -- field: {f}")
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_error_vs_Re_developing_{f}.png", dpi=150)
        plt.close(fig)

    # ---- fully-developed error-vs-Re plot, one figure, one subplot per field
    fig, axes = plt.subplots(1, len(FIELDS), figsize=(5.0 * len(FIELDS), 4.5))
    for k, f in enumerate(FIELDS):
        ax = axes[k]
        for Re, tag, err in fd_rows[f]:
            ax.scatter([Re], [err], c=COLORS[tag], label=tag, s=70, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        # Linear scale as required by report guidelines (not log)
        ax.set_xscale("linear")
        ax.set_xlabel("Re")
        ax.set_ylabel(f"{f} relative L2 error [%] (vs. exact closed form)")
        ax.set_title(f"Fully-developed error vs. Re -- field: {f}")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_error_vs_Re_fully_developed.png", dpi=150)
    plt.close(fig)

    # ---- 3. Generalization proof: train vs. unseen Re, per field --------
    summary = (
        "=== ERROR METRIC ===\n"
        "relL2(q) = 100 * sqrt(mean_i (q_pred_i - q_exact_i)^2) / scale(Re), "
        "mean over every wall-normal grid point at the station; "
        "scale(Re)=U0(Re) for u,v and scale(Re)=0.5*U0(Re)^2 for p "
        "(global, solution-independent reference -- see module docstring).\n\n"
        "=== Developing-region check (vs. Bodoia-Osterle), station fractions "
        f"{stations_frac} of each Re's Le ===\n"
    )
    for f in FIELDS:
        summary += f"\n-- field: {f} --\n"
        for Re, tag, errs, mean in dev_rows[f]:
            summary += (f"Re={Re:6.0f} [{tag:16s}]  per-station relL2=" +
                        "/".join(f"{e:.2f}%" for e in errs) + f"  mean={mean:.2f}%\n")

    summary += "\n=== Fully-developed check (vs. exact closed form, far downstream) ===\n"
    for f in FIELDS:
        summary += f"\n-- field: {f} --\n"
        for Re, tag, err in fd_rows[f]:
            summary += f"Re={Re:6.0f} [{tag:16s}]  relL2={err:.3f}%\n"

    summary += (
        "\n=== Generalization proof: TRAIN Re (direct data supervision) vs. "
        "unseen Re (physics + BCs only), per field ===\n"
    )
    for f in FIELDS:
        train_mean = np.mean([m for Re, tag, _, m in dev_rows[f] if tag == "TRAIN"])
        unseen_mean = np.mean([m for Re, tag, _, m in dev_rows[f] if tag != "TRAIN"])
        train_fd = np.mean([e for Re, tag, e in fd_rows[f] if tag == "TRAIN"])
        unseen_fd = np.mean([e for Re, tag, e in fd_rows[f] if tag != "TRAIN"])
        summary += (
            f"[{f}] Developing-region mean error -- TRAIN: {train_mean:.2f}%   UNSEEN: {unseen_mean:.2f}%\n"
            f"[{f}] Fully-developed mean error   -- TRAIN: {train_fd:.3f}%   UNSEEN: {unseen_fd:.3f}%\n"
        )
    summary += (
        "If UNSEEN is comparable to TRAIN (not dramatically worse) for all three\n"
        "fields, the network has learned the Re-dependence of the physics via the\n"
        "log-Re embedding and the PDE residual, rather than memorizing four isolated\n"
        "lookup tables (cf. physics-only report Sec. 7.3, adapted to the hybrid\n"
        "setting and extended here from u-only to u, v, p).\n"
    )
    print(summary)
    with open(f"{OUT}/evaluation_summary.txt", "w") as fh:
        fh.write(summary)
    print(f"\nSaved figures + metrics to {OUT}/")


if __name__ == "__main__":
    main()