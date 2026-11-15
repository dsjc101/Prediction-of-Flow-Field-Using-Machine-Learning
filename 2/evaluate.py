"""
evaluate.py -- Evaluates the hybrid pipe PINN's THREE outputs
(u = axial velocity, v = radial velocity, p = kinematic pressure)
against:
  1. The Hornbeck (1964) developing-flow reference (entrance_solver.py)
     at multiple stations spanning each Re's own development region, for
     every Re including the two held-out ones (Re=300 interpolation,
     Re=600 extrapolation).
  2. The exact fully-developed solution -- Hagen-Poiseuille for u
     (config.poiseuille_profile, White 2006 / Schlichting & Gersten 2017),
     v=0 (fully-developed continuity), and the exact constant pressure
     gradient (config.poiseuille_dpdx / poiseuille_pressure, derived from
     the same governing equation -- see config.py) for p -- at ONE common
     streamwise station near the domain outlet (see "FULLY-DEVELOPED
     STATION" below).
  3. A "not memorization" generalization check: TRAIN Re (direct data
     supervision everywhere) vs. INTERP/EXTRAP Re (no data anywhere --
     only the PDE residual and boundary conditions ever touched those
     two Re values), reported separately for u, v and p.

------------------------------------------------------------------------
METRIC DEFINITIONS -- WHICH points, WHY those points, WHAT formula
------------------------------------------------------------------------

(A) u ONLY: self-normalized relative L2
    At one fixed streamwise station x, over the nr=81 radial grid points
    r_ref (the SAME Hornbeck solver grid the reference was generated on
    -- no extra interpolation error is introduced by picking a different
    radial sampling than the reference itself used):

        relL2(field) = 100 * ||pred - ref||_2 / ||ref||_2
                     = 100 * sqrt(mean((pred-ref)^2)) / sqrt(mean(ref^2))

    This is `relative_l2()` below. ONLY u uses it: u never gets close to
    zero anywhere in this problem (uniform U0 at the inlet, a nonzero
    Poiseuille profile everywhere once developed, zero only exactly on
    the r=R wall line -- a measure-zero set of the check), so
    self-normalizing by that Re's own reference-profile magnitude is
    meaningful and Re-comparable for u specifically.

(B) v EVERYWHERE, and p EVERYWHERE: scale-normalized L2
    Self-normalizing like (A) breaks down whenever the reference
    quantity's own norm can be near zero at the points being checked:
      - v -> 0 as the flow approaches full development -- and this is
        NOT only true at the single fully-developed station X_FD; it is
        already happening inside the developing-region check itself
        (the f=0.9*Le station is, by construction, close to fully
        developed). ||v_ref||_2 at that station can be tiny, so
        self-normalizing there would blow up a metric that is actually
        reporting a tiny, harmless absolute error -- exactly what an
        earlier version of this file did (v "errors" up to several
        thousand percent that were pure normalization artifacts, not
        real model error, as shown by the fully-developed check's v
        error being a steady ~1-4% the whole time).
      - p is a single scalar per (x, Re) (the parabolized Hornbeck solver
        assumes p=p(x), not p(x,r) -- see entrance_solver.py's governing-
        equations docstring), and that scalar passes through (or near)
        zero at some stations by construction of the outlet gauge
        p(x=L)=0, so a self-normalized ratio is undefined or wildly noisy
        near the outlet -- exactly the reason evaluate_global.py's
        docstring already documents for the local (pointwise) error map.
    So v and p BOTH use scale-normalization EVERYWHERE in this file (not
    just at the fully-developed station): normalize
    by a fixed, always-nonzero, Re-characteristic SCALE instead of the
    local reference value -- standard practice for reporting error in
    wall-bounded/internal flows specifically to avoid dividing by a local
    quantity that is allowed to vanish (Roache, P.J. (1998), Verification
    and Validation in Computational Science and Engineering, Hermosa
    Publishers). The scale used is exactly the one the NETWORK ITSELF
    already normalizes by (model.py's u_ref(Re), p_scale(Re)) -- not a
    new arbitrary constant, but the same nondimensionalization the
    architecture is already built around:

        normL2(pred, ref, scale) = 100 * sqrt(mean((pred-ref)^2)) / scale

    v (both checks): scale = U_mean(Re) = model.u_ref(Re)
    p (both checks): scale = p_scale(Re) = model.p_scale(Re)
                     = 0.5*U_mean(Re)^2 (Reynolds pressure-coefficient
                     convention, Cp = p/(0.5 U^2))

    For p specifically, since the Hornbeck reference only gives one
    value per x (not a radial profile), the model is queried at the same
    nr radial points r_ref used for u/v (so the plot/CSV shapes line up,
    and the check also implicitly tests whether the network learned the
    physically-correct near-r-independence of p), but every one of those
    nr predictions is compared against the SAME single reference scalar
    for that station.

------------------------------------------------------------------------
STATIONS -- why the developing-region check x-locations differ by Re
------------------------------------------------------------------------
Entrance length scales linearly with Re (cfg.entrance_length: Le=K*D*Re,
K=0.06, functional form from Langhaar, H.L. (1942), J. Appl. Mech. 9(2),
A55-A58, https://doi.org/10.1115/1.4009183): Le(50)=0.30 m vs.
Le(600)=3.6 m, a 12x spread. Checking every Re at the same x would
compare different physical stages of development, so each station is a
FRACTION of that Re's own Le(Re):

    x_station(Re, f) = f * Le(Re),    f in STATION_FRACS               (*)

"f=0.5" always means "halfway through the entrance region" at every Re,
even though it's a different x in metres for each one. The actual x used
and the corresponding error for u, v AND p are all written to
metrics_developing.csv so this is auditable from the output alone.

------------------------------------------------------------------------
FULLY-DEVELOPED STATION -- exactly where, and why (fixed)
------------------------------------------------------------------------
ONE common station used for every Re:

    X_FD = FD_FRAC * L,   FD_FRAC = 0.975   =>   X_FD = 3.90 m (L=4 m)

chosen so it (a) sits past every Re's own Le(Re) -- checked below with an
assert, so the u, v AND p comparisons are all valid for all six Re -- and
(b) stays strictly before the outlet x=L, where the network's outlet
boundary condition (p=0, train_hybrid.py's loss_outlet term) is directly
enforced during training; checking exactly at x=L would partly be
grading whether the BC loss converged, not whether the interior solution
is physically correct -- this matters even more for p than for u, since
p=0 at x=L is trivially true by construction of the BC, not something
the network had to learn.

------------------------------------------------------------------------
PRESSURE CHECK -- what "exact" means here, and its own caveat
------------------------------------------------------------------------
config.poiseuille_pressure(x, Re) integrates the exact, constant,
fully-developed pressure gradient (config.poiseuille_dpdx -- derived from
the same NS x-momentum equation and the same exact Hagen-Poiseuille
profile used for the u check; see config.py's docstring) from the outlet
gauge p(L)=0. It is exact PROVIDED the flow really is fully developed all
the way from X_FD to L -- the same assumption the u check at X_FD already
relies on, so no new assumption is introduced by extending the check to
p.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from model import PipePINN, u_ref, p_scale

DEVICE = "cpu"
ALL_RE = cfg.ALL_RE_FOR_EVAL
TAGS = {Re: ("TRAIN" if Re in cfg.RE_TRAIN else
             ("INTERP (unseen)" if Re == 300.0 else "EXTRAP (unseen)"))
        for Re in ALL_RE}
OUT = "outputs"

STATION_FRACS = [0.05, 0.25, 0.5, 0.9]
FD_FRAC = 0.975
X_FD = FD_FRAC * cfg.L
assert X_FD > max(cfg.entrance_length(Re) for Re in ALL_RE), (
    f"X_FD={X_FD:.3f} m must exceed every Re's Le(Re); largest is "
    f"{max(cfg.entrance_length(Re) for Re in ALL_RE):.3f} m at Re={max(ALL_RE)}. "
    "Raise FD_FRAC (or shrink K in config.entrance_length) before trusting "
    "the fully-developed numbers."
)

FIELDS = ["u", "v", "p"]


def load_model():
    model = PipePINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(f"{OUT}/pipe_pinn_hybrid.pt", map_location=DEVICE))
    model.eval()
    return model


def predict(model, x_val, r_val, Re_val):
    x = torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1)
    r = torch.tensor(r_val, dtype=torch.float32).reshape(-1, 1)
    Re = torch.full_like(x, float(Re_val))
    with torch.no_grad():
        u, v, p = model(x, r, Re)
    return u.numpy().flatten(), v.numpy().flatten(), p.numpy().flatten()


def relative_l2(pred, exact):
    """100 * ||pred-exact||_2 / ||exact||_2 -- see module docstring, (A)."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (np.sqrt(np.mean(exact ** 2)) + 1e-12)


def normalized_l2(pred, exact, scale):
    """100 * ||pred-exact||_2 / scale, scale a fixed Re-characteristic
    constant (never the local reference value) -- see module docstring, (B)."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (abs(scale) + 1e-12)


def load_reference():
    """Loads Hornbeck reference data, and shifts P to the SAME
    outlet-referenced gauge (p(x=L)=0) train_hybrid.py's data-loss target
    uses (see train_hybrid.py's _build_interpolators), so the pressure
    comparison is apples-to-apples with what the network was actually
    trained/gauge-fixed against."""
    ref = np.load(f"{OUT}/reference_data.npz")
    p_shifted = {}
    for Re in ALL_RE:
        P = ref[f"P_{Re}"]
        p_shifted[Re] = P - P[-1]
    return ref, p_shifted


def main():
    model = load_model()
    ref, p_shifted = load_reference()

    dev_rows, fd_rows = [], []

    # ================= mid-development profile figure (u, unchanged station rule) =================
    fig, axes = plt.subplots(1, len(ALL_RE), figsize=(4.2 * len(ALL_RE), 4), sharey=True)
    for i, Re in enumerate(ALL_RE):
        x_ref = ref[f"x_{Re}"]; r_ref = ref[f"r_{Re}"]; U_ref = ref[f"U_{Re}"]
        Le = cfg.entrance_length(Re)

        x_station = min(0.5 * Le, cfg.L * 0.95)
        idx = np.argmin(np.abs(x_ref - x_station))
        u_pred, v_pred, p_pred = predict(model, np.full_like(r_ref, x_ref[idx]), r_ref, Re)
        err_mid = relative_l2(u_pred, U_ref[idx, :])

        ax = axes[i]
        ax.plot(U_ref[idx, :], r_ref, "k-", lw=2, label="Hornbeck 1964 (ref.)")
        ax.plot(U_ref[idx, :], -r_ref, "k-", lw=2)
        ax.plot(u_pred, r_ref, "o", ms=3, color="tab:orange", label="PINN")
        ax.plot(u_pred, -r_ref, "o", ms=3, color="tab:orange")
        ax.set_title(f"Re={Re:.0f} [{TAGS[Re]}]\nx={x_ref[idx]:.3f} m (u relL2={err_mid:.1f}%)", fontsize=10)
        ax.set_xlabel("u [m/s]")
        if i == 0:
            ax.set_ylabel("r [m]  (mirrored: full pipe diameter)"); ax.legend(fontsize=8)

        # ---- 4-station developing-region check, eq. (*) above, ALL 3 FIELDS ----
        xs_full = []
        errs = {"u": [], "v": [], "p": []}
        for frac in STATION_FRACS:
            xs = min(frac * Le, cfg.L * 0.98)
            j = np.argmin(np.abs(x_ref - xs))
            up, vp, pp = predict(model, np.full_like(r_ref, x_ref[j]), r_ref, Re)
            errs["u"].append(relative_l2(up, ref[f"U_{Re}"][j, :]))
            errs["v"].append(normalized_l2(vp, ref[f"V_{Re}"][j, :], u_ref(Re)))  # see (B): v -> 0
            # approaching full development even within the developing region (esp. the
            # f=0.9 station), so self-normalizing by ||v_ref|| (which shrinks toward 0
            # right along with v) blows up -- same fix as the fully-developed v check.
            p_target = np.full_like(r_ref, p_shifted[Re][j])  # p=p(x) only, see docstring (B)
            errs["p"].append(normalized_l2(pp, p_target, p_scale(Re)))
            xs_full.append(float(x_ref[j]))
        means = {f: float(np.mean(errs[f])) for f in FIELDS}
        dev_rows.append((Re, TAGS[Re], Le, xs_full, errs, means))

        # ---- fully-developed check, ONE common X_FD for every Re ----
        r_fd = np.linspace(0.0, cfg.R, 100)
        u_pred_fd, v_pred_fd, p_pred_fd = predict(model, np.full_like(r_fd, X_FD), r_fd, Re)
        u_exact_fd = cfg.poiseuille_profile(r_fd, Re)
        v_exact_fd = np.zeros_like(r_fd)  # fully-developed continuity => v=0 (see docstring)
        p_exact_fd = np.full_like(r_fd, cfg.poiseuille_pressure(X_FD, Re))
        fd_err = dict(
            u=relative_l2(u_pred_fd, u_exact_fd),
            v=normalized_l2(v_pred_fd, v_exact_fd, u_ref(Re)),
            p=normalized_l2(p_pred_fd, p_exact_fd, p_scale(Re)),
        )
        fd_rows.append((Re, TAGS[Re], fd_err))

    fig.tight_layout()
    fig.savefig(f"{OUT}/pipe_profile_comparison.png", dpi=140)
    plt.close(fig)

    # ================= fully-developed profile figure, ALL 3 FIELDS, one common x=X_FD =================
    fig2, axes2 = plt.subplots(3, len(ALL_RE), figsize=(4.2 * len(ALL_RE), 11), sharey=True)
    field_labels = {"u": "u [m/s]", "v": "v [m/s]", "p": "p [m^2/s^2]"}
    for i, Re in enumerate(ALL_RE):
        r_fd = np.linspace(0.0, cfg.R, 100)
        u_pred_fd, v_pred_fd, p_pred_fd = predict(model, np.full_like(r_fd, X_FD), r_fd, Re)
        exact = dict(
            u=cfg.poiseuille_profile(r_fd, Re),
            v=np.zeros_like(r_fd),
            p=np.full_like(r_fd, cfg.poiseuille_pressure(X_FD, Re)),
        )
        pred = dict(u=u_pred_fd, v=v_pred_fd, p=p_pred_fd)
        fd_err = fd_rows[i][2]
        for row, field in enumerate(FIELDS):
            ax = axes2[row, i]
            ax.plot(exact[field], r_fd, "k-", lw=2,
                    label="Exact Hagen-Poiseuille" if field == "u" else
                          ("Exact (v=0, fully developed)" if field == "v" else
                           "Exact (constant dp/dx)"))
            ax.plot(exact[field], -r_fd, "k-", lw=2)
            ax.plot(pred[field], r_fd, "o", ms=3, color="tab:green", label="PINN")
            ax.plot(pred[field], -r_fd, "o", ms=3, color="tab:green")
            metric_name = "relL2" if field == "u" else "normL2"
            if row == 0:
                ax.set_title(f"Re={Re:.0f} [{TAGS[Re]}]\n{field} {metric_name}={fd_err[field]:.2f}%", fontsize=10)
            else:
                ax.set_title(f"{field} {metric_name}={fd_err[field]:.2f}%", fontsize=10)
            ax.set_xlabel(field_labels[field])
            if i == 0:
                ax.set_ylabel("r [m]"); ax.legend(fontsize=7)
    fig2.suptitle(f"Fully-developed u, v, p at a single fixed station, x={X_FD:.2f} m (all Re)")
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/pipe_profile_fixed_x_fully_developed.png", dpi=140)
    plt.close(fig2)

    # ================= developing-region error-vs-Re figure, ALL 3 FIELDS =================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"TRAIN": "tab:blue", "INTERP (unseen)": "tab:green", "EXTRAP (unseen)": "tab:red"}
    for col, field in enumerate(FIELDS):
        ax = axes3[col]
        for Re, tag, Le, xs, errs, means in dev_rows:
            ax.scatter([Re], [means[field]], c=colors[tag], label=tag, s=70, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Re")
        metric_name = "relL2" if field == "u" else "normL2"
        ax.set_ylabel(f"{field} {metric_name} [%] (mean over the 4 dev.-region stations)")
        ax.set_title(f"Developing-region error vs. Re: {field}")
        ax.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3.savefig(f"{OUT}/fig_error_vs_Re_developing.png", dpi=150)
    plt.close(fig3)

    # ================= fully-developed error-vs-Re figure, ALL 3 FIELDS =================
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, field in enumerate(FIELDS):
        ax = axes4[col]
        for Re, tag, fd_err in fd_rows:
            ax.scatter([Re], [fd_err[field]], c=colors[tag], label=tag, s=70, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Re")
        metric_name = "relL2" if field == "u" else "normL2"
        ax.set_ylabel(f"{field} {metric_name} [%] vs. exact, at x={X_FD:.2f} m")
        ax.set_title(f"Fully-developed error vs. Re: {field}")
        ax.grid(alpha=0.3, which="both")
    fig4.tight_layout()
    fig4.savefig(f"{OUT}/fig_error_vs_Re_fully_developed.png", dpi=150)
    plt.close(fig4)

    # ================= CSVs =================
    with open(f"{OUT}/metrics_developing.csv", "w") as f:
        cols = ["Re", "tag", "Le_m"]
        cols += [f"x_{int(round(fr*100))}pct_m" for fr in STATION_FRACS]
        for field in FIELDS:
            cols += [f"{field}_relL2_{int(round(fr*100))}pct" if field == "u"
                     else f"{field}_normL2_{int(round(fr*100))}pct" for fr in STATION_FRACS]
        cols += [f"{field}_mean" for field in FIELDS]
        f.write(",".join(cols) + "\n")
        for Re, tag, Le, xs, errs, means in dev_rows:
            row = [str(Re), tag, f"{Le:.4f}"] + [f"{x:.4f}" for x in xs]
            for field in FIELDS:
                row += [f"{e:.3f}" for e in errs[field]]
            row += [f"{means[field]:.3f}" for field in FIELDS]
            f.write(",".join(row) + "\n")

    with open(f"{OUT}/metrics_fully_developed.csv", "w") as f:
        f.write("Re,tag,x_fd_m,u_relL2_pct,v_normL2_pct,p_normL2_pct\n")
        for Re, tag, fd_err in fd_rows:
            f.write(f"{Re},{tag},{X_FD:.4f},{fd_err['u']:.4f},{fd_err['v']:.4f},{fd_err['p']:.4f}\n")

    # ================= text summary, all 3 fields =================
    train_mean = {f: np.mean([m[f] for Re, tag, Le, xs, errs, m in dev_rows if tag == "TRAIN"]) for f in FIELDS}
    unseen_mean = {f: np.mean([m[f] for Re, tag, Le, xs, errs, m in dev_rows if tag != "TRAIN"]) for f in FIELDS}
    train_fd = {f: np.mean([e[f] for Re, tag, e in fd_rows if tag == "TRAIN"]) for f in FIELDS}
    unseen_fd = {f: np.mean([e[f] for Re, tag, e in fd_rows if tag != "TRAIN"]) for f in FIELDS}

    summary = (
        "=== Developing-region check (vs. Hornbeck 1964); stations = f*Le(Re), "
        f"f in {STATION_FRACS}, Le=K*D*Re, K=0.06 (Langhaar 1942 functional form) ===\n"
        "u: self-normalized relative L2 (eq. A). v, p: scale-normalized L2 (eq. B).\n"
    )
    for Re, tag, Le, xs, errs, means in dev_rows:
        summary += (f"Re={Re:6.0f} [{tag:16s}] Le={Le:5.3f} m  x=" +
                    "/".join(f"{x:.3f}" for x in xs) + " m\n"
                    f"    u relL2 =" + "/".join(f"{e:.1f}%" for e in errs["u"]) + f"  mean={means['u']:.2f}%\n"
                    f"    v normL2=" + "/".join(f"{e:.1f}%" for e in errs["v"]) + f"  mean={means['v']:.2f}%\n"
                    f"    p normL2=" + "/".join(f"{e:.1f}%" for e in errs["p"]) + f"  mean={means['p']:.2f}%\n")
    summary += (f"\n=== Fully-developed check (vs. exact Hagen-Poiseuille u, v=0, and constant-dp/dx p), "
                f"single common station x={X_FD:.3f} m ({FD_FRAC*100:.1f}% of L={cfg.L} m) ===\n")
    for Re, tag, fd_err in fd_rows:
        summary += (f"Re={Re:6.0f} [{tag:16s}]  u relL2={fd_err['u']:.3f}%  "
                    f"v normL2={fd_err['v']:.3f}%  p normL2={fd_err['p']:.3f}%\n")
    summary += (
        "\n=== Generalization proof: TRAIN Re (direct data supervision) vs. "
        "unseen Re (physics + BCs only), all 3 fields ===\n"
    )
    for f in FIELDS:
        summary += (f"[{f}] Developing-region mean -- TRAIN: {train_mean[f]:.2f}%   UNSEEN: {unseen_mean[f]:.2f}%   |   "
                    f"Fully-developed -- TRAIN: {train_fd[f]:.3f}%   UNSEEN: {unseen_fd[f]:.3f}%\n")
    summary += (
        "If UNSEEN is comparable to TRAIN for all three fields (not dramatically\n"
        "worse), the network has learned the Re-dependence of the axisymmetric\n"
        "physics via the log-Re embedding and the PDE residual, rather than\n"
        "memorizing four isolated lookup tables.\n"
    )
    print(summary)
    with open(f"{OUT}/evaluation_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved figures + metrics to {OUT}/")


if __name__ == "__main__":
    main()