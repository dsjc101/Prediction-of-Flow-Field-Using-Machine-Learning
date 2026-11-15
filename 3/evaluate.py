"""
evaluate.py
============
Developing-region evaluation: trained PINN vs the independent duct_solver
marching reference, at matched (x, Re) stations -- the duct analogue of
Section 10.1/10.2 in the pipe report.

CHANGED: reports relL2 for ALL FOUR predicted fields (u, v, w, p), not
just u, now that duct_solver.py provides reference v, w, p (via the
Patankar & Spalding (1972) pressure-splitting recovery documented in that
module) alongside the original u marching solution.

Produces, per Re: a mid-development-length profile comparison plot
(u, v, w, p vs y at a representative x-slice, both models overlaid), and
a per-station, per-field mean relative-L2 error table across the whole
developing region.
"""

import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from pinn_model import DuctPINN

# CHANGED (per requested corrections): the developing region is reported at
# fixed FRACTIONS OF EACH Re's OWN DEVELOPMENT LENGTH Le, not fixed
# fractions of the (Re-independent) marching-domain length L_DOMAIN. Le
# ranges from 0.61 m (Re=50) to 6.82 m (Re=600) in this project (Sec 3/4.3
# of the report), so a single physical x works for none of them equally.
LE_FRACTIONS = [0.05, 0.25, 0.50, 0.90]

# The four predicted quantities, in the order model.predict_physical
# returns them, mapped to the matching key in the reference npz.
QUANTITIES = ["U", "V", "W", "P"]


def load_reference_data():
    data = {}
    for f in sorted(glob.glob(os.path.join(config.DATA_DIR, "duct_Re*.npz"))):
        d = np.load(f)
        data[int(d["Re"])] = dict(
            x=d["x"], U=d["U"], V=d["V"], W=d["W"], P=d["P"],
            y=d["y"], z=d["z"], U_mean=float(d["U_mean"]),
            crossflow_valid_from=int(d["crossflow_valid_from"])
            if "crossflow_valid_from" in d else 3,
        )
    return data


def load_model(device="cpu"):
    model = DuctPINN().to(device)
    model.load_state_dict(torch.load(os.path.join(config.CKPT_DIR, "duct_pinn.pt"),
                                      map_location=device))
    model.eval()
    return model


def get_Le(d):
    """
    Development length for this Re. Prefers the value generate_data.py now
    stores directly in the npz (measured on the SAME full-resolution grid
    the data itself came from -- see generate_data.py). Falls back to
    recomputing it from the u_max(x) curve (same 1%-convergence criterion
    as DuctEntranceSolver.development_length) for npz files produced before
    this fix, so this function still works on old data without a re-run.
    """
    if "Le" in d:
        return float(d["Le"])
    u_max_x = d["U"].max(axis=(1, 2))
    u_final = u_max_x[-1]
    idx = int(np.argmax(np.abs(u_max_x - u_final) / u_final < 0.01))
    return float(d["x"][idx])


def station_index_for_x(x_array, x_target, min_index=0):
    """
    Index of the station whose x is closest to x_target, never below
    min_index. min_index is used to keep v, w, p queries out of the
    top-hat-inlet singular stations (d['crossflow_valid_from'], see
    duct_solver.py/generate_data.py) -- u itself has no such floor since
    its own, much smaller, artifact is confined to station 0.
    """
    idx = int(np.argmin(np.abs(x_array - x_target)))
    return max(idx, min_index)


def relL2_at_station(model, d, n, Re, field="U", device="cpu"):
    """
    Error of the predicted field against the marching reference at ONE
    streamwise station n, for field in {U, V, W, P}. Two DIFFERENT
    metrics are used, by field:

    U, P -- true relative L2 error:

        relL2(x_n) = || f_pinn(x_n, y, z) - f_ref(x_n, y, z) ||_2
                     ---------------------------------------------
                          || f_ref(x_n, y, z) ||_2 + eps

    both norms taken over every (y,z) grid point at that station (a
    standard field-comparison relative error, e.g. as used throughout
    Raissi, Perdikaris & Karniadakis 2019 Sec 4 for PINN accuracy
    reporting). eps = 1e-6*U_ref(Re) or 1e-6*p_scale(Re) is a small
    numerical-safety floor only -- u and p are never close to zero over
    an entire cross-section, so this floor rarely matters.

    V, W -- normalized ABSOLUTE (RMS) error, NOT relative (BUG FIX, see
    below):

        err(x_n) = RMS_{y,z}( |v_pinn(x_n,y,z) - v_ref(x_n,y,z)| ) / U_ref(Re)

    v, w are IDENTICALLY ZERO throughout the fully-developed region
    (analytical_duct.exact_cross_velocities) and small everywhere else --
    a RELATIVE error against a reference that is genuinely at or near
    zero is mathematically degenerate: dividing by ~0 turns any small,
    harmless prediction residual into an arbitrarily large percentage,
    regardless of how well the field is actually fit (no eps floor fixes
    this -- it only trades "blows up" for "an arbitrary floor-dependent
    number", neither of which is informative). The fix is to not use a
    relative metric here at all: normalizing the ABSOLUTE error by the
    flow's own bulk velocity scale U_ref(Re) -- the SAME normalization
    pinn_model.py already uses for its own v, w outputs -- gives a
    bounded, well-defined quantity ("typical v/w error as a fraction of
    the bulk flow speed") that behaves sensibly whether the true v, w at
    that station is large (near the entrance) or exactly 0 (fully
    developed). This is the same principle already used, correctly, for
    the local error HEATMAP in evaluate_global.py (|err|/scale, not
    |err|/|local reference|) -- V, W's relL2 numbers here previously
    disagreed wildly with that heatmap's low, sane percentages for the
    same field precisely because the two used different (relative vs.
    absolute) metrics; this fix brings them into agreement.
    """
    y, z = d["y"], d["z"]
    Y, Z = np.meshgrid(y, z, indexing="ij")
    x_val = d["x"][n]
    x_flat = np.full(Y.size, x_val)
    u_pred, v_pred, w_pred, p_pred = model.predict_physical(x_flat, Y.ravel(), Z.ravel(), Re, device=device)
    preds = dict(U=u_pred, V=v_pred, W=w_pred, P=p_pred)
    f_pred = preds[field].reshape(Y.shape)
    f_ref = d[field][n]

    U_ref_scale = Re * config.NU / config.DH

    if field in ("V", "W"):
        return float(np.sqrt(np.mean((f_pred - f_ref) ** 2)) / U_ref_scale)

    p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py
    eps = 1e-6 * (U_ref_scale if field == "U" else p_scale)
    num = np.linalg.norm(f_pred - f_ref)
    den = np.linalg.norm(f_ref) + eps
    return float(num / den)


def developing_region_relL2(model, ref, Re, field="U", device="cpu"):
    """
    Whole-domain error summary for this Re and field, from x=0 (or
    d['crossflow_valid_from'] for V, W, P) to x=L_DOMAIN.

    BUG FIX: this previously averaged relL2_at_station's per-station
    RATIO over every station individually. That aggregation is fragile
    for ANY field that is small or crosses zero at even a handful of
    stations: P in particular is gauge-fixed to 0 at the outlet, so its
    reference value shrinks toward 0 over the last few stations; a
    per-station ratio there blows up (small denominator, ordinary-sized
    numerator), and because it's a plain MEAN, a handful of such outlier
    stations can dominate an average over thousands of otherwise-normal
    values -- exactly what produced the previous 500,000%-scale P
    numbers here, even though the SAME field's single-global-norm metric
    (entire_domain_relL2 in evaluate_global.py) reported a sane ~80%.

    Fixed by computing ONE aggregate error over ALL included stations at
    once, the same robust approach entire_domain_relL2 already uses: for
    U, P this is a single relL2 = ||error||_2/||reference||_2 over the
    whole (x,y,z) block (a single near-zero station can no longer
    dominate, since it contributes only its own small share to both the
    global numerator and denominator); for V, W it is the RMS absolute
    error normalized by U_ref(Re), for the same reason given in
    relL2_at_station's docstring (v, w are near-zero almost everywhere,
    so even a GLOBAL relative norm would still be fragile for them,
    unlike P which is only near-zero in a small tail region).

    Returns a single float (not a per-station array -- this function's
    old per-station array return was only ever used to take its mean in
    run_all(), which the caller no longer needs to do).
    """
    d = ref[Re]
    n0 = d["crossflow_valid_from"] if field in ("V", "W", "P") else 0
    x, y, z = d["x"], d["y"], d["z"]
    x_eval = x[n0:]
    X, Y, Z = np.meshgrid(x_eval, y, z, indexing="ij")
    u_pred, v_pred, w_pred, p_pred = model.predict_physical(X.ravel(), Y.ravel(), Z.ravel(), Re, device=device)
    preds = dict(U=u_pred, V=v_pred, W=w_pred, P=p_pred)
    f_pred = preds[field].reshape(X.shape)
    f_ref = d[field][n0:]

    if field in ("V", "W"):
        U_ref_scale = Re * config.NU / config.DH
        return float(np.sqrt(np.mean((f_pred - f_ref) ** 2)) / U_ref_scale)

    U_ref_scale = Re * config.NU / config.DH
    p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)
    eps = 1e-6 * (U_ref_scale if field == "U" else p_scale)
    return float(np.linalg.norm(f_pred - f_ref) / (np.linalg.norm(f_ref) + eps))


def developing_region_relL2_at_Le_fractions(model, ref, Re, fracs=LE_FRACTIONS, device="cpu"):
    """
    relL2 evaluated at x = frac * Le for frac in {5%, 25%, 50%, 90%},
    using Re's OWN measured Le (get_Le), not a shared domain-fraction --
    see module/original docstring for why a shared fraction of L_DOMAIN
    is not physically comparable across Re. Returns a nested dict
    per_field[field][frac] = {x, n, relL2}, plus mean_relL2[field] and an
    ALL-FIELDS mean (u,v,w,p equally weighted) used as the single
    headline generalization number.
    """
    d = ref[Re]
    Le = get_Le(d)
    per_field = {}
    mean_relL2 = {}
    for field in QUANTITIES:
        n_floor = d["crossflow_valid_from"] if field in ("V", "W", "P") else 0
        per_frac = {}
        for frac in fracs:
            n = station_index_for_x(d["x"], frac * Le, min_index=n_floor)
            per_frac[frac] = dict(x=float(d["x"][n]), n=n,
                                   relL2=relL2_at_station(model, d, n, Re, field=field, device=device))
        per_field[field] = per_frac
        mean_relL2[field] = float(np.mean([per_frac[f]["relL2"] for f in fracs]))
    mean_relL2["ALL"] = float(np.mean([mean_relL2[q] for q in QUANTITIES]))
    return per_field, mean_relL2, Le


def plot_midstation_profile(model, ref, Re, device="cpu", le_frac=0.5, outpath=None):
    """
    4-panel figure (u, v, w, p vs y at mid-width z) at x = le_frac * Le
    (Re's own measured development length), so "mid-development-length"
    means the same physical stage of development for every Re.
    """
    d = ref[Re]
    Le = get_Le(d)
    n_floor = d["crossflow_valid_from"]
    n = station_index_for_x(d["x"], le_frac * Le, min_index=n_floor)
    x_val = d["x"][n]
    y, z = d["y"], d["z"]
    j_mid = len(z) // 2
    x_flat = np.full_like(y, x_val)
    z_flat = np.full_like(y, z[j_mid])
    u_pred, v_pred, w_pred, p_pred = model.predict_physical(x_flat, y, z_flat, Re, device=device)
    preds = dict(U=u_pred, V=v_pred, W=w_pred, P=p_pred)
    labels = dict(U="u [m/s]", V="v [m/s]", W="w [m/s]", P="p [Pa-equivalent, rho=1]")

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), sharex=True)
    for ax, field in zip(axes, QUANTITIES):
        f_ref = d[field][n, :, j_mid]
        ax.plot(y, f_ref, "k-", lw=2, label="Marching reference")
        ax.plot(y, preds[field], "--", color="orange", lw=2, label="PINN")
        ax.set_xlabel("y [m]")
        ax.set_ylabel(labels[field])
        ax.set_title(field)
    axes[0].legend()
    fig.suptitle(f"Re={Re}, x={x_val:.3f} m ({le_frac*100:.0f}% of Le={Le:.3f} m), "
                 f"mid-width slice (z={z[j_mid]:.3f} m)")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


def run_all(device="cpu"):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    ref = load_reference_data()
    model = load_model(device=device)

    def role_of(Re):
        return "TRAIN" if Re in config.RE_TRAIN else (
            "INTERP" if Re == config.RE_TEST_INTERP else "EXTRAP")

    print("NOTE: U, P columns are relative L2 error (% of the local field's own\n"
          "      magnitude). V, W columns are a DIFFERENT metric -- RMS absolute\n"
          "      error normalized by the bulk velocity U_ref(Re), NOT a relative\n"
          "      error -- because v, w are genuinely ~0 through most of the duct,\n"
          "      making a relative-to-near-zero metric degenerate (see\n"
          "      relL2_at_station's docstring). The 'ALL' row averages across\n"
          "      these two different metric types as a rough aggregate score;\n"
          "      read the per-field numbers, not just ALL, for real conclusions.\n")

    # ---- (A) headline table: relL2 at fixed fractions of EACH Re's own
    # Le, per field, plus the u+v+w+p mean ----
    header = f"{'Re':>5} {'Role':>8} {'Le[m]':>7} {'Field':>6} " + \
             "".join(f"{f'{int(fr*100)}%Le':>10}" for fr in LE_FRACTIONS) + f"{'mean':>10}"
    print(header)
    le_summary = {}
    for Re in config.RE_ALL:
        role = role_of(Re)
        per_field, mean_relL2, Le = developing_region_relL2_at_Le_fractions(model, ref, Re, device=device)
        le_summary[Re] = dict(role=role, Le=Le, per_field=per_field, mean_relL2=mean_relL2)
        for field in QUANTITIES:
            row = "".join(f"{per_field[field][fr]['relL2']*100:9.2f}%" for fr in LE_FRACTIONS)
            print(f"{Re:5d} {role:>8} {Le:7.3f} {field:>6}" + row + f"{mean_relL2[field]*100:9.2f}%")
        print(f"{'':5} {'':8} {'':7} {'ALL':>6}" + " " * (10 * len(LE_FRACTIONS))
              + f"{mean_relL2['ALL']*100:9.2f}%")
        plot_midstation_profile(model, ref, Re, device=device, le_frac=0.5,
                                 outpath=os.path.join(config.FIG_DIR, f"profile_Re{Re}.png"))

    train_le_errs = [le_summary[Re]["mean_relL2"]["ALL"] for Re in config.RE_TRAIN]
    unseen_le_errs = [le_summary[config.RE_TEST_INTERP]["mean_relL2"]["ALL"],
                       le_summary[config.RE_TEST_EXTRAP]["mean_relL2"]["ALL"]]
    print(f"\n[Le-fraction, u+v+w+p mean] Generalization check: TRAIN mean="
          f"{np.mean(train_le_errs)*100:.2f}%  UNSEEN mean={np.mean(unseen_le_errs)*100:.2f}%")

    # ---- (B) Full-domain sweep mean per field, kept for continuity with
    # the old table / whole-domain generalization check ----
    print(f"\n{'Re':>5} {'Role':>10} {'Field':>6} {'Whole-domain-sweep error (U,P=relL2; V,W=|err|/Uref)':>52}")
    summary = {}
    for Re in config.RE_ALL:
        role = role_of(Re)
        summary[Re] = dict(role=role, mean_relL2={})
        for field in QUANTITIES:
            mean_err = developing_region_relL2(model, ref, Re, field=field, device=device)
            summary[Re]["mean_relL2"][field] = mean_err
            print(f"{Re:5d} {role:>10} {field:>6} {mean_err*100:29.2f}%")
        summary[Re]["mean_relL2"]["ALL"] = float(np.mean([summary[Re]["mean_relL2"][q] for q in QUANTITIES]))

    train_errs = [summary[Re]["mean_relL2"]["ALL"] for Re in config.RE_TRAIN]
    unseen_errs = [summary[config.RE_TEST_INTERP]["mean_relL2"]["ALL"],
                   summary[config.RE_TEST_EXTRAP]["mean_relL2"]["ALL"]]
    print(f"\n[Whole-domain, u+v+w+p mean] Generalization check: TRAIN mean={np.mean(train_errs)*100:.2f}%  "
          f"UNSEEN mean={np.mean(unseen_errs)*100:.2f}%")

    with open(os.path.join(config.FIG_DIR, "developing_region_summary.txt"), "w") as f:
        f.write("=== Le-fraction stations (headline numbers) ===\n")
        for Re, s in le_summary.items():
            for field in QUANTITIES:
                f.write(f"Re={Re} role={s['role']} field={field} Le={s['Le']:.6f} "
                        + " ".join(f"relL2@{int(fr*100)}%Le={s['per_field'][field][fr]['relL2']:.6f}"
                                    for fr in LE_FRACTIONS)
                        + f" mean_relL2={s['mean_relL2'][field]:.6f}\n")
            f.write(f"Re={Re} role={s['role']} field=ALL mean_relL2={s['mean_relL2']['ALL']:.6f}\n")
        f.write(f"Le_fraction_TRAIN_mean={np.mean(train_le_errs):.6f}\n")
        f.write(f"Le_fraction_UNSEEN_mean={np.mean(unseen_le_errs):.6f}\n\n")
        f.write("=== Whole-domain-sweep mean (context only) ===\n")
        for Re, s in summary.items():
            for field in QUANTITIES + ["ALL"]:
                f.write(f"Re={Re} role={s['role']} field={field} mean_relL2={s['mean_relL2'][field]:.6f}\n")
        f.write(f"Whole_domain_TRAIN_mean={np.mean(train_errs):.6f}\n")
        f.write(f"Whole_domain_UNSEEN_mean={np.mean(unseen_errs):.6f}\n")

    print(f"\nSaved profile plots to {config.FIG_DIR}/profile_Re*.png and "
          f"summary table to {config.FIG_DIR}/developing_region_summary.txt")
    print("-> send both back for the report's Section-10-equivalent tables/figures.")


if __name__ == "__main__":
    run_all()