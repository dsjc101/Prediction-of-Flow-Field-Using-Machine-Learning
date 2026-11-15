"""
evaluate_global.py
====================
Whole-(x,y,z)-domain error against the full marching reference (not just a
few stations), AND fully-developed-region comparison against the EXACT
analytical rectangular-duct solution (analytical_duct.py) -- the duct
analogue of Sections 10.2 (fully-developed benchmark) and 10.3
(entire-domain error maps) in the pipe report.

CHANGED: every metric and every error map below is now computed for all
four predicted fields (u, v, w, p), not just u -- see duct_solver.py for
where the v, w, p reference data comes from (Patankar & Spalding (1972)
pressure-splitting) and analytical_duct.py for the exact fully-developed
v=w=0 and linear-in-x p used as the analytical benchmark for those three.
"""

import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

import config
from pinn_model import DuctPINN
from analytical_duct import fully_developed_profile
from evaluate import load_reference_data, load_model, QUANTITIES

# Physical scale each field is normalized by for the well-conditioned
# (non-pointwise-blowing-up) local-error maps below -- SAME convention as
# pinn_model.py's own input/output normalization (U_ref for velocities,
# q=0.5*U_ref^2 for pressure), so the plotted quantity is always on the
# problem's own natural scale.
def _field_scale(Re, field):
    U_ref = Re * config.NU / config.DH
    if field in ("U", "V", "W"):
        return U_ref
    return 0.5 * U_ref ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py


def entire_domain_relL2(model, ref, Re, field="U", device="cpu"):
    """
    U, P: relL2 = ||f_pinn - f_ref||_2 / (||f_ref||_2 + eps) over EVERY
    (x,y,z) grid point in the full marched domain for this Re -- the
    strictest single-number accuracy summary in this file.

    V, W: normalized ABSOLUTE (RMS) error / U_ref(Re), NOT relative (BUG
    FIX -- same reasoning as evaluate.relL2_at_station: v, w are
    identically 0 through most of the domain, so a relative-to-near-zero
    metric is degenerate and was previously producing meaningless
    millions-of-percent numbers that disagreed with this file's own,
    correctly-normalized, local error heatmap for the same field).

    v, w, p points below d['crossflow_valid_from'] (the top-hat-inlet
    singularity, see duct_solver.py/generate_data.py) are excluded from
    this norm for the same reason as in evaluate.py.
    """
    d = ref[Re]
    x, y, z = d["x"], d["y"], d["z"]
    x0 = d["crossflow_valid_from"] if field in ("V", "W", "P") else 0
    x_eval = x[x0:]
    X, Y, Z = np.meshgrid(x_eval, y, z, indexing="ij")
    u_pred, v_pred, w_pred, p_pred = model.predict_physical(X.ravel(), Y.ravel(), Z.ravel(), Re, device=device)
    preds = dict(U=u_pred, V=v_pred, W=w_pred, P=p_pred)
    f_pred = preds[field].reshape(X.shape)
    f_ref = d[field][x0:]

    if field in ("V", "W"):
        U_ref_scale = Re * config.NU / config.DH
        err = np.sqrt(np.mean((f_pred - f_ref) ** 2)) / U_ref_scale
        return err, f_pred, f_ref, x_eval

    eps = 1e-6 * _field_scale(Re, field)
    relL2 = np.linalg.norm(f_pred - f_ref) / (np.linalg.norm(f_ref) + eps)
    return relL2, f_pred, f_ref, x_eval


def fully_developed_relL2(model, ref, Re, field="U", device="cpu"):
    """
    Compare the PINN's outlet-station field against the EXACT analytical
    fully-developed solution (analytical_duct.py): u from the Fourier
    series (fully_developed_profile), v=w=0 exactly
    (exact_cross_velocities). This is the duct's Hagen-Poiseuille-
    equivalent headline benchmark.

    P is handled differently (BUG FIX): absolute pressure is only defined
    up to an arbitrary additive gauge constant (the PINN's own gauge is
    fixed by its outlet BC p=0 at x=config.L_DOMAIN; the analytical
    exact_pressure's gauge is whatever (x_ref, p_ref) it's called with).
    An earlier version of this function evaluated exact_pressure AT the
    same point used as its own zero-reference, which trivially returns
    exactly 0 regardless of the true pressure there and made the relL2
    denominator collapse to just the eps floor -- the direct cause of the
    billions-of-percent numbers previously seen for P here. The
    physically meaningful, GAUGE-INDEPENDENT quantity is not absolute p
    but the pressure SLOPE dP/dx (this is, after all, exactly what
    friction_factor_fRe/f*Re_Dh already reports and what this project
    treats as the real physical result for pressure, per
    analytical_duct.py). So for P, this function instead compares dP/dx
    computed from the PINN's own predictions at two stations, both
    comfortably past Le (so both are genuinely in the fully-developed
    region), against the analytical dP/dx used inside exact_pressure --
    a scalar relative error, not a field.
    """
    d = ref[Re]
    y, z = d["y"], d["z"]

    if field == "P":
        from analytical_duct import exact_dPdx
        Le = float(d["Le"]) if "Le" in d else None
        x_arr = d["x"]
        n2 = len(x_arr) - 1
        # n1: first station at least 1.2*Le downstream (comfortably past
        # the entrance region), falling back to the midpoint of the
        # domain if Le is unavailable or the margin doesn't exist.
        if Le is not None and x_arr[-1] > 1.2 * Le:
            n1 = int(np.argmax(x_arr >= 1.2 * Le))
        else:
            n1 = len(x_arr) // 2
        x1, x2 = x_arr[n1], x_arr[n2]
        Y, Z = np.meshgrid(y, z, indexing="ij")
        x1_flat = np.full(Y.size, x1)
        x2_flat = np.full(Y.size, x2)
        _, _, _, p1_pred = model.predict_physical(x1_flat, Y.ravel(), Z.ravel(), Re, device=device)
        _, _, _, p2_pred = model.predict_physical(x2_flat, Y.ravel(), Z.ravel(), Re, device=device)
        dPdx_pinn = (float(np.mean(p2_pred)) - float(np.mean(p1_pred))) / (x2 - x1)

        dPdx_exact = exact_dPdx(config.H, config.W, config.NU, d["U_mean"])

        eps = 1e-6 * abs(dPdx_exact) if dPdx_exact != 0 else 1e-12
        return float(abs(dPdx_pinn - dPdx_exact) / (abs(dPdx_exact) + eps))

    x_out = d["x"][-1]
    Y, Z = np.meshgrid(y, z, indexing="ij")
    x_flat = np.full(Y.size, x_out)
    u_pred, v_pred, w_pred, p_pred = model.predict_physical(x_flat, Y.ravel(), Z.ravel(), Re, device=device)
    preds = dict(U=u_pred.reshape(Y.shape), V=v_pred.reshape(Y.shape), W=w_pred.reshape(Y.shape))

    if field == "U":
        f_exact = fully_developed_profile(y, z, config.H, config.W, d["U_mean"])
        eps = 1e-6 * _field_scale(Re, field)
        return float(np.linalg.norm(preds[field] - f_exact) / (np.linalg.norm(f_exact) + eps))

    # V, W: exact value is IDENTICALLY 0 here (exact_cross_velocities),
    # so a relative error is undefined by construction -- ANY nonzero
    # PINN residual, however small, would read as infinite percent error
    # (this was the direct cause of the previous billions-of-percent
    # numbers). BUG FIX: report normalized absolute (RMS) error instead
    # -- same U_ref-normalized convention as entire_domain_relL2's V, W
    # branch above and evaluate.py's relL2_at_station, so all three V, W
    # metrics in this project are now on the same, meaningful footing.
    U_ref_scale = Re * config.NU / config.DH
    return float(np.sqrt(np.mean(preds[field] ** 2)) / U_ref_scale)


def run_all(device="cpu"):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    ref = load_reference_data()
    model = load_model(device=device)

    print("NOTE: U, P columns are relative L2 error. V, W columns are a\n"
          "      DIFFERENT metric -- RMS absolute error / U_ref(Re), not\n"
          "      relative -- because v, w are genuinely ~0 through most of\n"
          "      the duct (exactly 0 in the fully-developed region), making\n"
          "      a relative-to-near-zero metric degenerate. See\n"
          "      entire_domain_relL2/fully_developed_relL2 docstrings.\n")
    print(f"{'Re':>5} {'Role':>8} {'Field':>6} {'Entire-domain error':>20} {'Fully-dev vs exact error':>26}")
    rows = []
    for Re in config.RE_ALL:
        role = "TRAIN" if Re in config.RE_TRAIN else (
            "INTERP" if Re == config.RE_TEST_INTERP else "EXTRAP")
        d = ref[Re]
        j_mid = len(d["z"]) // 2
        U_ref_scale = Re * config.NU / config.DH

        fig, axes = plt.subplots(2, len(QUANTITIES), figsize=(5 * len(QUANTITIES), 6), sharex=True, sharey="row")
        for col, field in enumerate(QUANTITIES):
            glob_err, f_pred, f_ref, x_eval = entire_domain_relL2(model, ref, Re, field=field, device=device)
            fd_err = fully_developed_relL2(model, ref, Re, field=field, device=device)
            rows.append((Re, role, field, glob_err, fd_err))
            print(f"{Re:5d} {role:>8} {field:>6} {glob_err*100:19.2f}% {fd_err*100:25.2f}%")

            # ---- local error heatmap at mid-width slice ----
            # Normalized by a GLOBAL field scale (U_ref(Re) for velocities,
            # p_scale(Re) for p -- the SAME scale pinn_model.py already
            # normalizes its own outputs by), NOT the local reference
            # value: a purely-local-relative metric blows up wherever the
            # reference field is near zero by construction (every velocity
            # component at a no-slip wall; v, w everywhere in the
            # fully-developed region) even when the absolute error there
            # is small -- exactly the artifact already diagnosed and fixed
            # for u in this file's original version, and equally true for
            # v, w, p (arguably more so, since v, w are near-zero almost
            # everywhere, not just at the walls).
            scale = _field_scale(Re, field)
            local_err_abs = np.abs(f_pred[:, :, j_mid] - f_ref[:, :, j_mid])
            local_err_norm = local_err_abs / scale
            print(f"      [Re={Re}, {field}] max local err/scale = {local_err_norm.max()*100:.2f}%")

            im0 = axes[0, col].pcolormesh(x_eval, d["y"], local_err_norm.T, shading="auto", cmap="viridis")
            axes[0, col].set_title(f"{field}: |err|/scale, Re={Re}")
            axes[0, col].set_xlabel("x [m]")
            if col == 0:
                axes[0, col].set_ylabel("y [m]")
            fig.colorbar(im0, ax=axes[0, col], label="|err|/scale")

            # secondary panel: pointwise-relative metric (unclipped, log
            # scale) -- kept for transparency (shows where it WOULD blow
            # up), not used as the headline diagnostic, same rationale as
            # the original single-field version of this file.
            old_metric = local_err_abs / (np.abs(f_ref[:, :, j_mid]) + 1e-3 * scale)
            im1 = axes[1, col].pcolormesh(x_eval, d["y"], old_metric.T, shading="auto", cmap="magma",
                                           norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=max(old_metric.max(), 1e-2)))
            axes[1, col].set_xlabel("x [m]")
            if col == 0:
                axes[1, col].set_ylabel("y [m]")
            axes[1, col].set_title("pointwise-relative (log, unclipped, reference only)")
            fig.colorbar(im1, ax=axes[1, col], label="|err|/|ref|")

        fig.suptitle(f"Re={Re} ({role}) -- error maps, all 4 fields, mid-width slice")
        fig.tight_layout()
        fig.savefig(os.path.join(config.FIG_DIR, f"errormap_Re{Re}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(config.FIG_DIR, "global_summary.txt"), "w") as f:
        for Re, role, field, ge, fe in rows:
            f.write(f"Re={Re} role={role} field={field} entire_domain_relL2={ge:.6f} "
                    f"fully_developed_relL2={fe:.6f}\n")
        for field in QUANTITIES:
            field_rows = [(Re, role, ge, fe) for (Re, role, fld, ge, fe) in rows if fld == field]
            train_ge = np.mean([ge for (Re, role, ge, fe) in field_rows if role == "TRAIN"])
            unseen_ge = np.mean([ge for (Re, role, ge, fe) in field_rows if role in ("INTERP", "EXTRAP")])
            f.write(f"field={field} entire_domain_TRAIN_mean={train_ge:.6f} "
                    f"entire_domain_UNSEEN_mean={unseen_ge:.6f}\n")

    print(f"\nSaved error maps to {config.FIG_DIR}/errormap_Re*.png (4 fields x 2 metrics each) and "
          f"summary to {config.FIG_DIR}/global_summary.txt")
    print("-> send both back, plus data/meta.txt's measured K, for the "
          "report's Section-10.2/10.3/11-equivalent content.")


if __name__ == "__main__":
    run_all()