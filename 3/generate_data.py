"""
generate_data.py
=================
Runs duct_solver.DuctEntranceSolver across the full Re sweep (train + test),
exactly mirroring generate_data.py's role in the channel and pipe projects:
produces the supervised-data target AND the developing-region validation
benchmark from ONE independent reference solver, used identically for every
Re (including the two held-out Re -- the solver does not know which Re are
"train" and which are "test", same discipline as before).

Also runs the Section-4.3-style self-consistency battery (via
duct_solver.self_consistency_report) BEFORE saving anything, and refuses to
proceed if the checks look bad -- same "trust the reference before trusting
the network" discipline as the pipe report.
"""

import os
import numpy as np
import config
from duct_solver import DuctEntranceSolver, self_consistency_report


def run_all(Ny=41, Nz=81, dx_fraction_of_Dh=0.02, mass_err_tol=1e-3,
            outlet_relL2_tol=0.05):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    print("=" * 70)
    print("STEP 1: self-consistency checks (must pass before data is trusted)")
    print("=" * 70)
    Le_measured = {}
    for Re in config.RE_ALL:
        Um = config.u_mean_for_Re(Re)
        L_guess = max(config.K_ENTRANCE_GUESS * config.DH * Re * 1.6, 1.0)
        rep = self_consistency_report(config.H, config.W, config.NU, Um,
                                       L_guess, Ny=31, Nz=61)
        print(f"Re={Re:4d}  mass_err={rep['mass_conservation_max_rel_err']:.2e}  "
              f"outlet_vs_exact_relL2={rep['outlet_vs_analytical_relL2']:.4f}  "
              f"Le={rep['Le']:.3f} m  K={rep['K_measured']:.4f}  "
              f"outlet|v,w|/Umean={rep['outlet_crossflow_over_Umean']:.4f}  "
              f"p'/p_main_ratio={rep['secondary_pressure_ratio']:.4f}")
        assert rep["mass_conservation_max_rel_err"] < mass_err_tol, \
            f"Re={Re}: mass conservation check FAILED, do not proceed"
        assert rep["outlet_vs_analytical_relL2"] < outlet_relL2_tol, \
            f"Re={Re}: outlet vs analytical check FAILED, do not proceed"
        Le_measured[Re] = rep["Le"]

    K_values = [Le_measured[Re] / (config.DH * Re) for Re in config.RE_ALL]
    K_mean = float(np.mean(K_values))
    print(f"\nMeasured entrance-length coefficient K (Le=K*Dh*Re): "
          f"{K_mean:.4f}  (spread {min(K_values):.4f}-{max(K_values):.4f})")
    print("(Report this measured K in the final report next to Han (1960)'s "
          "quoted functional form, same pattern as the pipe report Sec 6.)")

    L_final = max(K_mean * config.DH * max(config.RE_ALL) * 1.4, 2.0)
    print(f"\nUsing L_DOMAIN = {L_final:.3f} m for full-resolution runs "
          f"(covers Le at Re={max(config.RE_ALL)} with margin).")

    print("\n" + "=" * 70)
    print("STEP 2: full-resolution marching runs for every Re")
    print("=" * 70)
    solver = DuctEntranceSolver(config.H, config.W, config.NU, Ny=Ny, Nz=Nz)
    dx = dx_fraction_of_Dh * config.DH
    all_data = {}
    Le_full_res = {}
    for Re in config.RE_ALL:
        Um = config.u_mean_for_Re(Re)
        print(f"Marching Re={Re} ... (Um={Um:.4f} m/s, dx={dx:.5f} m, "
              f"~{int(L_final/dx)} stations)")
        res = solver.march(Um, L_final, dx, verbose=True)
        all_data[Re] = res
        # Development length measured on THIS full-resolution run (Ny, Nz,
        # dx as actually used for training/eval data) -- distinct from
        # Le_measured above, which came from the coarser Ny=31,Nz=61
        # Section-4.3 self-consistency grid used only to SIZE L_DOMAIN
        # before the expensive full run. Both are "measured, not looked
        # up" in the Han (1960) sense; this one is the one evaluate.py
        # uses to place developing-region stations at fixed fractions of
        # Le (see evaluate.py), so it must match the grid the data itself
        # was generated on.
        Le_here = solver.development_length(res)
        Le_full_res[Re] = float(Le_here)
        # V, W, P are now saved alongside U (see duct_solver.py's
        # station-local elliptic recovery of v, w, p -- module docstring
        # there for the method and its citations). Stations 0-2 carry a
        # top-hat-inlet grid singularity in v, w, p (see
        # self_consistency_report's documentation of this same artifact);
        # kept in the saved array (not deleted) but flagged via
        # crossflow_valid_from so downstream code can exclude them.
        np.savez_compressed(
            os.path.join(config.DATA_DIR, f"duct_Re{Re}.npz"),
            x=res["x"], U=res["U"], V=res["V"], W=res["W"], P=res["P"],
            G=res["G"], y=res["y"], z=res["z"],
            Re=Re, U_mean=Um, Le=Le_here, crossflow_valid_from=3,
        )
        print(f"  saved data/duct_Re{Re}.npz  (U shape {res['U'].shape}, "
              f"Le_full_res={Le_here:.3f} m)")

    with open(os.path.join(config.DATA_DIR, "meta.txt"), "w") as f:
        f.write(f"K_mean_measured={K_mean}\nL_DOMAIN_used={L_final}\n")
        for Re in config.RE_ALL:
            f.write(f"Le_Re{Re}_coarse_grid_step1={Le_measured[Re]}\n")
        for Re in config.RE_ALL:
            f.write(f"Le_Re{Re}_full_res_step2={Le_full_res[Re]}\n")

    # BUG FIX: config.L_DOMAIN was a static pre-run guess (5.0 m) that its
    # own comment claimed would be "overwritten by the measured value" but
    # nothing ever actually did that -- every OTHER script (train_hybrid.py's
    # PDE/BC collocation sampling, pinn_model.py's x/L_DOMAIN input
    # normalization) kept using 5.0 even though the actual marched/saved
    # domain is L_final (here, computed from the MEASURED entrance-length
    # coefficient K_mean). That mismatch meant roughly half the saved
    # reference domain (x > 5.0 m for a typical run) was never sampled
    # during PDE/BC training and fell outside the network's normalized
    # input range -- silently corrupting v, w, p accuracy (and, to a
    # lesser extent, u) at higher Re and further downstream. Writing
    # L_final here, and having config.py read it back in (see the bottom
    # of config.py), makes L_DOMAIN self-correcting: every script that
    # imports config after this file has been written uses the ACTUAL
    # domain the data spans, with no manual step required.
    with open(os.path.join(config.DATA_DIR, "L_domain_used.txt"), "w") as f:
        f.write(f"{L_final}\n")

    print("\nDone. K_mean, L_DOMAIN, and Le per Re written to data/meta.txt")
    print("-> send data/meta.txt content back along with training results.")


if __name__ == "__main__":
    run_all()