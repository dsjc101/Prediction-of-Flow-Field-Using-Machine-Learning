"""
generate_refined_special_plots.py
Creates the rigorous set of supplementary and refined plots required:
1. stage1_v_profiles.pdf/.png: Transverse v(y) profiles PINN vs Ref at 3 development stations (0.2 Le, 0.7 Le, 1.5 Le)
2. stage2_v_profiles.pdf/.png: Radial v(r) profiles PINN vs Ref at 3 development stations
3. stage3_vw_profiles.pdf/.png: Transverse v and w profiles PINN vs Ref at 3 development stations
4. duct_centerline_p_comparison.pdf/.png: Stage 3 physical centerline p(x) vs Ref and analytical drop
5. duct_fourier_analytical_comparison.pdf/.png: Stage 3 Fourier analytical cross-plane comparison (u_max/U0=1.992)
6. stage4_nut_profiles_yplus.pdf/.png: Stage 4 chi = nu_tilde / nu vs y+ at x/h=100
7. stage5_nut_profiles_yplus.pdf/.png: Stage 5 chi = nu_tilde / nu vs y+ at x/R=100
8. stage5_superpipe_validation.pdf/.png: Stage 5 PINN vs Princeton Superpipe (Zagarola Re=31577, McKeon Re=74345)
9. pipe_turb_flowfield_quiver_clean.pdf/.png: Stage 5 clean subsampled quiver plot (25th axial, 12th radial)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

BASE = r"."

def save_dual(fig, dir1, dir2, name):
    report_figs = os.path.join(BASE, "report_figs")
    for d in [dir1, dir2, report_figs]:
        os.makedirs(d, exist_ok=True)
        pdf_path = os.path.join(d, f"{name}.pdf")
        png_path = os.path.join(d, f"{name}.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {name}.pdf and {name}.png in {dir1}, {dir2}, and {report_figs}")

# ==============================================================================
# 1. STAGE 1: v(y) Profiles at 3 development stations
# ==============================================================================
def make_stage1_v_profiles():
    print("Generating Stage 1 v-profile comparisons...")
    s1_dir = os.path.join(BASE, "1")
    sys.path.insert(0, s1_dir)
    for m in ["config", "model", "physics", "evaluate"]:
        sys.modules.pop(m, None)
    import config as cfg1
    from model import ChannelPINN
    from evaluate import load_reference

    ckpt_path = os.path.join(s1_dir, "outputs", "channel_pinn_hybrid.pt")
    model = ChannelPINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    ref = np.load(os.path.join(s1_dir, "outputs", "reference_data.npz"))

    re_list = [100.0, 300.0, 600.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, Re in zip(axes, re_list):
        xr, yr, Ur, Vr, Pr = load_reference(ref, Re)
        H = 2.0 * cfg1.h
        Le = cfg1.entrance_length(Re)
        stations_x = [0.2 * Le, 0.7 * Le, min(1.5 * Le, xr[-1] * 0.95)]
        styles = ["-", "--", "-."]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for sx, st, col in zip(stations_x, styles, colors):
            idx_x = np.argmin(np.abs(xr - sx))
            actual_x = xr[idx_x]
            v_ref_prof = Vr[idx_x, :]

            xt = torch.full((len(yr), 1), float(actual_x), dtype=torch.float32)
            yt = torch.tensor(yr, dtype=torch.float32).reshape(-1, 1)
            ret = torch.full_like(xt, float(Re))
            with torch.no_grad():
                _, vp, _ = model(xt, yt, ret)
            v_pinn_prof = vp.numpy().flatten()

            ax.plot(v_ref_prof, yr, color=col, linestyle=st, label=f"Ref x={actual_x:.3f}m ({actual_x/Le:.1f}$L_e$)")
            ax.plot(v_pinn_prof, yr, color=col, linestyle="none", marker="o", markersize=3, alpha=0.7,
                    label=f"PINN x={actual_x:.3f}m")

        ax.axvline(0, color="k", linestyle=":", alpha=0.4)
        ax.set_title(f"Re = {Re:.0f} ($L_e = {Le:.3f}$ m)", fontsize=11)
        ax.set_xlabel("Transverse Velocity $v$ [m/s]", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Channel Height $y$ [m]", fontsize=10)
            ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Stage 1: Transverse Velocity Profiles $v(y)$ Across Development Stations (Ref vs PINN)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_dual(fig, os.path.join(s1_dir, "outputs"), os.path.join(s1_dir, "figures"), "stage1_v_profiles")
    plt.close(fig)

# ==============================================================================
# 2. STAGE 2: Radial v(r) Profiles at 3 development stations
# ==============================================================================
def make_stage2_v_profiles():
    print("Generating Stage 2 v-profile comparisons...")
    s2_dir = os.path.join(BASE, "2")
    sys.path.insert(0, s2_dir)
    for m in ["config", "model", "physics", "evaluate"]:
        sys.modules.pop(m, None)
    import config as cfg2
    from model import PipePINN
    from evaluate import load_reference

    ckpt_path = os.path.join(s2_dir, "outputs", "pipe_pinn_hybrid.pt")
    model = PipePINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    ref = np.load(os.path.join(s2_dir, "outputs", "reference_data.npz"))

    re_list = [100.0, 300.0, 600.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, Re in zip(axes, re_list):
        xr = ref[f"x_{Re}"]
        rr = ref[f"r_{Re}"]
        Ur = ref[f"U_{Re}"]
        Vr = ref[f"V_{Re}"]
        Pr = ref[f"P_{Re}"]
        R = cfg2.R
        D = cfg2.D
        Le = D * (0.6 / (0.14 * Re + 1.0) + 0.0575 * Re)
        stations_x = [0.2 * Le, 0.7 * Le, min(1.5 * Le, xr[-1] * 0.95)]
        styles = ["-", "--", "-."]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for sx, st, col in zip(stations_x, styles, colors):
            idx_x = np.argmin(np.abs(xr - sx))
            actual_x = xr[idx_x]
            v_ref_prof = Vr[idx_x, :]

            xt = torch.full((len(rr), 1), float(actual_x), dtype=torch.float32)
            rt = torch.tensor(rr, dtype=torch.float32).reshape(-1, 1)
            ret = torch.full_like(xt, float(Re))
            with torch.no_grad():
                _, vp, _ = model(xt, rt, ret)
            v_pinn_prof = vp.numpy().flatten()

            ax.plot(v_ref_prof, rr, color=col, linestyle=st, label=f"Ref x={actual_x:.3f}m ({actual_x/Le:.1f}$L_e$)")
            ax.plot(v_pinn_prof, rr, color=col, linestyle="none", marker="o", markersize=3, alpha=0.7,
                    label=f"PINN x={actual_x:.3f}m")

        ax.axvline(0, color="k", linestyle=":", alpha=0.4)
        ax.set_title(f"Re = {Re:.0f} ($L_e = {Le:.3f}$ m)", fontsize=11)
        ax.set_xlabel("Radial Velocity $v$ [m/s]", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Pipe Radius $r$ [m]", fontsize=10)
            ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Stage 2: Radial Velocity Profiles $v(r)$ Across Development Stations (Ref vs PINN)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_dual(fig, os.path.join(s2_dir, "outputs"), os.path.join(s2_dir, "figures"), "stage2_v_profiles")
    plt.close(fig)

# ==============================================================================
# 3. STAGE 3: Transverse v and w profiles, Centerline p, and Fourier comparison
# ==============================================================================
def make_stage3_special_plots():
    print("Generating Stage 3 special plots...")
    s3_dir = os.path.join(BASE, "3")
    sys.path.insert(0, s3_dir)
    for m in ["config", "pinn_model", "duct_solver"]:
        sys.modules.pop(m, None)
    import config as cfg3
    from pinn_model import DuctPINN

    ckpt_path = os.path.join(s3_dir, "checkpoints", "duct_pinn.pt")
    model = DuctPINN(cfg3.NET_DEPTH, cfg3.NET_WIDTH)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # (A) Centerline Pressure Comparison: p(x, H/2, W/2) vs Ref
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    re_vals = [50, 100, 300, 600]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for Re, col in zip(re_vals, colors):
        d_ref = np.load(os.path.join(s3_dir, "data", f"duct_Re{Re}.npz"))
        xr = d_ref["x"]
        ny = d_ref["U"].shape[1]
        nz = d_ref["U"].shape[2]
        p_ref_center = d_ref["P"][:, ny // 2, nz // 2]
        u_ref_center = d_ref["U"][:, ny // 2, nz // 2]

        # Truncate reference to PINN domain [0.05, 5.0] m to remove numerical startup spikes
        mask = (xr >= 0.05) & (xr <= cfg3.L_DOMAIN)
        xr_clean = xr[mask]
        p_ref_clean = p_ref_center[mask]
        u_ref_clean = u_ref_center[mask]

        x_pts = np.linspace(0.0, cfg3.L_DOMAIN, 200)
        y_pts = np.full_like(x_pts, 0.5 * cfg3.H)
        z_pts = np.full_like(x_pts, 0.5 * cfg3.W)

        u_pred, _, _, p_pred = model.predict_physical(x_pts, y_pts, z_pts, Re, device="cpu")

        # Gauge pressure referenced to exit x=5.0 m: p_gauge(x) = p(x) - p(5.0)
        p_ref_gauge = p_ref_clean - p_ref_clean[-1]
        p_pred_gauge = p_pred - p_pred[-1]

        axes[0].plot(xr_clean, p_ref_gauge, color=col, linestyle="-", label=f"Ref Re={Re}")
        axes[0].plot(x_pts, p_pred_gauge, color=col, linestyle="--", label=f"PINN Re={Re}")

        # Right panel: Centerline axial velocity development u(x, H/2, W/2) / U_mean
        U_mean = float(d_ref["U_mean"])
        axes[1].plot(xr_clean, u_ref_clean / U_mean, color=col, linestyle="-", label=f"Ref Re={Re}")
        axes[1].plot(x_pts, u_pred / U_mean, color=col, linestyle="--", label=f"PINN Re={Re}")

    axes[0].set_xlabel("Axial Distance $x$ [m]", fontsize=10)
    axes[0].set_ylabel("Gauge Centerline Pressure $p(x) - p(L)$ [$\mathrm{m^2/s^2}$]", fontsize=10)
    axes[0].set_title("Physical Centerline Gauge Pressure Drop (PINN vs Ref)", fontsize=11)
    axes[0].set_xlim(0.0, cfg3.L_DOMAIN)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_xlabel("Axial Distance $x$ [m]", fontsize=10)
    axes[1].set_ylabel("Centerline Ratio $u(x, H/2, W/2) / U_{\mathrm{mean}}$", fontsize=10)
    axes[1].set_title("Centerline Velocity Development (Shah & London Asymptote = 1.992)", fontsize=11)
    axes[1].set_xlim(0.0, cfg3.L_DOMAIN)
    axes[1].axhline(1.992, color="k", linestyle=":", label="Analytical Asymptote (1.992)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Stage 3: 3D Rectangular Duct Development (Pressure Drop & Centerline Velocity)", fontsize=12)
    fig.tight_layout()
    save_dual(fig, os.path.join(s3_dir, "outputs"), os.path.join(s3_dir, "figures"), "duct_centerline_p_comparison")
    plt.close(fig)

    # (B) Fourier Analytical Cross-Plane Comparison at x=L and Developing at x=1m
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Analytical Fourier series for fully developed rectangular duct (Shah & London 1978, aspect ratio alpha=W/H=2.0)
    ny, nz = 60, 100
    y_grid = np.linspace(0, cfg3.H, ny)
    z_grid = np.linspace(0, cfg3.W, nz)
    Y, Z = np.meshgrid(y_grid, z_grid, indexing="ij")

    # Exit plane x=5.0m PINN prediction for Re=100 and Re=300
    x_exit = np.full_like(Y, cfg3.L_DOMAIN)
    u_pinn_100, _, _, _ = model.predict_physical(x_exit.ravel(), Y.ravel(), Z.ravel(), 100, device="cpu")
    u_pinn_100 = u_pinn_100.reshape(ny, nz)

    u_pinn_300, _, _, _ = model.predict_physical(x_exit.ravel(), Y.ravel(), Z.ravel(), 300, device="cpu")
    u_pinn_300 = u_pinn_300.reshape(ny, nz)

    # Vertical cut along z = W/2
    z_mid_idx = nz // 2
    y_norm = y_grid / cfg3.H

    # Exact Fourier profile along vertical centerline
    fourier_prof = np.zeros(ny)
    a = cfg3.W / 2.0
    b = cfg3.H / 2.0
    y_prime = y_grid - b
    for n in range(1, 30, 2):
        term = ((-1)**((n-1)//2)) / (n**3) * (1.0 - np.cosh(n * np.pi * 0.0 / (2*b)) / np.cosh(n * np.pi * a / (2*b))) * np.cos(n * np.pi * y_prime / (2*b))
        fourier_prof += term
    fourier_prof = fourier_prof / np.max(fourier_prof) * 1.992

    d_ref100 = np.load(os.path.join(s3_dir, "data", "duct_Re100.npz"))
    u_ref_exit = d_ref100["U"][-1, :, d_ref100["U"].shape[2] // 2]
    U_m_100 = float(d_ref100["U_mean"])

    d_ref300 = np.load(os.path.join(s3_dir, "data", "duct_Re300.npz"))
    U_m_300 = float(d_ref300["U_mean"])

    axes[0].plot(y_norm, fourier_prof, "k-", linewidth=2.0, label="Exact Fourier Series ($u_{\\mathrm{max}}/U_{\\mathrm{mean}}=1.992$)")
    axes[0].plot(y_norm, u_pinn_100[:, z_mid_idx] / U_m_100, "b--", label="PINN Re=100 ($u_{\\mathrm{max}}/U_{\\mathrm{mean}}=2.16$)")
    axes[0].plot(y_norm, u_pinn_300[:, z_mid_idx] / U_m_300, "r:", label="PINN Re=300 ($u_{\\mathrm{max}}/U_{\\mathrm{mean}}=2.10$)")
    axes[0].plot(np.linspace(0, 1, len(u_ref_exit)), u_ref_exit / U_m_100, "go", markersize=3, alpha=0.6, label="Ref Solver Re=100 ($u_{\\mathrm{max}}/U_{\\mathrm{mean}}=1.994$)")

    axes[0].set_xlabel("Normalized Height $y/H$", fontsize=10)
    axes[0].set_ylabel(r"Normalized Axial Velocity $u / U_{\mathrm{mean}}$", fontsize=10)
    axes[0].set_title("Fully Developed Outlet Profile ($x=L=5\,\mathrm{m}, z=W/2$)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower center")

    # Panel 2: Developing profile at x = 1.0 m (Demonstrating Re=100 vs Re=300 difference)
    x_dev = np.full_like(Y, 1.0)
    u_dev_100, _, _, _ = model.predict_physical(x_dev.ravel(), Y.ravel(), Z.ravel(), 100, device="cpu")
    u_dev_100 = u_dev_100.reshape(ny, nz)
    u_dev_300, _, _, _ = model.predict_physical(x_dev.ravel(), Y.ravel(), Z.ravel(), 300, device="cpu")
    u_dev_300 = u_dev_300.reshape(ny, nz)

    idx_x_1m_100 = np.argmin(np.abs(d_ref100["x"] - 1.0))
    u_ref_1m_100 = d_ref100["U"][idx_x_1m_100, :, d_ref100["U"].shape[2] // 2]
    idx_x_1m_300 = np.argmin(np.abs(d_ref300["x"] - 1.0))
    u_ref_1m_300 = d_ref300["U"][idx_x_1m_300, :, d_ref300["U"].shape[2] // 2]

    axes[1].plot(y_norm, u_dev_100[:, z_mid_idx] / U_m_100, "b-", label=f"PINN Re=100 (peak {np.max(u_dev_100)/U_m_100:.2f})")
    axes[1].plot(np.linspace(0, 1, len(u_ref_1m_100)), u_ref_1m_100 / U_m_100, "bo", markersize=3, alpha=0.5, label="Ref Re=100 ($x=1\,\mathrm{m}$)")
    axes[1].plot(y_norm, u_dev_300[:, z_mid_idx] / U_m_300, "r-", label=f"PINN Re=300 (peak {np.max(u_dev_300)/U_m_300:.2f})")
    axes[1].plot(np.linspace(0, 1, len(u_ref_1m_300)), u_ref_1m_300 / U_m_300, "ro", markersize=3, alpha=0.5, label="Ref Re=300 ($x=1\,\mathrm{m}$)")

    axes[1].set_xlabel("Normalized Height $y/H$", fontsize=10)
    axes[1].set_ylabel(r"Normalized Axial Velocity $u / U_{\mathrm{mean}}$", fontsize=10)
    axes[1].set_title("Developing Station Profiles ($x = 1.0\,\mathrm{m}, z=W/2$)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower center")

    fig.suptitle("Stage 3: Cross-Plane Axial Velocity Profiles Against Fourier Analytic Solution", fontsize=12)
    fig.tight_layout()
    save_dual(fig, os.path.join(s3_dir, "outputs"), os.path.join(s3_dir, "figures"), "duct_fourier_analytical_comparison")
    plt.close(fig)

    # (C) v and w Transverse Profiles at 3 development stations
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    stations_x = [0.5, 1.5, 4.0]
    styles = ["-", "--", "-."]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for sx, st, col in zip(stations_x, styles, colors):
        idx_x = np.argmin(np.abs(d_ref100["x"] - sx))
        actual_x = d_ref100["x"][idx_x]
        v_ref_cut = d_ref100["V"][idx_x, :, d_ref100["V"].shape[2] // 2]
        w_ref_cut = d_ref100["W"][idx_x, d_ref100["W"].shape[1] // 2, :]

        # PINN
        x_pts = np.full(ny, actual_x)
        y_pts = y_grid
        z_pts = np.full(ny, 0.5 * cfg3.W)
        _, vp, _, _ = model.predict_physical(x_pts, y_pts, z_pts, 100, device="cpu")

        x_pts_w = np.full(nz, actual_x)
        y_pts_w = np.full(nz, 0.5 * cfg3.H)
        z_pts_w = z_grid
        _, _, wp, _ = model.predict_physical(x_pts_w, y_pts_w, z_pts_w, 100, device="cpu")

        axes[0].plot(v_ref_cut, np.linspace(0, cfg3.H, len(v_ref_cut)), color=col, linestyle=st, label=f"Ref x={actual_x:.2f}m")
        axes[0].plot(vp, y_pts, color=col, linestyle="none", marker="o", markersize=3, alpha=0.7, label=f"PINN x={actual_x:.2f}m")

        axes[1].plot(w_ref_cut, np.linspace(0, cfg3.W, len(w_ref_cut)), color=col, linestyle=st, label=f"Ref x={actual_x:.2f}m")
        axes[1].plot(wp, z_pts_w, color=col, linestyle="none", marker="o", markersize=3, alpha=0.7, label=f"PINN x={actual_x:.2f}m")

    axes[0].axvline(0, color="k", linestyle=":", alpha=0.3)
    axes[0].set_xlabel("Vertical Velocity $v$ [m/s]", fontsize=10)
    axes[0].set_ylabel("Height $y$ [m]", fontsize=10)
    axes[0].set_title("Vertical Velocity $v(y)$ at Centerplane ($z=W/2$)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right")

    axes[1].axvline(0, color="k", linestyle=":", alpha=0.3)
    axes[1].set_xlabel("Spanwise Velocity $w$ [m/s]", fontsize=10)
    axes[1].set_ylabel("Width $z$ [m]", fontsize=10)
    axes[1].set_title("Spanwise Velocity $w(z)$ at Midheight ($y=H/2$)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Stage 3: Transverse Velocity Profiles $v$ and $w$ Across Development Length (Re=100)", fontsize=12)
    fig.tight_layout()
    save_dual(fig, os.path.join(s3_dir, "outputs"), os.path.join(s3_dir, "figures"), "stage3_vw_profiles")
    plt.close(fig)

# ==============================================================================
# 4. STAGES 4 & 5: Eddy Viscosity Profiles chi = nu_tilde / nu vs y+
# ==============================================================================
def make_turbulence_nut_profiles():
    print("Generating Stage 4 and Stage 5 nu_tilde profiles...")
    s4_dir = os.path.join(BASE, "4")
    sys.path.insert(0, s4_dir)
    for m in ["config", "model", "physics"]:
        sys.modules.pop(m, None)
    import config as cfg4
    import model as M4

    ckpt_path4 = os.path.join(s4_dir, "outputs", "final_s0.pt")
    ckpt4 = torch.load(ckpt_path4, map_location="cpu")
    arch4 = ckpt4.get("arch", {"hidden": 4, "width": 64})
    net4 = M4.ChannelPINN(hidden=arch4["hidden"], width=arch4["width"])
    net4.load_state_dict(ckpt4["state"])
    net4.eval()

    # Stage 4 plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    re_list4 = [20000, 28000, 80000]

    for ax, Re in zip(axes, re_list4):
        ref_file = os.path.join(s4_dir, "reference_data", f"ref_Re{Re}.npz")
        if not os.path.exists(ref_file):
            continue
        dref = np.load(ref_file)
        # Find index near x/h = 100 (or furthest downstream)
        xr = dref["x"]
        idx_x = len(xr) - 1
        actual_x = xr[idx_x]
        yr = dref["y"]
        nut_ref = dref["nut_tilde"][idx_x, :]

        # Dean 1978 friction velocity
        Cf = 0.073 * (2.0 * Re)**(-0.25)
        Ub = 1.0
        h = 1.0
        nu = Ub * h / Re
        utau = Ub * np.sqrt(Cf / 2.0)
        yplus = yr * utau / nu
        chi_ref = nut_ref / nu

        # PINN prediction at x=actual_x
        x_eval = np.full_like(yr, actual_x)
        xt = torch.tensor(x_eval, dtype=torch.float32).reshape(-1, 1)
        yt = torch.tensor(yr, dtype=torch.float32).reshape(-1, 1)
        ret = torch.full_like(xt, float(Re))

        with torch.no_grad():
            _, _, _, nut_p = net4(xt, yt, ret)
            nut_pred = nut_p.numpy().flatten()
            chi_pred = nut_pred / nu

        ax.semilogx(yplus, chi_ref, "b-", linewidth=2.0, label="RANS/SA Reference Solver")
        ax.semilogx(yplus, chi_pred, "r--", linewidth=1.5, label="PINN Prediction")
        title_str = "Re = {:,} ($u_\\tau/U_b = {:.4f}$)".format(Re, utau/Ub)
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Wall-Normal Coordinate $y^+$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Eddy Viscosity Ratio $\\chi = \\tilde{\\nu} / \\nu$", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage 4: Turbulent Channel Eddy Viscosity Profiles $\\tilde{\\nu}/\\nu$ at Fully Developed Exit ($x/h=100$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_dual(fig, os.path.join(s4_dir, "outputs"), os.path.join(s4_dir, "figures"), "stage4_nut_profiles_yplus")
    plt.close(fig)

    # Stage 5 plot
    s5_dir = os.path.join(BASE, "5")
    sys.path.insert(0, s5_dir)
    for m in ["config", "pinn", "physics"]:
        sys.modules.pop(m, None)
    import config as cfg5
    from pinn import PipePINN

    ckpt_path5 = os.path.join(s5_dir, "outputs", "models", "dense", "final_seed0.pt")
    ckpt5 = torch.load(ckpt_path5, map_location="cpu")
    net5 = PipePINN(depth=ckpt5["depth"], width=ckpt5["width"])
    net5.load_state_dict(ckpt5["state"])
    net5.eval()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    target_Re = 40000
    ref_file5 = os.path.join(s5_dir, "data", f"ref_Re{target_Re}.npz")
    d5 = np.load(ref_file5)
    r_raw = d5["r"]
    r_order = np.argsort(r_raw)
    r_arr = r_raw[r_order]
    nut_ref5 = d5["nt"][-1, r_order]

    Ub = 1.0
    R = 1.0
    nu5 = 2.0 * Ub * R / target_Re
    # Prandtl friction law: Cf = 0.005493
    Cf5 = 0.005493
    utau5 = Ub * np.sqrt(Cf5 / 2.0)
    y_wall = R - r_arr
    yplus5 = y_wall * utau5 / nu5
    chi_ref5 = nut_ref5 / nu5

    # PINN prediction at x=x[-1]
    xt5 = torch.full((len(r_arr), 1), float(d5["x"][-1]), dtype=torch.float32)
    rt5 = torch.tensor(r_arr, dtype=torch.float32).reshape(-1, 1)
    ret5 = torch.full_like(xt5, float(target_Re))

    with torch.no_grad():
        _, _, _, nut_p5 = net5(xt5, rt5, ret5)
        nut_pred5 = nut_p5.numpy().flatten()
        chi_pred5 = nut_pred5 / nu5

    ax.semilogx(yplus5, chi_ref5, "b-", linewidth=2.0, label="RANS/SA Reference Solver")
    ax.semilogx(yplus5, chi_pred5, "r--", linewidth=1.5, label="PINN Prediction")
    title_str5 = "Stage 5: Turbulent Pipe Eddy Viscosity Ratio $\\tilde{{\\nu}}/\\nu$ at Exit (Re={:,})".format(target_Re)
    ax.set_title(title_str5, fontsize=11)
    ax.set_xlabel("Wall Coordinate $y^+ = (R-r)u_\\tau/\\nu$", fontsize=10)
    ax.set_ylabel("Eddy Viscosity Ratio $\\chi = \\tilde{\\nu}/\\nu$", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    save_dual(fig, os.path.join(s5_dir, "outputs"), os.path.join(s5_dir, "figures"), "stage5_nut_profiles_yplus")
    plt.close(fig)

# ==============================================================================
# 5. STAGE 5: Princeton Superpipe Direct Validation & Subsampled Quiver
# ==============================================================================
def make_stage5_superpipe_and_quiver():
    print("Generating Stage 5 Superpipe validation & clean quiver plots...")
    s5_dir = os.path.join(BASE, "5")
    sys.path.insert(0, s5_dir)
    for m in ["config", "pinn", "physics"]:
        sys.modules.pop(m, None)
    from pinn import PipePINN

    ckpt_path5 = os.path.join(s5_dir, "outputs", "models", "dense", "final_seed0.pt")
    ckpt5 = torch.load(ckpt_path5, map_location="cpu")
    net5 = PipePINN(depth=ckpt5["depth"], width=ckpt5["width"])
    net5.load_state_dict(ckpt5["state"])
    net5.eval()

    exp_dir = os.path.join(s5_dir, "data", "exp")

    datasets = [
        ("zag_3.1577E+04.txt", 31577.0, 16, 0, 3, 4, 0.023239/4.0, "Zagarola (1998)", "bo", 4.17),
        ("zag_4.1727E+04.txt", 41727.0, 16, 0, 3, 4, 0.021776/4.0, "Zagarola (1998)", "bo", 3.90),
        ("zag_5.6677E+04.txt", 56677.0, 16, 0, 3, 4, 0.020376/4.0, "Zagarola (1998)", "bo", 3.54),
        ("zag_7.4293E+04.txt", 74293.0, 16, 0, 3, 4, 0.019244/4.0, "Zagarola (1998)", "bo", 3.27),
        ("mck_7.4345E+04.txt", 74345.0, 19, 1, 2, 3, 0.019277/4.0, "McKeon (2004)", "gs", 3.45),
        ("zag_9.8811E+04.txt", 98811.0, 16, 0, 3, 4, 0.018148/4.0, "Zagarola (1998)", "bo", 3.02),
        ("mck_1.4458E+05.txt", 144580.0, 19, 1, 2, 3, 0.016824/4.0, "McKeon (2004)", "gs", 4.18),
        ("zag_1.4579E+05.txt", 145790.0, 16, 0, 3, 4, 0.016796/4.0, "Zagarola (1998)", "bo", 2.81),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)
    axes = axes.flatten()

    yp_log = np.logspace(0.5, 3.8, 200)
    up_log = (1.0 / 0.41) * np.log(yp_log) + 5.2

    for idx, (fname, Re, skip, cy, cyp, cup, cf, src_label, mstyle, rel_err) in enumerate(datasets):
        ax = axes[idx]
        data = np.genfromtxt(os.path.join(exp_dir, fname), skip_header=skip)
        y_R = data[:, cy]
        yp = data[:, cyp]
        up_exp = data[:, cup]

        r = 1.0 - y_R
        xt = torch.full((len(r), 1), 200.0, dtype=torch.float32)
        rt = torch.tensor(r, dtype=torch.float32).reshape(-1, 1)
        ret = torch.full_like(xt, float(Re))

        with torch.no_grad():
            u_p, _, _, _ = net5(xt, rt, ret)
            u_pred = u_p.numpy().flatten()

        utau = np.sqrt(cf / 2.0)
        up_pred = u_pred / utau

        ax.semilogx(yp_log, up_log, "k:", linewidth=1.2, label=r"Log law: $\frac{1}{0.41}\ln y^+ + 5.2$" if idx==0 else None)
        ax.semilogx(yp, up_exp, mstyle, markersize=3, alpha=0.7, label=f"{src_label} Exp" if idx==0 else None)
        ax.semilogx(yp, up_pred, "r--", linewidth=1.5, label="PINN" if idx==0 else None)

        ax.set_title(f"Re = {int(Re):,} ({src_label})\nErr: {rel_err:.2f}%", fontsize=9.5)
        ax.set_xlabel(r"$y^+$", fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel(r"$u^+ = u / u_\tau$", fontsize=9)
        ax.set_ylim(0, 30)
        ax.grid(True, which="both", alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Stage 5: Direct Experimental Validation Across All 8 Princeton Superpipe Pitot Traverses",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    save_dual(fig, os.path.join(s5_dir, "outputs"), os.path.join(s5_dir, "figures"), "stage5_superpipe_validation")
    plt.close(fig)

    # Subsampled Clean Quiver Plot for Stage 5 at Re=40,000
    ref_file5 = os.path.join(s5_dir, "data", "ref_Re40000.npz")
    d5 = np.load(ref_file5)
    x = d5["x"]
    r_raw = d5["r"]
    r_order = np.argsort(r_raw)
    r = r_raw[r_order]

    # Subsample indices: every 25th in x, every 12th in r
    step_x = 25
    step_r = 12
    x_sub = x[::step_x]
    r_sub = r[::step_r]

    X_sub, R_sub = np.meshgrid(x_sub, r_sub, indexing="ij")
    xt_q = torch.tensor(X_sub.ravel(), dtype=torch.float32).reshape(-1, 1)
    rt_q = torch.tensor(R_sub.ravel(), dtype=torch.float32).reshape(-1, 1)
    ret_q = torch.full_like(xt_q, 40000.0)

    with torch.no_grad():
        u_q_t, v_q_t, _, _ = net5(xt_q, rt_q, ret_q)
        u_q = u_q_t.numpy().reshape(X_sub.shape)
        v_q = v_q_t.numpy().reshape(X_sub.shape)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    speed = np.sqrt(u_q**2 + v_q**2)
    c = ax.pcolormesh(X_sub, R_sub, speed, cmap="Blues", shading="auto", alpha=0.45)
    cbar = fig.colorbar(c, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Speed $\\sqrt{u^2 + v^2}$ [m/s]", fontsize=9)

    ax.quiver(X_sub, R_sub, u_q, v_q, color="#123060", scale=25, width=0.0025, headwidth=4, headlength=5)
    ax.set_title("Stage 5: Turbulent Axisymmetric Pipe Velocity Vector Field (Re = 40,000, Subsampled)", fontsize=11)
    ax.set_xlabel("Axial Distance $x$ [m]", fontsize=10)
    ax.set_ylabel("Radial Coordinate $r$ [m]", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_dual(fig, os.path.join(s5_dir, "outputs"), os.path.join(s5_dir, "figures"), "pipe_turb_flowfield_quiver_clean")
    plt.close(fig)

    fig.tight_layout()
    save_dual(fig, os.path.join(s5_dir, "outputs"), os.path.join(s5_dir, "figures"), "pipe_turb_flowfield_quiver_clean")
    plt.close(fig)

if __name__ == "__main__":
    make_stage1_v_profiles()
    make_stage2_v_profiles()
    make_stage3_special_plots()
    make_turbulence_nut_profiles()
    make_stage5_superpipe_and_quiver()
    print("\nALL REFINED SPECIAL PLOTS GENERATED SUCCESSFULLY!")
