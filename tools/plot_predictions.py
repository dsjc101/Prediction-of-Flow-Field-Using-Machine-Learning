"""
tools/plot_predictions.py
Generates high-resolution predicted flow-field contour figures for all 5 stages.
Follows exact formatting rules:
- font.size: 9, axes.labelsize: 9, xtick.labelsize: 8, ytick.labelsize: 8
- NO ax.set_title, NO fig.suptitle
- figure size: 6.4 x 2.6 in per panel, constrained_layout=True
- savefig(..., dpi=300) WITHOUT bbox_inches='tight'
- y-label short: "$y$ [m]" (x-label "$x$ [m]" or "$z$ [m]" as applicable); symbol and unit only
- colorbar: cb.locator = MaxNLocator(5), cb.formatter = FormatStrFormatter('%.3g')
  (for v use ScalarFormatter with scientific offset placed above the bar),
  cb.set_label(symbol + unit only), pad=0.02, fraction=0.046
- exact data, color maps, and limits preserved
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "report_figs", "pred")
os.makedirs(OUT_DIR, exist_ok=True)

NX_2D, NY_2D = 350, 120
NY_3D, NZ_3D = 120, 120
NX_3D = 350

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

def save_contour(X, Y, Z, xlabel, ylabel, cbar_label, out_name, cmap="viridis", is_v=False):
    fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=8/72, h_pad=10/72)
    cp = ax.contourf(X, Y, Z, levels=60, cmap=cmap)
    cb = fig.colorbar(cp, ax=ax, pad=0.02, fraction=0.046)
    cb.locator = MaxNLocator(5)
    if is_v:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        cb.formatter = fmt
    else:
        cb.formatter = FormatStrFormatter("%.3g")
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_fontsize(8)
    cb.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=300)
    plt.close(fig)
    print(f"  Saved {out_name} (grid: {X.shape}), range: [{float(Z.min()):.4e}, {float(Z.max()):.4e}]")

def save_streamplot(X, Y, U, V, xlabel, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=8/72, h_pad=10/72)
    speed = np.sqrt(U**2 + V**2)
    x_1d = X[:, 0]
    y_1d = Y[0, :]
    strm = ax.streamplot(x_1d, y_1d, U.T, V.T, color=speed.T, cmap="viridis", density=0.7, linewidth=1.5, arrowsize=1.3)
    cb = fig.colorbar(strm.lines, ax=ax, pad=0.02, fraction=0.046)
    cb.locator = MaxNLocator(5)
    cb.formatter = FormatStrFormatter("%.3g")
    cb.update_ticks()
    cb.set_label(r"$|\mathbf{u}|$ [m/s]")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=300)
    plt.close(fig)
    print(f"  Saved {out_name} (grid: {X.shape}), speed range: [{float(speed.min()):.4e}, {float(speed.max()):.4e}]")

def plot_stage1():
    print("Plotting Stage 1 predictions...")
    s1_dir = os.path.join(BASE_DIR, "1")
    orig_cwd = os.getcwd()
    try:
        os.chdir(s1_dir)
        sys.path.insert(0, s1_dir)
        sys.modules.pop("model", None)
        sys.modules.pop("config", None)
        import config as cfg1
        from model import ChannelPINN

        ckpt_path = os.path.join(s1_dir, "outputs", "channel_pinn_hybrid.pt")
        device = torch.device("cpu")
        model = ChannelPINN(n_hidden=4, width=80).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for Re in [100, 600]:
            x_1d = np.linspace(0.0, cfg1.L, NX_2D)
            y_1d = np.linspace(-cfg1.h, cfg1.h, NY_2D)
            X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")
            
            xt = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
            yt = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
            ret = torch.full_like(xt, float(Re))
            
            with torch.no_grad():
                u_norm, v_norm, p_norm = model(xt, yt, ret)
                
            u = u_norm.numpy().reshape(X.shape)
            v = v_norm.numpy().reshape(X.shape)
            p = p_norm.numpy().reshape(X.shape)
            
            save_contour(X, Y, u, "$x$ [m]", "$y$ [m]",
                         r"$u$ [m/s]", f"stage1_pred_u_Re{Re}.png", cmap="viridis", is_v=False)
            save_contour(X, Y, v, "$x$ [m]", "$y$ [m]",
                         r"$v$ [m/s]", f"stage1_pred_v_Re{Re}.png", cmap="coolwarm", is_v=True)
            save_contour(X, Y, p, "$x$ [m]", "$y$ [m]",
                         r"$p$ [m$^2$/s$^2$]", f"stage1_pred_p_Re{Re}.png", cmap="plasma", is_v=False)
            save_streamplot(X, Y, u, v, "$x$ [m]", "$y$ [m]",
                            f"stage1_pred_streamlines_Re{Re}.png")
    finally:
        os.chdir(orig_cwd)

def plot_stage2():
    print("Plotting Stage 2 predictions...")
    s2_dir = os.path.join(BASE_DIR, "2")
    orig_cwd = os.getcwd()
    try:
        os.chdir(s2_dir)
        sys.path.insert(0, s2_dir)
        sys.modules.pop("model", None)
        sys.modules.pop("config", None)
        import config as cfg2
        from model import PipePINN

        ckpt_path = os.path.join(s2_dir, "checkpoints", "pipe_pinn.pt")
        device = torch.device("cpu")
        model = PipePINN(n_hidden=4, width=80).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for Re in [100, 600]:
            x_1d = np.linspace(0.0, cfg2.L, NX_2D)
            r_1d = np.linspace(0.0, cfg2.R, NY_2D)
            X, R = np.meshgrid(x_1d, r_1d, indexing="ij")
            
            xt = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
            rt = torch.tensor(R.flatten(), dtype=torch.float32).unsqueeze(1)
            ret = torch.full_like(xt, float(Re))
            
            with torch.no_grad():
                ux_t, ur_t, p_t = model(xt, rt, ret)
                
            ux = ux_t.numpy().reshape(X.shape)
            ur = ur_t.numpy().reshape(X.shape)
            p = p_t.numpy().reshape(X.shape)
            
            save_contour(X, R, ux, "$x$ [m]", "$r$ [m]",
                         r"$u_x$ [m/s]", f"stage2_pred_ux_Re{Re}.png", cmap="viridis", is_v=False)
            save_contour(X, R, ur, "$x$ [m]", "$r$ [m]",
                         r"$u_r$ [m/s]", f"stage2_pred_ur_Re{Re}.png", cmap="coolwarm", is_v=True)
            save_contour(X, R, p, "$x$ [m]", "$r$ [m]",
                         r"$p$ [m$^2$/s$^2$]", f"stage2_pred_p_Re{Re}.png", cmap="plasma", is_v=False)
            save_streamplot(X, R, ux, ur, "$x$ [m]", "$r$ [m]",
                            f"stage2_pred_streamlines_Re{Re}.png")
    finally:
        os.chdir(orig_cwd)

def plot_stage3():
    print("Plotting Stage 3 predictions...")
    s3_dir = os.path.join(BASE_DIR, "3")
    orig_cwd = os.getcwd()
    try:
        os.chdir(s3_dir)
        sys.path.insert(0, s3_dir)
        sys.modules.pop("model", None)
        sys.modules.pop("config", None)
        import config as cfg3
        from pinn_model import DuctPINN

        ckpt_path = os.path.join(s3_dir, "checkpoints", "duct_pinn.pt")
        device = torch.device("cpu")
        model = DuctPINN().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for Re in [100, 600]:
            y_1d = np.linspace(0.0, cfg3.H, NY_3D)
            z_1d = np.linspace(0.0, cfg3.W, NZ_3D)
            Y, Z = np.meshgrid(y_1d, z_1d, indexing="ij")

            # Cross-planes at x = 1.0m and x = 5.0m
            for xs, xname in [(1.0, "x1m"), (5.0, "x5m")]:
                x_pts = np.full_like(Y, xs)
                u_phys, v_phys, w_phys, _ = model.predict_physical(x_pts.flatten(), Y.flatten(), Z.flatten(), Re, device="cpu")
                U_pl = u_phys.reshape(Y.shape)
                V_pl = v_phys.reshape(Y.shape)
                W_pl = w_phys.reshape(Y.shape)
                save_contour(Z, Y, U_pl, "$z$ [m]", "$y$ [m]", r"$u$ [m/s]", f"stage3_pred_u_{xname}_Re{Re}.png", cmap="viridis", is_v=False)
                save_contour(Z, Y, V_pl, "$z$ [m]", "$y$ [m]", r"$v$ [m/s]", f"stage3_pred_v_{xname}_Re{Re}.png", cmap="coolwarm", is_v=True)
                save_contour(Z, Y, W_pl, "$z$ [m]", "$y$ [m]", r"$w$ [m/s]", f"stage3_pred_w_{xname}_Re{Re}.png", cmap="coolwarm", is_v=True)

            # Mid-plane u contour
            x_long = np.linspace(0.0, cfg3.L_DOMAIN, NX_3D)
            y_long = np.linspace(0.0, cfg3.H, NY_3D)
            X_mid, Y_mid = np.meshgrid(x_long, y_long, indexing="ij")
            Z_mid = np.full_like(X_mid, 0.5 * cfg3.W)
            
            u_mid_f, _, _, _ = model.predict_physical(X_mid.flatten(), Y_mid.flatten(), Z_mid.flatten(), Re, device="cpu")
            u_mid = u_mid_f.reshape(X_mid.shape)
            save_contour(X_mid, Y_mid, u_mid, "$x$ [m]", "$y$ [m]",
                         r"$u$ [m/s]", f"stage3_pred_u_midplane_Re{Re}.png", cmap="viridis", is_v=False)

            # Centerline pressure line plot vs x
            xc_1d = np.linspace(0.0, cfg3.L_DOMAIN, 400)
            yc_1d = np.full_like(xc_1d, 0.5 * cfg3.H)
            zc_1d = np.full_like(xc_1d, 0.5 * cfg3.W)
            _, _, _, p_pred = model.predict_physical(xc_1d, yc_1d, zc_1d, Re, device="cpu")
            
            ref_path = os.path.join(s3_dir, "data", f"duct_Re{Re}.npz")
            fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=8/72, h_pad=10/72)
            if os.path.exists(ref_path):
                ref_d = np.load(ref_path)
                p_ref = ref_d["P"][:, ref_d["P"].shape[1]//2, ref_d["P"].shape[2]//2] if ref_d["P"].ndim == 3 else ref_d["P"]
                ax.plot(ref_d["x"], p_ref, "k-", lw=1.5, label="Parabolic Marching Ref")
            ax.plot(xc_1d, p_pred, "r--", lw=1.5, label="Stage 3 PINN")
            ax.set_xlabel("$x$ [m]")
            ax.set_ylabel(r"$p$ [m$^2$/s$^2$]")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True, fontsize=8)
            out_p1 = f"stage3_pred_centerline_p_Re{Re}.png"
            out_p2 = f"stage3_pred_p_centerline_Re{Re}.png"
            fig.savefig(os.path.join(OUT_DIR, out_p1), dpi=300)
            fig.savefig(os.path.join(OUT_DIR, out_p2), dpi=300)
            plt.close(fig)
            print(f"  Saved {out_p1} and {out_p2}")
    finally:
        os.chdir(orig_cwd)

def plot_stage4():
    print("Plotting Stage 4 predictions...")
    s4_dir = os.path.join(BASE_DIR, "4")
    orig_cwd = os.getcwd()
    try:
        os.chdir(s4_dir)
        sys.path.insert(0, s4_dir)
        sys.modules.pop("model", None)
        sys.modules.pop("config", None)
        import config as cfg4
        import train as T4

        ckpt_path = os.path.join(s4_dir, "models", "best.pt")
        device = torch.device("cpu")
        model, _ = T4.load_model(ckpt_path, device)
        model.eval()

        for Re in [28000, 120000]:
            x_1d = np.linspace(0.05, cfg4.L_OVER_H, NX_2D)
            y_1d = np.linspace(0.0, 1.0, NY_2D)
            X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")
            
            xt = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
            yt = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
            ret = torch.full_like(xt, float(Re))
            
            with torch.no_grad():
                U, V, P, nut = model(xt, yt, ret)
                
            u = U.numpy().reshape(X.shape)
            v = V.numpy().reshape(X.shape)
            p = P.numpy().reshape(X.shape)
            nt = nut.numpy().reshape(X.shape)
            
            save_contour(X, Y, u, "$x/h$", "$y/h$",
                         r"$U / U_{\mathrm{in}}$", f"stage4_pred_U_Re{Re}.png", cmap="viridis", is_v=False)
            save_contour(X, Y, v, "$x/h$", "$y/h$",
                         r"$V / U_{\mathrm{in}}$", f"stage4_pred_V_Re{Re}.png", cmap="coolwarm", is_v=True)
            save_contour(X, Y, p, "$x/h$", "$y/h$",
                         r"$P$", f"stage4_pred_P_Re{Re}.png", cmap="plasma", is_v=False)
            save_contour(X, Y, nt, "$x/h$", "$y/h$",
                         r"$\tilde{\nu} / \nu$", f"stage4_pred_nut_Re{Re}.png", cmap="magma", is_v=True)
    finally:
        os.chdir(orig_cwd)

def plot_stage5():
    print("Plotting Stage 5 predictions...")
    s5_dir = os.path.join(BASE_DIR, "5")
    orig_cwd = os.getcwd()
    try:
        os.chdir(s5_dir)
        sys.path.insert(0, s5_dir)
        sys.modules.pop("pinn", None)
        sys.modules.pop("config", None)
        import config as cfg5
        from pinn import PipePINN as PipePINN5

        dense_path = os.path.join(s5_dir, "outputs", "models", "dense", "final_seed0.pt")
        sparse_path = os.path.join(s5_dir, "outputs", "models", "sparse", "final_seed0.pt")
        ckpt_path = dense_path if os.path.exists(dense_path) else sparse_path
        device = torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        depth = ckpt.get("depth", cfg5.NET_DEPTH)
        width = ckpt.get("width", cfg5.NET_WIDTH)
        model = PipePINN5(depth=depth, width=width).to(device)
        model.load_state_dict(ckpt["state"])
        model.eval()

        for Re in [28000, 120000]:
            x_1d = np.linspace(0.05, cfg5.L, NX_2D)
            r_1d = np.linspace(0.0, 1.0, NY_2D)
            X, R = np.meshgrid(x_1d, r_1d, indexing="ij")
            
            xt = torch.tensor(X.ravel(), dtype=torch.float32)
            rt = torch.tensor(R.ravel(), dtype=torch.float32)
            ret = torch.full_like(xt, float(Re))
            
            with torch.no_grad():
                U, V, P, nut = model(xt, rt, ret)
                
            u = U.numpy().reshape(X.shape)
            v = V.numpy().reshape(X.shape)
            p = P.numpy().reshape(X.shape)
            nt = nut.numpy().reshape(X.shape)
            
            save_contour(X, R, u, "$x/R$", "$r/R$",
                         r"$U / U_{\mathrm{in}}$", f"stage5_pred_U_Re{Re}.png", cmap="viridis", is_v=False)
            save_contour(X, R, v, "$x/R$", "$r/R$",
                         r"$V / U_{\mathrm{in}}$", f"stage5_pred_V_Re{Re}.png", cmap="coolwarm", is_v=True)
            save_contour(X, R, p, "$x/R$", "$r/R$",
                         r"$P$", f"stage5_pred_P_Re{Re}.png", cmap="plasma", is_v=False)
            save_contour(X, R, nt, "$x/R$", "$r/R$",
                         r"$\tilde{\nu} / \nu$", f"stage5_pred_nut_Re{Re}.png", cmap="magma", is_v=True)
    finally:
        os.chdir(orig_cwd)

if __name__ == "__main__":
    plot_stage1()
    plot_stage2()
    plot_stage3()
    plot_stage4()
    plot_stage5()
    print("All prediction plots generated successfully!")
