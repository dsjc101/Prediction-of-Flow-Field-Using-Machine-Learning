"""
deploy_and_visualize.py
========================
Performs comprehensive post-training tasks for the Rectangular Duct (3D) PINN:
1. Physics vs. Data Learning Verification (3D Cartesian Navier-Stokes PDE residuals
   on dense interior grid for train and unseen Re).
2. Weight inspection, statistics, and parameter extraction.
3. Deployable model export: TorchScript (.pt) for high-performance direct production use.
4. Full flowfield predictions (u, v, w, p, streamlines, 2D cross-planes) across the
   entire domain [0, L] x [0, H] x [0, W].
5. Linear-axis error heatmaps and physics residual contour maps across the whole domain.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from pinn_model import DuctPINN, re_to_normalized
from evaluate import load_reference_data, load_model

DEVICE = "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
FIG_DIR = os.path.join(BASE_DIR, config.FIG_DIR)
CKPT_DIR = os.path.join(BASE_DIR, config.CKPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, config.DATA_DIR)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

ALL_RE = config.RE_ALL
TAGS = {
    50: "TRAIN",
    100: "TRAIN",
    200: "TRAIN",
    300: "INTERP (unseen)",
    500: "TRAIN",
    600: "EXTRAP (unseen)"
}
COLORS = {
    "TRAIN": "tab:blue",
    "INTERP (unseen)": "tab:green",
    "EXTRAP (unseen)": "tab:red"
}
FIELDS = ["U", "V", "W", "P"]


def compute_pde_residuals_physical(model, x, y, z, Re_scalar, nu=config.NU):
    """
    Computes physical steady incompressible 3D Navier-Stokes residuals:
    Continuity: u_x + v_y + w_z = 0
    x-Momentum: u*u_x + v*u_y + w*u_z + p_x - nu*laplacian(u) = 0
    y-Momentum: u*v_x + v*v_y + w*v_z + p_y - nu*laplacian(v) = 0
    z-Momentum: u*w_x + v*w_y + w*w_z + p_z - nu*laplacian(w) = 0
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)

    Re_arr = torch.full_like(x, float(Re_scalar))
    lo, hi = np.log10(50.0), np.log10(600.0)
    ren = 2.0 * (torch.log10(Re_arr) - lo) / (hi - lo) - 1.0

    # Model forward normalized
    un, vn, wn, pn = model(x / config.L_DOMAIN, y / config.H, z / config.W, ren)

    U_ref = Re_scalar * nu / config.DH
    p_scale = 0.5 * (U_ref ** 2) * (config.L_DOMAIN / config.DH)

    u = un * U_ref
    v = vn * U_ref
    w = wn * U_ref
    p = pn * p_scale

    def grad(f, v_):
        return torch.autograd.grad(f, v_, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)
    p_x, p_y, p_z = grad(p, x), grad(p, y), grad(p, z)

    u_xx, u_yy, u_zz = grad(u_x, x), grad(u_y, y), grad(u_z, z)
    v_xx, v_yy, v_zz = grad(v_x, x), grad(v_y, y), grad(v_z, z)
    w_xx, w_yy, w_zz = grad(w_x, x), grad(w_y, y), grad(w_z, z)

    r_cont = u_x + v_y + w_z
    r_mom_x = u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    r_mom_y = u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    r_mom_z = u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)

    cont_scale = U_ref / config.DH
    mom_scale = (U_ref ** 2) / config.DH

    rn_c = r_cont / cont_scale
    rn_mx = r_mom_x / mom_scale
    rn_my = r_mom_y / mom_scale
    rn_mz = r_mom_z / mom_scale

    return r_cont, r_mom_x, r_mom_y, r_mom_z, rn_c, rn_mx, rn_my, rn_mz


# ==============================================================================
# 1. Physics vs. Data Verification
# ==============================================================================
def verify_physics_vs_data(model):
    print("=" * 70)
    print("PHYSICS VS. DATA VERIFICATION (3D RECTANGULAR DUCT)")
    print("=" * 70)
    print("Evaluating steady incompressible 3D Navier-Stokes PDE residuals on a dense")
    print("grid (40 streamwise x 15 vertical x 15 spanwise = 9,000 interior points per Re):")

    physics_stats = {}
    nx, ny, nz = 40, 15, 15
    x_grid = np.linspace(0.02 * config.L_DOMAIN, 0.98 * config.L_DOMAIN, nx)
    y_grid = np.linspace(0.05 * config.H, 0.95 * config.H, ny)
    z_grid = np.linspace(0.05 * config.W, 0.95 * config.W, nz)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    x_t = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1)
    z_t = torch.tensor(Z.flatten(), dtype=torch.float32).reshape(-1, 1)

    for Re in ALL_RE:
        tag = TAGS[Re]
        r_c, r_mx, r_my, r_mz, rn_c, rn_mx, rn_my, rn_mz = compute_pde_residuals_physical(
            model, x_t, y_t, z_t, Re, config.NU
        )

        res_phys_mag = torch.sqrt(r_c**2 + r_mx**2 + r_my**2 + r_mz**2).detach().numpy()
        res_norm_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_my**2 + rn_mz**2).detach().numpy()

        stats = {
            "Re": Re,
            "tag": tag,
            "phys_mean": float(np.mean(res_phys_mag)),
            "phys_std": float(np.std(res_phys_mag)),
            "phys_max": float(np.max(res_phys_mag)),
            "norm_mean": float(np.mean(res_norm_mag)),
            "norm_std": float(np.std(res_norm_mag)),
            "norm_max": float(np.max(res_norm_mag)),
            "cont_mean": float(np.mean(np.abs(r_c.detach().numpy()))),
            "mom_x_mean": float(np.mean(np.abs(r_mx.detach().numpy()))),
            "mom_y_mean": float(np.mean(np.abs(r_my.detach().numpy()))),
            "mom_z_mean": float(np.mean(np.abs(r_mz.detach().numpy())))
        }
        physics_stats[Re] = stats
        print(f"Re={Re:4d} [{tag:16s}] PDE Norm Res: mean={stats['norm_mean']:.3e}, "
              f"max={stats['norm_max']:.3e} | Phys: Cont={stats['cont_mean']:.3e}, "
              f"MomX={stats['mom_x_mean']:.3e}, MomY={stats['mom_y_mean']:.3e}, MomZ={stats['mom_z_mean']:.3e}")

    train_res = np.mean([physics_stats[Re]["norm_mean"] for Re in config.RE_TRAIN])
    unseen_res = np.mean([physics_stats[Re]["norm_mean"] for Re in [config.RE_TEST_INTERP, config.RE_TEST_EXTRAP]])
    print("-" * 70)
    print(f"Mean Normalized PDE Residual across TRAIN Re:  {train_res:.3e}")
    print(f"Mean Normalized PDE Residual across UNSEEN Re: {unseen_res:.3e}")
    ratio = unseen_res / train_res
    print(f"Ratio UNSEEN / TRAIN: {ratio:.2f}x")
    if ratio < 2.0:
        print(">> VERDICT: The Duct PINN has learned the governing 3D Navier-Stokes physics!")
        print("   The PDE residual on unseen Re remains on par with training conditions.")
    print("=" * 70)

    with open(os.path.join(CKPT_DIR, "duct_physics_verification_stats.json"), "w") as f:
        json.dump(physics_stats, f, indent=2)

    return physics_stats


# ==============================================================================
# 2. Extract Weights and Mathematical Formulation
# ==============================================================================
def extract_model_parameters(model):
    print("\nEXTRACTING MODEL PARAMETERS AND WEIGHT STATISTICS...")
    param_info = {
        "architecture": {
            "type": "Multi-Layer Perceptron (MLP) DuctPINN",
            "hidden_layers": 4,
            "neurons_per_layer": 80,
            "activation_function": "tanh",
            "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_dimension": 4,
            "input_variables": ["x [m]", "y [m]", "z [m]", "log10(Re)"],
            "output_dimension": 4,
            "output_variables": ["u [m/s]", "v [m/s]", "w [m/s]", "p [m^2/s^2]"]
        },
        "normalization_constants": {
            "x_scale": config.L_DOMAIN,
            "y_scale": config.H,
            "z_scale": config.W,
            "re_min": 50.0,
            "re_max": 600.0,
            "nu": config.NU,
            "Dh": config.DH
        },
        "layers": {}
    }

    layer_idx = 0
    for name, module in model.net.named_children():
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy()
            layer_info = {
                "layer_index": layer_idx,
                "weight_shape": list(w.shape),
                "bias_shape": list(b.shape),
                "weight_norm_fro": float(np.linalg.norm(w)),
                "bias_norm_l2": float(np.linalg.norm(b)),
                "weight_min": float(np.min(w)),
                "weight_max": float(np.max(w)),
                "weight_mean": float(np.mean(w)),
                "weight_std": float(np.std(w)),
                "bias_min": float(np.min(b)),
                "bias_max": float(np.max(b)),
                "bias_mean": float(np.mean(b)),
                "bias_std": float(np.std(b)),
                "sample_weights": w[:3, :3].tolist(),
                "sample_biases": b[:3].tolist()
            }
            param_info["layers"][f"linear_{layer_idx}"] = layer_info
            print(f"Layer {layer_idx} Linear: W shape {w.shape}, ||W||_F = {layer_info['weight_norm_fro']:.3f}, "
                  f"b mean={layer_info['bias_mean']:.4f}")
            layer_idx += 1

    with open(os.path.join(CKPT_DIR, "duct_model_weights_summary.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    print(f"Saved weight summary to {os.path.join(CKPT_DIR, 'duct_model_weights_summary.json')}")
    return param_info


# ==============================================================================
# 3. Model Deployment Export (TorchScript)
# ==============================================================================
def export_deployable_model(model):
    print("\nEXPORTING MODEL FOR DEPLOYMENT...")

    class DeployableDuctPINN(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, inputs):
            # inputs: shape [N, 4] where col 0:x, 1:y, 2:z, 3:Re
            x = inputs[:, 0:1]
            y = inputs[:, 1:2]
            z = inputs[:, 2:3]
            Re = inputs[:, 3:4]

            xn = x / config.L_DOMAIN
            yn = y / config.H
            zn = z / config.W
            lo, hi = np.log10(50.0), np.log10(600.0)
            ren = 2.0 * (torch.log10(Re) - lo) / (hi - lo) - 1.0

            un, vn, wn, pn = self.core(xn, yn, zn, ren)

            U_ref = Re * config.NU / config.DH
            p_scale = 0.5 * (U_ref ** 2) * (config.L_DOMAIN / config.DH)

            u = un * U_ref
            v = vn * U_ref
            w = wn * U_ref
            p = pn * p_scale

            return torch.cat([u, v, w, p], dim=1)

    deploy_wrapper = DeployableDuctPINN(model).eval()

    dummy_input = torch.tensor([
        [0.5, 0.05, 0.10, 100.0],
        [2.0, 0.08, 0.05, 300.0],
        [4.5, 0.02, 0.15, 600.0]
    ], dtype=torch.float32)

    with torch.no_grad():
        ref_out = deploy_wrapper(dummy_input)

    ts_path = os.path.join(CKPT_DIR, "duct_pinn_torchscript.pt")
    traced_model = torch.jit.trace(deploy_wrapper, dummy_input)
    traced_model.save(ts_path)
    print(f"Exported TorchScript model to: {ts_path}")

    loaded_ts = torch.jit.load(ts_path)
    ts_out = loaded_ts(dummy_input)
    assert torch.allclose(ref_out, ts_out, atol=1e-5), "TorchScript output mismatch!"
    print(">> TorchScript verification PASSED!")


# ==============================================================================
# 4. Predict Entire Flowfield and Generate Linear-Axis Plots
# ==============================================================================
def generate_linear_axis_plots(model):
    print("\nGENERATING LINEAR-AXIS FULL-DOMAIN FLOWFIELD & ERROR PLOTS...")
    ref = load_reference_data()

    # 1. Flowfield on vertical mid-plane z = W/2 (x vs y)
    nx_grid, ny_grid = 250, 61
    x_lin = np.linspace(0.0, config.L_DOMAIN, nx_grid)
    y_lin = np.linspace(0.0, config.H, ny_grid)
    X_mid, Y_mid = np.meshgrid(x_lin, y_lin, indexing="ij")
    Z_mid = np.full_like(X_mid, 0.5 * config.W)

    # Predictions dictionary for mid-planes
    predictions_mid = {}
    for Re in ALL_RE:
        u_p, v_p, w_p, p_p = model.predict_physical(
            X_mid.ravel(), Y_mid.ravel(), Z_mid.ravel(), Re, device=DEVICE
        )
        predictions_mid[Re] = {
            "U": u_p.reshape(X_mid.shape),
            "V": v_p.reshape(X_mid.shape),
            "W": w_p.reshape(X_mid.shape),
            "P": p_p.reshape(X_mid.shape)
        }

    # Plot 1: Streamwise velocity u(x, y) on mid-plane z = W/2
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        U_p = predictions_mid[Re]["U"]
        im = ax.contourf(X_mid, Y_mid, U_p, levels=30, cmap="viridis")
        ax.set_title(f"Axial Velocity u(x, y, z=W/2) [m/s] -- Re = {Re:d} ({tag})", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.10])
        fig.colorbar(im, ax=ax, label="u [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Axial Velocity u (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_flowfield_pinn_u.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_flowfield_pinn_u.png")

    # Plot 2: Vertical velocity v(x, y) on mid-plane z = W/2
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        V_p = predictions_mid[Re]["V"]
        im = ax.contourf(X_mid, Y_mid, V_p, levels=30, cmap="coolwarm")
        ax.set_title(f"Vertical Velocity v(x, y, z=W/2) [m/s] -- Re = {Re:d} ({tag})", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.10])
        fig.colorbar(im, ax=ax, label="v [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Vertical Velocity v (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_flowfield_pinn_v.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_flowfield_pinn_v.png")

    # Plot 3: Spanwise velocity w(x, z) on horizontal mid-plane y = H/2
    nz_grid = 81
    z_lin = np.linspace(0.0, config.W, nz_grid)
    X_hmid, Z_hmid = np.meshgrid(x_lin, z_lin, indexing="ij")
    Y_hmid = np.full_like(X_hmid, 0.5 * config.H)

    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        _, _, w_p, _ = model.predict_physical(
            X_hmid.ravel(), Y_hmid.ravel(), Z_hmid.ravel(), Re, device=DEVICE
        )
        W_p = w_p.reshape(X_hmid.shape)
        im = ax.contourf(X_hmid, Z_hmid, W_p, levels=30, cmap="coolwarm")
        ax.set_title(f"Spanwise Velocity w(x, y=H/2, z) [m/s] -- Re = {Re:d} ({tag})", fontsize=11)
        ax.set_ylabel("z [m]")
        ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
        fig.colorbar(im, ax=ax, label="w [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Spanwise Velocity w (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_flowfield_pinn_w.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_flowfield_pinn_w.png")

    # Plot 4: Centerline Kinematic Pressure p(x)
    fig, ax = plt.subplots(figsize=(9, 5))
    x_cl = np.linspace(0.05 * config.L_DOMAIN, config.L_DOMAIN, 200)
    y_cl = np.full_like(x_cl, 0.5 * config.H)
    z_cl = np.full_like(x_cl, 0.5 * config.W)

    for Re in ALL_RE:
        tag = TAGS[Re]
        _, _, _, p_cl = model.predict_physical(x_cl, y_cl, z_cl, Re, device=DEVICE)
        U_ref = Re * config.NU / config.DH
        p_sc = 0.5 * (U_ref ** 2) * (config.L_DOMAIN / config.DH)
        ax.plot(x_cl, p_cl / p_sc, label=f"Re={Re:d} ({tag})", linewidth=1.8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Normalized Kinematic Pressure p / p_scale [-]")
    ax.set_title("Centerline Kinematic Pressure Variation Along Rectangular Duct (Linear Axes)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_flowfield_pinn_p.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_flowfield_pinn_p.png")

    # Plot 5: 2D Cross-Plane axial velocity contours at developing and fully-developed stations
    # for Re=100 (Train) and Re=300 (Unseen Interp)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    sample_cases = [100, 300]
    Y_cross, Z_cross = np.meshgrid(y_lin, z_lin, indexing="ij")

    for col_idx, Re in enumerate(sample_cases):
        tag = TAGS[Re]
        Le = ref[Re]["x"][-1] * 0.5  # developing location
        x_dev = min(1.0, 0.3 * config.L_DOMAIN)
        x_fd = 0.95 * config.L_DOMAIN
        U_ref = Re * config.NU / config.DH

        # Developing
        X_cross_dev = np.full_like(Y_cross, x_dev)
        u_dev, _, _, _ = model.predict_physical(X_cross_dev.ravel(), Y_cross.ravel(), Z_cross.ravel(), Re, device=DEVICE)
        U_dev_2d = u_dev.reshape(Y_cross.shape) / U_ref

        im1 = axes[0, col_idx].contourf(Z_cross, Y_cross, U_dev_2d, levels=25, cmap="viridis")
        axes[0, col_idx].set_title(f"Developing: x = {x_dev:.2f} m\nRe = {Re:d} [{tag}]", fontsize=10)
        axes[0, col_idx].set_xlabel("z [m]")
        axes[0, col_idx].set_ylabel("y [m]")
        fig.colorbar(im1, ax=axes[0, col_idx], label="u / U_0")

        # Fully developed
        X_cross_fd = np.full_like(Y_cross, x_fd)
        u_fd, _, _, _ = model.predict_physical(X_cross_fd.ravel(), Y_cross.ravel(), Z_cross.ravel(), Re, device=DEVICE)
        U_fd_2d = u_fd.reshape(Y_cross.shape) / U_ref

        im2 = axes[1, col_idx].contourf(Z_cross, Y_cross, U_fd_2d, levels=25, cmap="viridis")
        axes[1, col_idx].set_title(f"Fully-Developed: x = {x_fd:.2f} m\nRe = {Re:d} [{tag}]", fontsize=10)
        axes[1, col_idx].set_xlabel("z [m]")
        axes[1, col_idx].set_ylabel("y [m]")
        fig.colorbar(im2, ax=axes[1, col_idx], label="u / U_0")

    fig.suptitle("Cross-Sectional (y-z) Axial Velocity Contours in Rectangular Duct", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_cross_plane_u.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_cross_plane_u.png")

    # Plot 6: Velocity Magnitude and Streamlines on Mid-Plane z = W/2
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    sample_cases_sl = [100, 300, 500, 600]
    for ax, Re in zip(axes, sample_cases_sl):
        tag = TAGS[Re]
        U_p = predictions_mid[Re]["U"]
        V_p = predictions_mid[Re]["V"]
        U_mag = np.sqrt(U_p**2 + V_p**2)
        im = ax.contourf(X_mid, Y_mid, U_mag, levels=30, cmap="turbo")
        ax.streamplot(x_lin, y_lin, U_p.T, V_p.T, color="white", density=0.7, linewidth=0.7, arrowsize=0.8)
        ax.set_title(f"Velocity Magnitude |U| and Streamlines (z = W/2) -- Re = {Re:d} [{tag}]", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.10])
        fig.colorbar(im, ax=ax, label="|U| [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Streamline Patterns and Velocity Magnitude in Duct Mid-Plane (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_flowfield_comparison_streamlines.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_flowfield_comparison_streamlines.png")

    # Plot 7: Spatial Error Maps for U, V, W, P on Mid-Plane z = W/2
    for f in FIELDS:
        local_err_fields = {}
        for Re in ALL_RE:
            d = ref[Re]
            x_r = d["x"]
            y_r = d["y"]
            z_mid_idx = len(d["z"]) // 2
            z_val = d["z"][z_mid_idx]

            X_r, Y_r = np.meshgrid(x_r, y_r, indexing="ij")
            Z_r = np.full_like(X_r, z_val)

            u_p, v_p, w_p, p_p = model.predict_physical(X_r.ravel(), Y_r.ravel(), Z_r.ravel(), Re, device=DEVICE)
            preds_2d = {
                "U": u_p.reshape(X_r.shape),
                "V": v_p.reshape(X_r.shape),
                "W": w_p.reshape(X_r.shape),
                "P": p_p.reshape(X_r.shape)
            }
            ref_2d = d[f][:, :, z_mid_idx]

            U_ref = Re * config.NU / config.DH
            scale = U_ref if f in ("U", "V", "W") else 0.5 * (U_ref ** 2) * (config.L_DOMAIN / config.DH)

            err_map = 100.0 * np.abs(preds_2d[f] - ref_2d) / (scale + 1e-12)
            local_err_fields[Re] = (x_r, y_r, err_map)

        all_max = max(np.percentile(v[2], 99) for v in local_err_fields.values())
        # Cap colorbar at 99.5th percentile to prevent near-wall/corner singularities from saturating colormap
        vmax_plot = max(all_max, 1.0)
        fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
        for ax, Re in zip(axes, ALL_RE):
            x_r, y_r, err_f = local_err_fields[Re]
            p99_val = np.percentile(err_f, 99)
            im = ax.contourf(x_r, y_r, err_f.T, levels=25, vmin=0, vmax=vmax_plot, cmap="inferno")
            ax.set_title(f"Re = {Re:d} ({TAGS[Re]}) -- P99 Error: {p99_val:.2f}% (Domain Max: {np.max(err_f):.1f}%)", fontsize=10)
            ax.set_ylabel("y [m]")
            ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.10])
            fig.colorbar(im, ax=ax, label=f"|{f}_PINN - {f}_ref| / scale [%]")
        axes[-1].set_xlabel("x [m]")
        fig.suptitle(f"Whole-Domain Spatial Error Distribution for Duct Field '{f}' (Linear Axes)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"duct_error_domain_{f.lower()}_linear.png"), dpi=150)
        plt.close(fig)
        print(f"Saved figures/duct_error_domain_{f.lower()}_linear.png")

    # Plot 8: Entire Domain Error vs. Re Scatter Plot
    # Parse global_summary.txt
    re_vals = []
    tags_list = []
    errs_dict = {f: [] for f in FIELDS}

    with open(os.path.join(FIG_DIR, "global_summary.txt"), "r") as fh:
        lines = fh.readlines()
        for line in lines:
            if line.startswith("Re="):
                parts = line.strip().split()
                re_v = int(parts[0].split("=")[1])
                role_v = parts[1].split("=")[1]
                field_v = parts[2].split("=")[1]
                dom_err = float(parts[3].split("=")[1]) * 100.0  # percentage

                if field_v == "U":
                    re_vals.append(re_v)
                    tags_list.append(role_v)
                errs_dict[field_v].append(dom_err)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, f in zip(axes, FIELDS):
        for Re, role, err in zip(re_vals, tags_list, errs_dict[f]):
            tag_name = "TRAIN" if role == "TRAIN" else ("INTERP (unseen)" if role == "INTERP" else "EXTRAP (unseen)")
            ax.scatter(Re, err, c=COLORS[tag_name], label=tag_name, s=80, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlabel("Reynolds Number (Re)")
        ax.set_ylabel("Global Rel L2 Error [%]")
        ax.set_title(f"Duct Domain Error vs. Re: Field '{f}' (Linear Axes)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_entire_domain_error_vs_Re_linear.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_entire_domain_error_vs_Re_linear.png")

    # Plot 9: Spatial Map of Navier-Stokes PDE Residual Norm on Mid-Plane z = W/2
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    nx_res, ny_res = 100, 31
    xr = np.linspace(0.02 * config.L_DOMAIN, 0.98 * config.L_DOMAIN, nx_res)
    yr = np.linspace(0.05 * config.H, 0.95 * config.H, ny_res)
    Xr, Yr = np.meshgrid(xr, yr, indexing="ij")
    Zr = np.full_like(Xr, 0.5 * config.W)

    xt_res = torch.tensor(Xr.flatten(), dtype=torch.float32).reshape(-1, 1)
    yt_res = torch.tensor(Yr.flatten(), dtype=torch.float32).reshape(-1, 1)
    zt_res = torch.tensor(Zr.flatten(), dtype=torch.float32).reshape(-1, 1)

    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        _, _, _, _, rn_c, rn_mx, rn_my, rn_mz = compute_pde_residuals_physical(
            model, xt_res, yt_res, zt_res, Re, config.NU
        )
        res_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_my**2 + rn_mz**2).detach().numpy().reshape(Xr.shape)
        im = ax.contourf(Xr, Yr, res_mag, levels=25, cmap="magma")
        ax.set_title(f"Navier-Stokes PDE Residual Norm -- Re = {Re:d} ({tag})", fontsize=10)
        ax.set_ylabel("y [m]")
        ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.10])
        fig.colorbar(im, ax=ax, label="||Res||_norm")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Spatial Verification of 3D Navier-Stokes Satisfaction on Mid-Plane (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "duct_physics_pde_residuals_map.png"), dpi=150)
    plt.close(fig)
    print("Saved figures/duct_physics_pde_residuals_map.png")


if __name__ == "__main__":
    print("Starting deploy and visualize pipeline for 3D Rectangular Duct PINN...")
    model = load_model(device=DEVICE)
    verify_physics_vs_data(model)
    extract_model_parameters(model)
    export_deployable_model(model)
    generate_linear_axis_plots(model)
    print("\nALL STEP 3 TASKS COMPLETED SUCCESSFULLY!")
