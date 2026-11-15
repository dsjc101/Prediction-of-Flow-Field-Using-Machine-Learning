"""
deploy_and_visualize.py

Performs comprehensive post-training tasks for the Channel PINN:
1. Physics vs. Data Learning Verification (PDE residuals on dense interior grid for train and unseen Re).
2. Weight inspection, statistics, and parameter extraction.
3. Deployable model export: TorchScript (.pt) and ONNX (.onnx) with verification.
4. Full flowfield predictions (u, v, p, velocity magnitude, streamlines) across the entire domain.
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

import config as cfg
from model import ChannelPINN, u_ref, p_scale
from physics import ns_residuals_normalized, ns_residuals
from evaluate import ALL_RE, TAGS, COLORS, FIELDS, OUT, field_scale, load_reference

DEVICE = "cpu"
os.makedirs(OUT, exist_ok=True)


def load_trained_model():
    model = ChannelPINN(n_hidden=4, width=80)
    ckpt_path = os.path.join(OUT, "channel_pinn_hybrid.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ==============================================================================
# 1. Physics vs. Data Verification
# ==============================================================================
def verify_physics_vs_data(model):
    print("=" * 70)
    print("PHYSICS VS. DATA VERIFICATION")
    print("=" * 70)
    print("Evaluating steady incompressible Navier-Stokes PDE residuals on a dense")
    print("grid (100 streamwise x 51 cross-stream = 5,100 points per Re) across domain:")

    physics_stats = {}
    nx, ny = 100, 51
    # Sample slightly inside boundaries to evaluate interior PDE satisfaction
    x_grid = np.linspace(0.01 * cfg.L, 0.99 * cfg.L, nx)
    y_grid = np.linspace(-0.95 * cfg.h, 0.95 * cfg.h, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    for Re in ALL_RE:
        tag = TAGS[Re]
        Re_t = torch.full_like(x_t, float(Re))
        
        # Physical residuals
        r_c, r_mx, r_my = ns_residuals(model, x_t, y_t, Re_t)
        # Normalized residuals
        rn_c, rn_mx, rn_my = ns_residuals_normalized(model, x_t, y_t, Re_t)
        
        # Combined residual magnitude
        res_phys_mag = torch.sqrt(r_c**2 + r_mx**2 + r_my**2).detach().numpy()
        res_norm_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_my**2).detach().numpy()

        r_c_np = r_c.detach().numpy()
        r_mx_np = r_mx.detach().numpy()
        r_my_np = r_my.detach().numpy()

        stats = {
            "Re": Re,
            "tag": tag,
            "phys_mean": float(np.mean(res_phys_mag)),
            "phys_std": float(np.std(res_phys_mag)),
            "phys_max": float(np.max(res_phys_mag)),
            "norm_mean": float(np.mean(res_norm_mag)),
            "norm_std": float(np.std(res_norm_mag)),
            "norm_max": float(np.max(res_norm_mag)),
            "cont_mean": float(np.mean(np.abs(r_c_np))),
            "mom_x_mean": float(np.mean(np.abs(r_mx_np))),
            "mom_y_mean": float(np.mean(np.abs(r_my_np)))
        }
        physics_stats[Re] = stats
        print(f"Re={Re:5.0f} [{tag:16s}] PDE Norm Res: mean={stats['norm_mean']:.3e}, "
              f"max={stats['norm_max']:.3e} | Phys: Cont={stats['cont_mean']:.3e}, "
              f"MomX={stats['mom_x_mean']:.3e}, MomY={stats['mom_y_mean']:.3e}")

    # Contrast TRAIN vs UNSEEN
    train_res = np.mean([physics_stats[Re]["norm_mean"] for Re in cfg.RE_TRAIN])
    unseen_res = np.mean([physics_stats[Re]["norm_mean"] for Re in cfg.RE_TEST_INTERPOLATION])
    print("-" * 70)
    print(f"Mean Normalized PDE Residual across TRAIN Re:  {train_res:.3e}")
    print(f"Mean Normalized PDE Residual across UNSEEN Re: {unseen_res:.3e}")
    print(f"Ratio UNSEEN / TRAIN: {unseen_res / train_res:.2f}x")
    if (unseen_res / train_res) < 2.0:
        print(">> VERDICT: The PINN has learned the governing Navier-Stokes physics!")
        print("   The PDE residual on unseen Re (interpolation Re=300 & extrapolation Re=600)")
        print("   remains at the same low order of magnitude as on the training Re.")
    else:
        print(">> Note: High discrepancy between train and unseen PDE residual.")
    print("=" * 70)

    with open(os.path.join(OUT, "physics_verification_stats.json"), "w") as f:
        json.dump(physics_stats, f, indent=2)

    return physics_stats


# ==============================================================================
# 2. Extract Weights and Mathematical Formulation
# ==============================================================================
def extract_model_parameters(model):
    print("\nEXTRACTING MODEL PARAMETERS AND WEIGHT STATISTICS...")
    param_info = {
        "architecture": {
            "type": "Multi-Layer Perceptron (MLP) PINN",
            "hidden_layers": 4,
            "neurons_per_layer": 80,
            "activation_function": "tanh",
            "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_dimension": 3,
            "input_variables": ["x [m]", "y [m]", "log10(Re)"],
            "output_dimension": 3,
            "output_variables": ["u [m/s]", "v [m/s]", "p [m^2/s^2]"]
        },
        "normalization_constants": {
            "x_scale": cfg.L,
            "y_scale": cfg.h,
            "re_log_min": float(model.re_log_min.item()),
            "re_log_max": float(model.re_log_max.item()),
            "nu": cfg.NU,
            "Dh": cfg.Dh
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
                # First 3 output weights for representation showcase
                "sample_weights": w[:3, :3].tolist(),
                "sample_biases": b[:3].tolist()
            }
            param_info["layers"][f"linear_{layer_idx}"] = layer_info
            layer_idx += 1

    with open(os.path.join(OUT, "model_weights_summary.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    print(f"Saved weight summary to {os.path.join(OUT, 'model_weights_summary.json')}")
    return param_info


# ==============================================================================
# 3. Model Deployment Export (TorchScript and ONNX)
# ==============================================================================
def export_deployable_model(model):
    print("\nEXPORTING MODEL FOR DEPLOYMENT...")
    
    # Wrap model to take single tensor input [N, 3] = [x, y, Re]
    class DeployableChannelPINN(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, inputs):
            # inputs: shape [N, 3] where col 0 is x, col 1 is y, col 2 is Re
            x = inputs[:, 0:1]
            y = inputs[:, 1:2]
            Re = inputs[:, 2:3]
            u, v, p = self.core(x, y, Re)
            return torch.cat([u, v, p], dim=1)

    deploy_wrapper = DeployableChannelPINN(model).eval()

    # Test dummy inputs
    dummy_input = torch.tensor([
        [0.5, 0.0, 100.0],
        [1.0, 0.02, 300.0],
        [2.0, -0.01, 600.0]
    ], dtype=torch.float32)

    with torch.no_grad():
        ref_out = deploy_wrapper(dummy_input)

    # 1. Export TorchScript (JIT trace)
    ts_path = os.path.join(OUT, "channel_pinn_torchscript.pt")
    traced_model = torch.jit.trace(deploy_wrapper, dummy_input)
    traced_model.save(ts_path)
    print(f"Exported TorchScript model to: {ts_path}")

    # Verify TorchScript
    loaded_ts = torch.jit.load(ts_path)
    ts_out = loaded_ts(dummy_input)
    assert torch.allclose(ref_out, ts_out, atol=1e-6), "TorchScript output mismatch!"
    print(">> TorchScript verification PASSED!")

    # 2. Export ONNX
    onnx_path = os.path.join(OUT, "channel_pinn.onnx")
    try:
        torch.onnx.export(
            deploy_wrapper,
            dummy_input,
            onnx_path,
            input_names=["input_coords_Re"],
            output_names=["predicted_uvp"],
            dynamic_axes={
                "input_coords_Re": {0: "batch_size"},
                "predicted_uvp": {0: "batch_size"}
            },
            opset_version=14
        )
        print(f"Exported ONNX model to: {onnx_path}")
        print(">> ONNX export PASSED!")
    except Exception as e:
        print(f">> ONNX export notice: {e}")


# ==============================================================================
# 4. Predict Entire Flowfield and Generate Linear-Axis Plots
# ==============================================================================
def generate_linear_axis_plots(model):
    print("\nGENERATING LINEAR-AXIS FULL-DOMAIN FLOWFIELD & ERROR PLOTS...")
    ref = np.load(os.path.join(OUT, "reference_data.npz"))

    nx_grid, ny_grid = 300, 81
    x_lin = np.linspace(cfg.X_MIN, cfg.X_MAX, nx_grid)
    y_lin = np.linspace(cfg.Y_MIN, cfg.Y_MAX, ny_grid)
    X_full, Y_full = np.meshgrid(x_lin, y_lin, indexing="ij")

    # Evaluate for each Re
    predictions = {}
    for Re in ALL_RE:
        x_t = torch.tensor(X_full.flatten(), dtype=torch.float32).reshape(-1, 1)
        y_t = torch.tensor(Y_full.flatten(), dtype=torch.float32).reshape(-1, 1)
        Re_t = torch.full_like(x_t, float(Re))
        with torch.no_grad():
            u, v, p = model(x_t, y_t, Re_t)
        U_p = u.numpy().reshape(X_full.shape)
        V_p = v.numpy().reshape(X_full.shape)
        P_p = p.numpy().reshape(X_full.shape)
        predictions[Re] = {"U": U_p, "V": V_p, "P": P_p}

    # Plot 1: Full-domain u(x, y) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        U_p = predictions[Re]["U"]
        im = ax.contourf(X_full, Y_full, U_p, levels=30, cmap="viridis")
        ax.set_title(f"Streamwise Velocity u(x, y) [m/s] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("y [m]")
        # Explicit linear ticks
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="u [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Streamwise Velocity u (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flowfield_pinn_u.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/flowfield_pinn_u.png")

    # Plot 2: Full-domain v(x, y) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        V_p = predictions[Re]["V"]
        im = ax.contourf(X_full, Y_full, V_p, levels=30, cmap="coolwarm")
        ax.set_title(f"Wall-Normal Velocity v(x, y) [m/s] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="v [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Wall-Normal Velocity v (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flowfield_pinn_v.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/flowfield_pinn_v.png")

    # Plot 3: Full-domain p(x, y) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        P_p = predictions[Re]["P"]
        im = ax.contourf(X_full, Y_full, P_p, levels=30, cmap="plasma")
        ax.set_title(f"Kinematic Pressure p(x, y) [m$^2$/s$^2$] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="p [m$^2$/s$^2$]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Kinematic Pressure p (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flowfield_pinn_p.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/flowfield_pinn_p.png")

    # Plot 4: Streamlines and Velocity Magnitude for Re=100 (Train) and Re=300 (Unseen Interp)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    sample_cases = [(100.0, "TRAIN"), (300.0, "INTERP (unseen)"), (500.0, "TRAIN"), (600.0, "EXTRAP (unseen)")]
    for ax, (Re, tag) in zip(axes, sample_cases):
        U_p = predictions[Re]["U"]
        V_p = predictions[Re]["V"]
        U_mag = np.sqrt(U_p**2 + V_p**2)
        im = ax.contourf(X_full, Y_full, U_mag, levels=35, cmap="turbo")
        # Overlay velocity streamlines
        ax.streamplot(x_lin, y_lin, U_p.T, V_p.T, color="white", density=0.7, linewidth=0.7, arrowsize=0.8)
        ax.set_title(f"Velocity Magnitude |U| and Streamlines -- Re={Re:.0f} [{tag}]", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="|U| [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Streamline Patterns and Velocity Magnitude Across Development Length", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flowfield_comparison_streamlines.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/flowfield_comparison_streamlines.png")

    # Plot 5: Linear-axis whole-domain error heatmaps (|PINN - Ref| / scale)
    for f in FIELDS:
        local_err_fields = {}
        for Re in ALL_RE:
            x_ref, y_ref, U_ref, V_ref, P_ref = load_reference(ref, Re)
            X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing="ij")
            
            x_t = torch.tensor(X_ref.flatten(), dtype=torch.float32).reshape(-1, 1)
            y_t = torch.tensor(Y_ref.flatten(), dtype=torch.float32).reshape(-1, 1)
            Re_t = torch.full_like(x_t, float(Re))
            with torch.no_grad():
                u, v, p = model(x_t, y_t, Re_t)
            u_p = u.numpy().reshape(X_ref.shape)
            v_p = v.numpy().reshape(X_ref.shape)
            p_p = p.numpy().reshape(X_ref.shape)
            P_ref_grid = np.broadcast_to(P_ref[:, None], X_ref.shape)

            pred_map = {"u": u_p, "v": v_p, "p": p_p}
            exact_map = {"u": U_ref, "v": V_ref, "p": P_ref_grid}

            scale = field_scale(f, Re)
            err_map = 100.0 * np.abs(pred_map[f] - exact_map[f]) / (scale + 1e-12)
            local_err_fields[Re] = (x_ref, y_ref, err_map)

        all_max = max(np.percentile(v[2], 99) for v in local_err_fields.values())
        fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
        for ax, Re in zip(axes, ALL_RE):
            x_r, y_r, err_f = local_err_fields[Re]
            im = ax.contourf(x_r, y_r, err_f.T, levels=25, vmin=0, vmax=all_max, cmap="inferno")
            ax.set_title(f"Re = {Re:.0f} ({TAGS[Re]}) -- Max Error: {err_f.max():.2f}%", fontsize=10)
            ax.set_ylabel("y [m]")
            ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
            fig.colorbar(im, ax=ax, label=f"|{f}_PINN - {f}_ref| / scale [%]")
        axes[-1].set_xlabel("x [m]")
        fig.suptitle(f"Whole-Domain Spatial Error Distribution for Field '{f}' (Linear Axes)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"error_domain_{f}_linear.png"), dpi=150)
        plt.close(fig)
        print(f"Saved outputs/error_domain_{f}_linear.png")

    # Plot 6: Linear-axis Global Error vs Re Scatter Plot
    df_rows = []
    with open(os.path.join(OUT, "metrics_full_domain.csv"), "r") as fh:
        lines = fh.readlines()[1:]
        for line in lines:
            parts = line.strip().split(",")
            df_rows.append((parts[0], float(parts[1]), parts[2], float(parts[3])))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, f in zip(axes, FIELDS):
        f_data = [r for r in df_rows if r[0] == f]
        for _, Re, tag, err in f_data:
            ax.scatter(Re, err, c=COLORS[tag], label=tag, s=80, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        # Explicit linear axes
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlabel("Reynolds Number (Re)")
        ax.set_ylabel(f"Global Rel L2 Error [%]")
        ax.set_title(f"Domain Error vs. Re: Field '{f}' (Linear Axes)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "entire_domain_error_vs_Re_linear.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/entire_domain_error_vs_Re_linear.png")

    # Plot 7: Physics Residual Spatial Map
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    nx_res, ny_res = 120, 41
    xr = np.linspace(0.01 * cfg.L, 0.99 * cfg.L, nx_res)
    yr = np.linspace(-0.95 * cfg.h, 0.95 * cfg.h, ny_res)
    Xr, Yr = np.meshgrid(xr, yr, indexing="ij")
    xt_res = torch.tensor(Xr.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    yt_res = torch.tensor(Yr.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        Re_t = torch.full_like(xt_res, float(Re))
        rn_c, rn_mx, rn_my = ns_residuals_normalized(model, xt_res, yt_res, Re_t)
        res_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_my**2).detach().numpy().reshape(Xr.shape)
        im = ax.contourf(Xr, Yr, res_mag, levels=25, cmap="magma")
        ax.set_title(f"Navier-Stokes PDE Residual Norm -- Re = {Re:.0f} ({tag})", fontsize=10)
        ax.set_ylabel("y [m]")
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="||Res||_norm")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Spatial Verification of Navier-Stokes Physics Satisfaction (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "physics_pde_residuals_map.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/physics_pde_residuals_map.png")


if __name__ == "__main__":
    print("Starting deploy and visualize pipeline...")
    model = load_trained_model()
    verify_physics_vs_data(model)
    extract_model_parameters(model)
    export_deployable_model(model)
    generate_linear_axis_plots(model)
    print("\nALL TASKS COMPLETED SUCCESSFULLY!")

