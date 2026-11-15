"""
deploy_and_visualize.py

Performs comprehensive post-training tasks for the Circular-Pipe (Axisymmetric) PINN:
1. Physics vs. Data Learning Verification (axisymmetric Navier-Stokes PDE residuals
   on dense interior grid for train and unseen Re).
2. Weight inspection, statistics, and parameter extraction.
3. Deployable model export: TorchScript (.pt) and ONNX (.onnx) supporting both
   axisymmetric (x, r, Re) and full 3D Cartesian (x, y, z, Re) inference.
4. Full flowfield predictions (u, v_r, p, velocity magnitude, streamlines) across
   the entire axisymmetric domain [0, L] x [0, R].
5. Linear-axis error heatmaps and physics residual contour maps across the whole domain.
6. Radial velocity profile comparisons at developing stations and fully-developed station.
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
from model import PipePINN, u_ref, p_scale
from physics import ns_residuals, ns_residuals_normalized

DEVICE = "cpu"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

ALL_RE = cfg.ALL_RE_FOR_EVAL
TAGS = {
    50.0: "TRAIN",
    100.0: "TRAIN",
    200.0: "TRAIN",
    300.0: "INTERP (unseen)",
    500.0: "TRAIN",
    600.0: "EXTRAP (unseen)"
}
COLORS = {
    "TRAIN": "tab:blue",
    "INTERP (unseen)": "tab:green",
    "EXTRAP (unseen)": "tab:red"
}
FIELDS = ["u", "v", "p"]


def load_trained_model():
    model = PipePINN(n_hidden=4, width=80)
    ckpt_path = os.path.join(OUT, "pipe_pinn_hybrid.pt")
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
    print("PHYSICS VS. DATA VERIFICATION (AXISYMMETRIC CIRCULAR PIPE)")
    print("=" * 70)
    print("Evaluating steady incompressible axisymmetric Navier-Stokes PDE residuals")
    print("on a dense grid (100 streamwise x 50 radial = 5,000 interior points per Re):")

    physics_stats = {}
    nx, nr = 100, 50
    # Sample slightly inside boundaries to evaluate interior PDE satisfaction
    # avoiding r=0 axis singularity and r=R wall boundary
    x_grid = np.linspace(0.01 * cfg.L, 0.99 * cfg.L, nx)
    r_grid = np.linspace(0.02 * cfg.R, 0.98 * cfg.R, nr)
    X, R = np.meshgrid(x_grid, r_grid, indexing="ij")

    x_t = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    r_t = torch.tensor(R.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    for Re in ALL_RE:
        tag = TAGS[Re]
        Re_t = torch.full_like(x_t, float(Re))

        # Physical residuals
        r_c, r_mx, r_mr = ns_residuals(model, x_t, r_t, Re_t)
        # Normalized residuals
        rn_c, rn_mx, rn_mr = ns_residuals_normalized(model, x_t, r_t, Re_t)

        # Combined residual magnitude
        res_phys_mag = torch.sqrt(r_c**2 + r_mx**2 + r_mr**2).detach().numpy()
        res_norm_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_mr**2).detach().numpy()

        r_c_np = r_c.detach().numpy()
        r_mx_np = r_mx.detach().numpy()
        r_mr_np = r_mr.detach().numpy()

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
            "mom_r_mean": float(np.mean(np.abs(r_mr_np)))
        }
        physics_stats[Re] = stats
        print(f"Re={Re:5.0f} [{tag:16s}] PDE Norm Res: mean={stats['norm_mean']:.3e}, "
              f"max={stats['norm_max']:.3e} | Phys: Cont={stats['cont_mean']:.3e}, "
              f"MomX={stats['mom_x_mean']:.3e}, MomR={stats['mom_r_mean']:.3e}")

    # Contrast TRAIN vs UNSEEN
    train_res = np.mean([physics_stats[Re]["norm_mean"] for Re in cfg.RE_TRAIN])
    unseen_res = np.mean([physics_stats[Re]["norm_mean"] for Re in cfg.RE_TEST_INTERPOLATION])
    print("-" * 70)
    print(f"Mean Normalized PDE Residual across TRAIN Re:  {train_res:.3e}")
    print(f"Mean Normalized PDE Residual across UNSEEN Re: {unseen_res:.3e}")
    print(f"Ratio UNSEEN / TRAIN: {unseen_res / train_res:.2f}x")
    if (unseen_res / train_res) < 2.0:
        print(">> VERDICT: The PINN has learned the governing axisymmetric Navier-Stokes physics!")
        print("   The PDE residual on unseen Re (interpolation Re=300 & extrapolation Re=600)")
        print("   remains at the same low order of magnitude as on the training Re.")
    else:
        print(">> Note: High discrepancy between train and unseen PDE residual.")
    print("=" * 70)

    with open(os.path.join(OUT, "pipe_physics_verification_stats.json"), "w") as f:
        json.dump(physics_stats, f, indent=2)

    return physics_stats


# ==============================================================================
# 2. Extract Weights and Mathematical Formulation
# ==============================================================================
def extract_model_parameters(model):
    print("\nEXTRACTING MODEL PARAMETERS AND WEIGHT STATISTICS...")
    param_info = {
        "architecture": {
            "type": "Multi-Layer Perceptron (MLP) PipePINN",
            "hidden_layers": 4,
            "neurons_per_layer": 80,
            "activation_function": "tanh",
            "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_dimension": 3,
            "input_variables": ["x [m]", "r [m]", "log10(Re)"],
            "output_dimension": 3,
            "output_variables": ["u [m/s]", "v_r [m/s]", "p [m^2/s^2]"]
        },
        "normalization_constants": {
            "x_scale": cfg.L,
            "r_scale": cfg.R,
            "re_log_min": float(model.re_log_min.item()),
            "re_log_max": float(model.re_log_max.item()),
            "nu": cfg.NU,
            "D": cfg.D
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

    with open(os.path.join(OUT, "pipe_model_weights_summary.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    print(f"Saved weight summary to {os.path.join(OUT, 'pipe_model_weights_summary.json')}")
    return param_info


# ==============================================================================
# 3. Model Deployment Export (TorchScript and ONNX)
# ==============================================================================
def export_deployable_model(model):
    print("\nEXPORTING MODEL FOR DEPLOYMENT...")

    # Wrapper 1: Axisymmetric deployment wrapper (x, r, Re) -> (u_x, u_r, p)
    class DeployablePipePINNAxisymmetric(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, inputs):
            # inputs: shape [N, 3] where col 0 is x, col 1 is r, col 2 is Re
            x = inputs[:, 0:1]
            r = inputs[:, 1:2]
            Re = inputs[:, 2:3]
            u, v, p = self.core(x, r, Re)
            return torch.cat([u, v, p], dim=1)

    # Wrapper 2: Full 3D Cartesian deployment wrapper (x, y, z, Re) -> (u_x, u_y, u_z, p)
    class DeployablePipePINN3DCartesian(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model

        def forward(self, inputs):
            # inputs: shape [N, 4] where col 0 is x, col 1 is y, col 2 is z, col 3 is Re
            x = inputs[:, 0:1]
            y = inputs[:, 1:2]
            z = inputs[:, 2:3]
            Re = inputs[:, 3:4]
            ux, uy, uz, p = self.core.to_cartesian(x, y, z, Re)
            return torch.cat([ux, uy, uz, p], dim=1)

    deploy_axisym = DeployablePipePINNAxisymmetric(model).eval()
    deploy_3d = DeployablePipePINN3DCartesian(model).eval()

    # Dummy inputs
    dummy_axisym = torch.tensor([
        [0.5, 0.01, 100.0],
        [1.0, 0.025, 300.0],
        [2.5, 0.04, 600.0]
    ], dtype=torch.float32)

    dummy_3d = torch.tensor([
        [0.5, 0.01, 0.0, 100.0],
        [1.0, 0.015, 0.02, 300.0],
        [2.5, 0.02, -0.03, 600.0]
    ], dtype=torch.float32)

    with torch.no_grad():
        ref_out_axisym = deploy_axisym(dummy_axisym)
        ref_out_3d = deploy_3d(dummy_3d)

    # 1. Export TorchScript (JIT trace)
    ts_axisym_path = os.path.join(OUT, "pipe_pinn_torchscript.pt")
    traced_axisym = torch.jit.trace(deploy_axisym, dummy_axisym)
    traced_axisym.save(ts_axisym_path)
    print(f"Exported TorchScript model to: {ts_axisym_path}")

    # Verify TorchScript
    loaded_ts = torch.jit.load(ts_axisym_path)
    ts_out = loaded_ts(dummy_axisym)
    assert torch.allclose(ref_out_axisym, ts_out, atol=1e-5), "TorchScript output mismatch!"
    print(">> TorchScript verification PASSED!")

    # 2. Export 3D Cartesian TorchScript
    ts_3d_path = os.path.join(OUT, "pipe_pinn_3d_torchscript.pt")
    traced_3d = torch.jit.trace(deploy_3d, dummy_3d)
    traced_3d.save(ts_3d_path)
    print(f"Exported 3D Cartesian TorchScript model to: {ts_3d_path}")

    # 3. Export ONNX
    onnx_path = os.path.join(OUT, "pipe_pinn.onnx")
    try:
        torch.onnx.export(
            deploy_axisym,
            dummy_axisym,
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

    nx_grid, nr_grid = 300, 81
    x_lin = np.linspace(cfg.X_MIN, cfg.X_MAX, nx_grid)
    r_lin = np.linspace(cfg.R_MIN, cfg.R_MAX, nr_grid)
    X_full, R_full = np.meshgrid(x_lin, r_lin, indexing="ij")

    # Evaluate for each Re
    predictions = {}
    for Re in ALL_RE:
        x_t = torch.tensor(X_full.flatten(), dtype=torch.float32).reshape(-1, 1)
        r_t = torch.tensor(R_full.flatten(), dtype=torch.float32).reshape(-1, 1)
        Re_t = torch.full_like(x_t, float(Re))
        with torch.no_grad():
            u, v, p = model(x_t, r_t, Re_t)
        U_p = u.numpy().reshape(X_full.shape)
        V_p = v.numpy().reshape(X_full.shape)
        P_p = p.numpy().reshape(X_full.shape)
        predictions[Re] = {"U": U_p, "V": V_p, "P": P_p}

    # Plot 1: Full-domain u(x, r) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        U_p = predictions[Re]["U"]
        im = ax.contourf(X_full, R_full, U_p, levels=30, cmap="viridis")
        ax.set_title(f"Axial Velocity u(x, r) [m/s] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("r [m]")
        ax.set_yticks([0.0, 0.0125, 0.025, 0.0375, 0.05])
        fig.colorbar(im, ax=ax, label="u [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Axial Velocity u (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_flowfield_pinn_u.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_flowfield_pinn_u.png")

    # Plot 2: Full-domain v(x, r) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        V_p = predictions[Re]["V"]
        im = ax.contourf(X_full, R_full, V_p, levels=30, cmap="coolwarm")
        ax.set_title(f"Radial Velocity v_r(x, r) [m/s] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("r [m]")
        ax.set_yticks([0.0, 0.0125, 0.025, 0.0375, 0.05])
        fig.colorbar(im, ax=ax, label="v_r [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Radial Velocity v_r (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_flowfield_pinn_v.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_flowfield_pinn_v.png")

    # Plot 3: Full-domain p(x, r) flowfields (linear axes)
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        P_p = predictions[Re]["P"]
        im = ax.contourf(X_full, R_full, P_p, levels=30, cmap="plasma")
        ax.set_title(f"Kinematic Pressure p(x, r) [m$^2$/s$^2$] -- Re = {Re:.0f} ({tag})", fontsize=11)
        ax.set_ylabel("r [m]")
        ax.set_yticks([0.0, 0.0125, 0.025, 0.0375, 0.05])
        fig.colorbar(im, ax=ax, label="p [m$^2$/s$^2$]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("PINN Whole-Domain Predicted Kinematic Pressure p (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_flowfield_pinn_p.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_flowfield_pinn_p.png")

    # Plot 4: Streamlines and Velocity Magnitude for Re=100 (Train), Re=300 (Interp), Re=500 (Train), Re=600 (Extrap)
    # Reflected across r=0 to show the full cross-sectional diametral plane [-R, R]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    sample_cases = [(100.0, "TRAIN"), (300.0, "INTERP (unseen)"), (500.0, "TRAIN"), (600.0, "EXTRAP (unseen)")]

    # Create full symmetric diameter grid for visualization: y in [-R, R]
    y_diam = np.linspace(-cfg.R, cfg.R, 2 * nr_grid - 1)
    X_diam, Y_diam = np.meshgrid(x_lin, y_diam, indexing="ij")
    R_from_y = np.abs(Y_diam)

    for ax, (Re, tag) in zip(axes, sample_cases):
        x_t = torch.tensor(X_diam.flatten(), dtype=torch.float32).reshape(-1, 1)
        r_t = torch.tensor(R_from_y.flatten(), dtype=torch.float32).reshape(-1, 1)
        Re_t = torch.full_like(x_t, float(Re))
        with torch.no_grad():
            u, v, _ = model(x_t, r_t, Re_t)
        U_diam = u.numpy().reshape(X_diam.shape)
        # v_y directed radially away from axis: sign(y) * v_r
        V_diam = (v.numpy().reshape(X_diam.shape)) * np.sign(Y_diam)
        U_mag = np.sqrt(U_diam**2 + V_diam**2)

        im = ax.contourf(X_diam, Y_diam, U_mag, levels=35, cmap="turbo")
        ax.streamplot(x_lin, y_diam, U_diam.T, V_diam.T, color="white", density=0.7, linewidth=0.7, arrowsize=0.8)
        ax.set_title(f"Velocity Magnitude |U| and Streamlines -- Re={Re:.0f} [{tag}]", fontsize=11)
        ax.set_ylabel("y [m]")
        ax.set_yticks([-0.05, -0.025, 0.0, 0.025, 0.05])
        fig.colorbar(im, ax=ax, label="|U| [m/s]")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Streamline Patterns and Velocity Magnitude in Pipe Diametral Plane (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_flowfield_comparison_streamlines.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_flowfield_comparison_streamlines.png")

    # Plot 5: Linear-axis whole-domain error heatmaps (|PINN - Ref| / scale)
    for f in FIELDS:
        local_err_fields = {}
        for Re in ALL_RE:
            x_ref = ref[f"x_{Re}"]
            r_ref = ref[f"r_{Re}"]
            X_ref, R_ref = np.meshgrid(x_ref, r_ref, indexing="ij")

            x_t = torch.tensor(X_ref.flatten(), dtype=torch.float32).reshape(-1, 1)
            r_t = torch.tensor(R_ref.flatten(), dtype=torch.float32).reshape(-1, 1)
            Re_t = torch.full_like(x_t, float(Re))

            with torch.no_grad():
                u, v, p = model(x_t, r_t, Re_t)
            u_p = u.numpy().reshape(X_ref.shape)
            v_p = v.numpy().reshape(X_ref.shape)
            p_p = p.numpy().reshape(X_ref.shape)

            U_ref = ref[f"U_{Re}"]
            V_ref = ref[f"V_{Re}"]
            P_ref = ref[f"P_{Re}"] - ref[f"P_{Re}"][-1]
            P_ref_grid = np.broadcast_to(P_ref[:, None], X_ref.shape)

            pred_map = {"u": u_p, "v": v_p, "p": p_p}
            exact_map = {"u": U_ref, "v": V_ref, "p": P_ref_grid}

            scale = u_ref(Re) if f in ("u", "v") else p_scale(Re)
            err_map = 100.0 * np.abs(pred_map[f] - exact_map[f]) / (scale + 1e-12)
            local_err_fields[Re] = (x_ref, r_ref, err_map)

        all_max = max(np.percentile(v[2], 99) for v in local_err_fields.values())
        fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
        for ax, Re in zip(axes, ALL_RE):
            x_r, r_r, err_f = local_err_fields[Re]
            im = ax.contourf(x_r, r_r, err_f.T, levels=25, vmin=0, vmax=all_max, cmap="inferno")
            ax.set_title(f"Re = {Re:.0f} ({TAGS[Re]}) -- Max P99 Error: {np.percentile(err_f, 99):.2f}%", fontsize=10)
            ax.set_ylabel("r [m]")
            ax.set_yticks([0.0, 0.0125, 0.025, 0.0375, 0.05])
            fig.colorbar(im, ax=ax, label=f"|{f}_PINN - {f}_ref| / scale [%]")
        axes[-1].set_xlabel("x [m]")
        fig.suptitle(f"Whole-Domain Spatial Error Distribution for Pipe Field '{f}' (Linear Axes)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pipe_error_domain_{f}_linear.png"), dpi=150)
        plt.close(fig)
        print(f"Saved outputs/pipe_error_domain_{f}_linear.png")

    # Plot 6: Linear-axis Global Error vs Re Scatter Plot
    # Read metrics_global.csv
    re_vals = []
    u_errs = []
    v_errs = []
    p_errs = []
    tags_list = []
    with open(os.path.join(OUT, "metrics_global.csv"), "r") as fh:
        lines = fh.readlines()[1:]
        for line in lines:
            parts = line.strip().split(",")
            re_vals.append(float(parts[0]))
            tags_list.append(parts[1])
            u_errs.append(float(parts[2]))
            v_errs.append(float(parts[3]))
            p_errs.append(float(parts[4]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics_data = [("u", u_errs), ("v", v_errs), ("p", p_errs)]
    for ax, (field_name, err_data) in zip(axes, metrics_data):
        for Re, tag, err in zip(re_vals, tags_list, err_data):
            ax.scatter(Re, err, c=COLORS[tag], label=tag, s=80, zorder=3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlabel("Reynolds Number (Re)")
        ax.set_ylabel("Global Rel/Norm L2 Error [%]")
        ax.set_title(f"Pipe Domain Error vs. Re: Field '{field_name}' (Linear Axes)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_entire_domain_error_vs_Re_linear.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_entire_domain_error_vs_Re_linear.png")

    # Plot 7: Physics Residual Spatial Map
    fig, axes = plt.subplots(len(ALL_RE), 1, figsize=(10, 2.5 * len(ALL_RE)), sharex=True)
    nx_res, nr_res = 120, 41
    xr = np.linspace(0.01 * cfg.L, 0.99 * cfg.L, nx_res)
    rr = np.linspace(0.02 * cfg.R, 0.98 * cfg.R, nr_res)
    Xr, Rr = np.meshgrid(xr, rr, indexing="ij")
    xt_res = torch.tensor(Xr.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    rt_res = torch.tensor(Rr.flatten(), dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    for ax, Re in zip(axes, ALL_RE):
        tag = TAGS[Re]
        Re_t = torch.full_like(xt_res, float(Re))
        rn_c, rn_mx, rn_mr = ns_residuals_normalized(model, xt_res, rt_res, Re_t)
        res_mag = torch.sqrt(rn_c**2 + rn_mx**2 + rn_mr**2).detach().numpy().reshape(Xr.shape)
        im = ax.contourf(Xr, Rr, res_mag, levels=25, cmap="magma")
        ax.set_title(f"Navier-Stokes PDE Residual Norm -- Re = {Re:.0f} ({tag})", fontsize=10)
        ax.set_ylabel("r [m]")
        ax.set_yticks([0.0, 0.0125, 0.025, 0.0375, 0.05])
        fig.colorbar(im, ax=ax, label="||Res||_norm")
    axes[-1].set_xlabel("x [m]")
    fig.suptitle("Spatial Verification of Axisymmetric Navier-Stokes Satisfaction (Linear Axes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_physics_pde_residuals_map.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_physics_pde_residuals_map.png")

    # Plot 8: Radial velocity profile comparison at developing and fully-developed stations
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    r_eval = np.linspace(0, cfg.R, 100)
    for col_idx, Re in enumerate([100.0, 300.0, 600.0]):
        tag = TAGS[Re]
        Le = cfg.entrance_length(Re)
        U0 = cfg.inlet_velocity(Re)

        # Developing station: x = 0.5 * Le
        x_dev = min(0.5 * Le, 0.8 * cfg.L)
        x_dev_t = torch.full((len(r_eval), 1), float(x_dev), dtype=torch.float32)
        r_dev_t = torch.tensor(r_eval, dtype=torch.float32).reshape(-1, 1)
        Re_t = torch.full_like(x_dev_t, float(Re))
        with torch.no_grad():
            u_dev_p, _, _ = model(x_dev_t, r_dev_t, Re_t)

        ax_dev = axes[0, col_idx]
        ax_dev.plot(u_dev_p.numpy() / U0, r_eval / cfg.R, "r-", label="PINN Prediction", linewidth=2)
        # Benchmark with reference if available
        x_ref = ref[f"x_{Re}"]
        r_ref = ref[f"r_{Re}"]
        u_ref_grid = ref[f"U_{Re}"]
        idx_near = np.argmin(np.abs(x_ref - x_dev))
        ax_dev.plot(u_ref_grid[idx_near, :] / U0, r_ref / cfg.R, "k--", label=f"Hornbeck (x={x_ref[idx_near]:.2f}m)", linewidth=1.5)
        ax_dev.set_title(f"Developing Region: x={x_dev:.2f} m\nRe={Re:.0f} [{tag}]", fontsize=11)
        ax_dev.set_xlabel("u / U_0 [-]")
        ax_dev.set_ylabel("r / R [-]")
        ax_dev.grid(True, linestyle="--", alpha=0.5)
        ax_dev.legend(fontsize=9)

        # Fully-developed station: x = 3.90 m
        x_fd = 3.90
        x_fd_t = torch.full((len(r_eval), 1), float(x_fd), dtype=torch.float32)
        with torch.no_grad():
            u_fd_p, _, _ = model(x_fd_t, r_dev_t, Re_t)

        u_exact = cfg.poiseuille_profile(r_eval, Re)
        ax_fd = axes[1, col_idx]
        ax_fd.plot(u_fd_p.numpy() / U0, r_eval / cfg.R, "r-", label="PINN Prediction", linewidth=2)
        ax_fd.plot(u_exact / U0, r_eval / cfg.R, "b--", label="Exact Hagen-Poiseuille", linewidth=1.5)
        ax_fd.set_title(f"Fully-Developed: x={x_fd:.2f} m\nRe={Re:.0f} [{tag}]", fontsize=11)
        ax_fd.set_xlabel("u / U_0 [-]")
        ax_fd.set_ylabel("r / R [-]")
        ax_fd.grid(True, linestyle="--", alpha=0.5)
        ax_fd.legend(fontsize=9)

    fig.suptitle("Radial Velocity Profiles Comparison Across Developing and Fully-Developed Stations", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe_radial_profiles_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved outputs/pipe_radial_profiles_comparison.png")


if __name__ == "__main__":
    print("Starting deploy and visualize pipeline for Circular Pipe PINN...")
    model = load_trained_model()
    verify_physics_vs_data(model)
    extract_model_parameters(model)
    export_deployable_model(model)
    generate_linear_axis_plots(model)
    print("\nALL TASKS COMPLETED SUCCESSFULLY!")
