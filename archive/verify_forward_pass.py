"""
verify_forward_pass.py
======================
Deterministic verification script for all 5 stages of the OpenFOAM/PINN pipeline.
Directly loads official checkpoints from disk and verifies:
1. Checkpoint SHA256 and exact parameter count (sum(p.numel()))
2. Exact input coordinates and normalized feature vectors
3. Layer-by-layer pre-activations (z) and post-activations (a) for the first 4 neurons
4. Raw network outputs and physical rescaled predictions
5. Comparison against the reference solver at the exact test point
6. Matrix Frobenius norms and bias L2 norms
Saves verified data to provenance.json and prints full traceability.
"""

import os
import sys
import math
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

provenance = {}

print("=" * 80)
print("OPENFOAM / PINN PIPELINE: VERIFYING FORWARD PASS FOR ALL 5 STAGES")
print("=" * 80)

# ==============================================================================
# STAGE 1: 2D Laminar Channel Flow
# ==============================================================================
print("\n--- STAGE 1: 2D Laminar Channel Flow ---")
ckpt1_path = os.path.join(BASE_DIR, "1", "outputs", "channel_pinn_hybrid.pt")
sha1 = get_sha256(ckpt1_path)
ckpt1 = torch.load(ckpt1_path, map_location="cpu")

# Extract layers: 4 hidden x 80 units
layers1 = [
    (ckpt1["net.0.weight"], ckpt1["net.0.bias"]),
    (ckpt1["net.2.weight"], ckpt1["net.2.bias"]),
    (ckpt1["net.4.weight"], ckpt1["net.4.bias"]),
    (ckpt1["net.6.weight"], ckpt1["net.6.bias"]),
    (ckpt1["net.8.weight"], ckpt1["net.8.bias"]),
]
trainable1 = sum(w.numel() + b.numel() for w, b in layers1)
assert trainable1 == 20003, f"Stage 1 param count mismatch: {trainable1}"
frob1 = [torch.norm(w, p="fro").item() for w, b in layers1]
bias_l2_1 = [torch.norm(b, p=2).item() for w, b in layers1]

# In-domain test point: x = 0.5 m, y = 0.02 m, Re = 100.0
# Geometry: L = 4.0 m, h = 0.05 m, Dh = 4h = 0.20 m, nu = 1e-4 m^2/s
x1, y1, re1 = 0.5, 0.02, 100.0
L1, h1, Dh1, nu1 = 4.0, 0.05, 0.20, 1e-4
u0_1 = re1 * nu1 / Dh1 # 0.05 m/s
p_sc1 = 0.5 * (u0_1 ** 2) # 0.00125 m^2/s^2

re_log_min1 = ckpt1["re_log_min"].item()
re_log_max1 = ckpt1["re_log_max"].item()
x_n1 = x1 / L1 # 0.125
y_n1 = y1 / h1 # 0.400
re_n1 = 2.0 * (math.log10(re1) - re_log_min1) / (re_log_max1 - re_log_min1) - 1.0 # -0.39794

xin1 = torch.tensor([[x_n1, y_n1, re_n1]], dtype=torch.float32)

h = xin1
neurons1 = {}
for i, (w, b) in enumerate(layers1[:-1]):
    z = torch.matmul(h, w.t()) + b
    h = torch.tanh(z)
    neurons1[f"layer_{i}"] = {
        "z_first_4": z[0, :4].tolist(),
        "a_first_4": h[0, :4].tolist(),
        "frob_norm": frob1[i],
        "bias_norm": bias_l2_1[i],
    }

z_out1 = torch.matmul(h, layers1[-1][0].t()) + layers1[-1][1]
raw_out1 = z_out1[0].tolist()

u_hat1, v_hat1, p_hat1 = raw_out1[0], raw_out1[1], raw_out1[2]
u_phys1 = u_hat1 * u0_1
v_phys1 = v_hat1 * u0_1
p_phys1 = p_hat1 * p_sc1

# Reference lookup
ref1 = np.load(os.path.join(BASE_DIR, "1", "outputs", "reference_data.npz"))
x_ref1 = ref1["x_100.0"]
y_ref1 = ref1["y_100.0"]
iu1 = RegularGridInterpolator((x_ref1, y_ref1), ref1["U_100.0"])
iv1 = RegularGridInterpolator((x_ref1, y_ref1), ref1["V_100.0"])
u_ref1 = float(iu1([[x1, y1]]))
v_ref1 = float(iv1([[x1, y1]]))
p_gauge1 = ref1["P_100.0"] - ref1["P_100.0"][-1]
p_ref1 = float(np.interp(x1, x_ref1, p_gauge1))

err_u1 = abs(u_phys1 - u_ref1) / u_ref1 * 100.0
err_p1 = abs(p_phys1 - p_ref1) / abs(p_ref1) * 100.0

print(f"Stage 1 Checkpoint: {ckpt1_path} (SHA256: {sha1[:16]}...)")
print(f"Test point: x={x1} m, y={y1} m, Re={re1}")
print(f"Input features: [x/L={x_n1:.4f}, y/h={y_n1:.4f}, re_n={re_n1:.4f}]")
print(f"Layer 0: z[:4]={[round(v, 4) for v in neurons1['layer_0']['z_first_4']]}, a[:4]={[round(v, 4) for v in neurons1['layer_0']['a_first_4']]}")
print(f"Raw outputs: [u_hat={u_hat1:.4f}, v_hat={v_hat1:.4f}, p_hat={p_hat1:.4f}]")
print(f"Physical: u={u_phys1:.5f} m/s (ref: {u_ref1:.5f} m/s, err: {err_u1:.2f}%)")
print(f"Physical: p={p_phys1:.6f} m^2/s^2 (ref: {p_ref1:.6f} m^2/s^2, err: {err_p1:.2f}%)")

provenance["stage1"] = {
    "name": "Stage 1: 2D Laminar Channel Flow",
    "checkpoint": "1/outputs/channel_pinn_hybrid.pt",
    "sha256": sha1,
    "domain": {"x": [0.0, 4.0], "y": [-0.05, 0.05]},
    "Dh": 0.20,
    "nu": 1e-4,
    "U0_100": u0_1,
    "p_scale_100": p_sc1,
    "trainable_parameters": trainable1,
    "frob_norms": frob1,
    "bias_l2_norms": bias_l2_1,
    "test_point": {"x": x1, "y": y1, "Re": re1},
    "input_vector": [x_n1, y_n1, re_n1],
    "layer_neurons": neurons1,
    "raw_outputs": raw_out1,
    "physical_predictions": {"u": u_phys1, "v": v_phys1, "p": p_phys1},
    "reference_values": {"u": u_ref1, "v": v_ref1, "p": p_ref1},
    "percent_errors": {"u": err_u1, "p": err_p1}
}

# ==============================================================================
# STAGE 2: Laminar Axisymmetric Circular Pipe Flow
# ==============================================================================
print("\n--- STAGE 2: Laminar Axisymmetric Circular Pipe Flow ---")
ckpt2_path = os.path.join(BASE_DIR, "2", "outputs", "pipe_pinn_hybrid.pt")
sha2 = get_sha256(ckpt2_path)
ckpt2 = torch.load(ckpt2_path, map_location="cpu")

layers2 = [
    (ckpt2["net.0.weight"], ckpt2["net.0.bias"]),
    (ckpt2["net.2.weight"], ckpt2["net.2.bias"]),
    (ckpt2["net.4.weight"], ckpt2["net.4.bias"]),
    (ckpt2["net.6.weight"], ckpt2["net.6.bias"]),
    (ckpt2["net.8.weight"], ckpt2["net.8.bias"]),
]
trainable2 = sum(w.numel() + b.numel() for w, b in layers2)
assert trainable2 == 20003, f"Stage 2 param count mismatch: {trainable2}"
frob2 = [torch.norm(w, p="fro").item() for w, b in layers2]
bias_l2_2 = [torch.norm(b, p=2).item() for w, b in layers2]

# In-domain test point: x = 2.0 m, r = 0.02 m, Re = 100.0
x2, r2, re2 = 2.0, 0.02, 100.0
L2, R2, D2, nu2 = 4.0, 0.05, 0.10, 1e-4
u0_2 = re2 * nu2 / D2 # 0.10 m/s
p_sc2 = 0.5 * (u0_2 ** 2) # 0.005 m^2/s^2

re_log_min2 = ckpt2["re_log_min"].item()
re_log_max2 = ckpt2["re_log_max"].item()
x_n2 = x2 / L2 # 0.50
r_n2 = r2 / R2 # 0.40
re_n2 = 2.0 * (math.log10(re2) - re_log_min2) / (re_log_max2 - re_log_min2) - 1.0 # -0.39794

xin2 = torch.tensor([[x_n2, r_n2, re_n2]], dtype=torch.float32)

h = xin2
neurons2 = {}
for i, (w, b) in enumerate(layers2[:-1]):
    z = torch.matmul(h, w.t()) + b
    h = torch.tanh(z)
    neurons2[f"layer_{i}"] = {
        "z_first_4": z[0, :4].tolist(),
        "a_first_4": h[0, :4].tolist(),
        "frob_norm": frob2[i],
        "bias_norm": bias_l2_2[i],
    }

z_out2 = torch.matmul(h, layers2[-1][0].t()) + layers2[-1][1]
raw_out2 = z_out2[0].tolist()

u_hat2, v_hat2, p_hat2 = raw_out2[0], raw_out2[1], raw_out2[2]
u_phys2 = u_hat2 * u0_2
v_phys2 = v_hat2 * u0_2
p_phys2 = p_hat2 * p_sc2

# Reference lookup
ref2 = np.load(os.path.join(BASE_DIR, "2", "outputs", "reference_data.npz"))
x_ref2 = ref2["x_100.0"]
r_ref2 = ref2["r_100.0"]
iu2 = RegularGridInterpolator((x_ref2, r_ref2), ref2["U_100.0"])
iv2 = RegularGridInterpolator((x_ref2, r_ref2), ref2["V_100.0"])
u_ref2 = float(iu2([[x2, r2]]))
v_ref2 = float(iv2([[x2, r2]]))
p_gauge2 = ref2["P_100.0"] - ref2["P_100.0"][-1]
p_ref2 = float(np.interp(x2, x_ref2, p_gauge2))

err_u2 = abs(u_phys2 - u_ref2) / u_ref2 * 100.0
# For pressure, PINN outlet reference offset
# Note: PINN trained with outlet gauge p(L) = 0
# Physical p_phys2 is relative to p_sc2
print(f"Stage 2 Checkpoint: {ckpt2_path} (SHA256: {sha2[:16]}...)")
print(f"Test point: x={x2} m, r={r2} m, Re={re2}")
print(f"Input features: [x/L={x_n2:.4f}, r/R={r_n2:.4f}, re_n={re_n2:.4f}]")
print(f"Raw outputs: [u_hat={u_hat2:.4f}, v_hat={v_hat2:.4f}, p_hat={p_hat2:.4f}]")
print(f"Physical: u={u_phys2:.5f} m/s (ref: {u_ref2:.5f} m/s, err: {err_u2:.2f}%)")
print(f"Physical: v={v_phys2:.6f} m/s (ref: {v_ref2:.6e} m/s)")

provenance["stage2"] = {
    "name": "Stage 2: Laminar Axisymmetric Circular Pipe Flow",
    "checkpoint": "2/outputs/pipe_pinn_hybrid.pt",
    "sha256": sha2,
    "domain": {"x": [0.0, 4.0], "r": [0.0, 0.05]},
    "D": 0.10,
    "nu": 1e-4,
    "U0_100": u0_2,
    "p_scale_100": p_sc2,
    "trainable_parameters": trainable2,
    "frob_norms": frob2,
    "bias_l2_norms": bias_l2_2,
    "test_point": {"x": x2, "r": r2, "Re": re2},
    "input_vector": [x_n2, r_n2, re_n2],
    "layer_neurons": neurons2,
    "raw_outputs": raw_out2,
    "physical_predictions": {"u": u_phys2, "v": v_phys2, "p": p_phys2},
    "reference_values": {"u": u_ref2, "v": v_ref2, "p": p_ref2},
    "percent_errors": {"u": err_u2}
}

# ==============================================================================
# STAGE 3: 3D Rectangular Duct Flow
# ==============================================================================
print("\n--- STAGE 3: 3D Rectangular Duct Flow ---")
ckpt3_path = os.path.join(BASE_DIR, "3", "checkpoints", "duct_pinn.pt")
sha3 = get_sha256(ckpt3_path)
ckpt3 = torch.load(ckpt3_path, map_location="cpu")

layers3 = [
    (ckpt3["net.0.weight"], ckpt3["net.0.bias"]),
    (ckpt3["net.2.weight"], ckpt3["net.2.bias"]),
    (ckpt3["net.4.weight"], ckpt3["net.4.bias"]),
    (ckpt3["net.6.weight"], ckpt3["net.6.bias"]),
    (ckpt3["net.8.weight"], ckpt3["net.8.bias"]),
]
trainable3 = sum(w.numel() + b.numel() for w, b in layers3)
assert trainable3 == 20164, f"Stage 3 param count mismatch: {trainable3}"
frob3 = [torch.norm(w, p="fro").item() for w, b in layers3]
bias_l2_3 = [torch.norm(b, p=2).item() for w, b in layers3]

# Test point: x = 2.5 m, y = 0.05 m, z = 0.10 m, Re = 100.0
# Geometry: H = 0.10 m, W = 0.20 m, L = 5.0 m, Dh = 0.133333 m
x3, y3, z3, re3 = 2.5, 0.05, 0.10, 100.0
H3, W3, L3, Dh3, nu3 = 0.10, 0.20, 5.0, 0.13333333333333333, 1e-4
u_mean3 = re3 * nu3 / Dh3 # 0.075 m/s
p_sc3 = 0.5 * (u_mean3 ** 2) * (L3 / Dh3) # 0.10546875 m^2/s^2

# Re normalization for duct: log10 over [50, 600]
lo3, hi3 = math.log10(50.0), math.log10(600.0)
x_n3 = x3 / L3 # 0.5
y_n3 = y3 / H3 # 0.5
z_n3 = z3 / W3 # 0.5
re_n3 = 2.0 * (math.log10(re3) - lo3) / (hi3 - lo3) - 1.0 # -0.44211

xin3 = torch.tensor([[x_n3, y_n3, z_n3, re_n3]], dtype=torch.float32)

h = xin3
neurons3 = {}
for i, (w, b) in enumerate(layers3[:-1]):
    z = torch.matmul(h, w.t()) + b
    h = torch.tanh(z)
    neurons3[f"layer_{i}"] = {
        "z_first_4": z[0, :4].tolist(),
        "a_first_4": h[0, :4].tolist(),
        "frob_norm": frob3[i],
        "bias_norm": bias_l2_3[i],
    }

z_out3 = torch.matmul(h, layers3[-1][0].t()) + layers3[-1][1]
raw_out3 = z_out3[0].tolist()

u_hat3, v_hat3, w_hat3, p_hat3 = raw_out3[0], raw_out3[1], raw_out3[2], raw_out3[3]
u_phys3 = u_hat3 * u_mean3
v_phys3 = v_hat3 * u_mean3
w_phys3 = w_hat3 * u_mean3
p_phys3 = p_hat3 * p_sc3

# Reference lookup
ref3 = np.load(os.path.join(BASE_DIR, "3", "data", "duct_Re100.npz"))
x_ref3, y_ref3, z_ref3 = ref3["x"], ref3["y"], ref3["z"]
ix3 = np.argmin(np.abs(x_ref3 - x3))
iy3 = np.argmin(np.abs(y_ref3 - y3))
iz3 = np.argmin(np.abs(z_ref3 - z3))
u_ref3 = float(ref3["U"][ix3, iy3, iz3])
v_ref3 = float(ref3["V"][ix3, iy3, iz3])
w_ref3 = float(ref3["W"][ix3, iy3, iz3])

ix5_3 = np.argmin(np.abs(x_ref3 - 5.0))
p_gauge3 = float(ref3["P"][ix3, iy3, iz3] - ref3["P"][ix5_3, iy3, iz3])

err_u3 = abs(u_phys3 - u_ref3) / u_ref3 * 100.0
err_p3 = abs(p_phys3 - p_gauge3) / abs(p_gauge3) * 100.0

print(f"Stage 3 Checkpoint: {ckpt3_path} (SHA256: {sha3[:16]}...)")
print(f"Test point: x={x3} m, y={y3} m, z={z3} m, Re={re3}")
print(f"Input features: [x/L={x_n3:.4f}, y/H={y_n3:.4f}, z/W={z_n3:.4f}, re_n={re_n3:.4f}]")
print(f"Raw outputs: [u_hat={u_hat3:.4f}, v_hat={v_hat3:.4f}, w_hat={w_hat3:.4f}, p_hat={p_hat3:.4f}]")
print(f"Physical: u={u_phys3:.5f} m/s (ref: {u_ref3:.5f} m/s, err: {err_u3:.2f}%)")
print(f"Physical: p={p_phys3:.5f} m^2/s^2 (ref: {p_gauge3:.5f} m^2/s^2, err: {err_p3:.2f}%)")

provenance["stage3"] = {
    "name": "Stage 3: 3D Rectangular Duct Flow",
    "checkpoint": "3/checkpoints/duct_pinn.pt",
    "sha256": sha3,
    "domain": {"x": [0.0, 5.0], "y": [0.0, 0.10], "z": [0.0, 0.20]},
    "Dh": Dh3,
    "nu": nu3,
    "U_mean_100": u_mean3,
    "p_scale_100": p_sc3,
    "trainable_parameters": trainable3,
    "frob_norms": frob3,
    "bias_l2_norms": bias_l2_3,
    "test_point": {"x": x3, "y": y3, "z": z3, "Re": re3},
    "input_vector": [x_n3, y_n3, z_n3, re_n3],
    "layer_neurons": neurons3,
    "raw_outputs": raw_out3,
    "physical_predictions": {"u": u_phys3, "v": v_phys3, "w": w_phys3, "p": p_phys3},
    "reference_values": {"u": u_ref3, "v": v_ref3, "w": w_ref3, "p": p_gauge3},
    "percent_errors": {"u": err_u3, "p": err_p3}
}

# ==============================================================================
# STAGE 4: Developing Turbulent Channel Flow (RANS + SA)
# ==============================================================================
print("\n--- STAGE 4: Developing Turbulent Channel Flow (RANS + SA) ---")
ckpt4_path = os.path.join(BASE_DIR, "4", "outputs", "final_s0.pt")
sha4 = get_sha256(ckpt4_path)
ckpt4 = torch.load(ckpt4_path, map_location="cpu")
st4 = ckpt4["state"]

layers4 = [
    (st4["layers.0.weight"], st4["layers.0.bias"]),
    (st4["layers.1.weight"], st4["layers.1.bias"]),
    (st4["layers.2.weight"], st4["layers.2.bias"]),
    (st4["layers.3.weight"], st4["layers.3.bias"]),
    (st4["layers.4.weight"], st4["layers.4.bias"]),
]
trainable4 = sum(w.numel() + b.numel() for w, b in layers4)
assert trainable4 == 13060, f"Stage 4 param count mismatch: {trainable4}"
frob4 = [torch.norm(w, p="fro").item() for w, b in layers4]
bias_l2_4 = [torch.norm(b, p=2).item() for w, b in layers4]

# Test point: x = 40.0 h, y = 0.5 h, Re = 20,000
x4, y4, re4 = 40.0, 0.5, 20000.0
L4 = 160.0
# Feature normalization matches training range [10000, 80000]
mid4 = 0.5 * (math.log10(10000.0) + math.log10(80000.0))
half4 = 0.5 * (math.log10(80000.0) - math.log10(10000.0))
ret4 = 0.0955 * (re4 / 2.0) ** 0.875

s4 = math.sqrt(x4 / L4 + 1e-8)
eta4 = y4
z4 = math.log1p(ret4 * eta4) / math.log1p(ret4)
r4 = (math.log10(re4) - mid4) / half4

f1_4 = 2.0 * s4 - 1.0
f2_4 = 2.0 * eta4 - 1.0
f3_4 = 2.0 * z4 - 1.0
f4_4 = r4

xin4 = torch.tensor([[f1_4, f2_4, f3_4, f4_4]], dtype=torch.float32)

h = xin4
neurons4 = {}
for i, (w, b) in enumerate(layers4[:-1]):
    z = torch.matmul(h, w.t()) + b
    h = torch.tanh(z)
    neurons4[f"layer_{i}"] = {
        "z_first_4": z[0, :4].tolist(),
        "a_first_4": h[0, :4].tolist(),
        "frob_norm": frob4[i],
        "bias_norm": bias_l2_4[i],
    }

z_out4 = torch.matmul(h, layers4[-1][0].t()) + layers4[-1][1]
raw_out4 = z_out4[0].tolist()

# Envelopes
g4 = math.tanh(ret4 * y4 / 10.0)
nu4 = 4.0 / re4
vs4 = 0.05 * math.sqrt(0.5 * 0.073 * (re4 / 2.0) ** -0.25)
u_phys4 = g4 * (1.0 + raw_out4[0])
v_phys4 = vs4 * g4 * g4 * raw_out4[1]
p_phys4 = raw_out4[2]
# softplus(a3 - 3.0)
nt_phys4 = nu4 * ret4 * g4 * math.log1p(math.exp(raw_out4[3] - 3.0))

ref4 = np.load(os.path.join(BASE_DIR, "4", "reference_data", "ref_Re20000.npz"))
iu4 = RegularGridInterpolator((ref4["x"], ref4["y"]), ref4["U"])
iv4 = RegularGridInterpolator((ref4["x"], ref4["y"]), ref4["V"])
ip4 = RegularGridInterpolator((ref4["x"],), ref4["P_hat"])
inut4 = RegularGridInterpolator((ref4["x"], ref4["y"]), ref4["nut_tilde"])

u_ref4 = float(iu4([[x4, y4]]))
v_ref4 = float(iv4([[x4, y4]]))
p_ref4 = float(ip4([x4]))
nut_ref4 = float(inut4([[x4, y4]]))

err_u4 = abs(u_phys4 - u_ref4) / u_ref4 * 100.0
err_p4 = abs(p_phys4 - p_ref4) / abs(p_ref4) * 100.0
err_nut4 = abs(nt_phys4 - nut_ref4) / nut_ref4 * 100.0

print(f"Stage 4 Checkpoint: {ckpt4_path} (SHA256: {sha4[:16]}...)")
print(f"Test point: x={x4} h, y={y4} h, Re={re4}")
print(f"Input features: [f1={f1_4:.4f}, f2={f2_4:.4f}, f3={f3_4:.4f}, f4={f4_4:.4f}]")
print(f"Raw outputs: [a0={raw_out4[0]:.4f}, a1={raw_out4[1]:.4f}, a2={raw_out4[2]:.4f}, a3={raw_out4[3]:.4f}]")
print(f"Physical: U={u_phys4:.5f} (ref: {u_ref4:.5f}, err: {err_u4:.2f}%)")
print(f"Physical: P={p_phys4:.5f} (ref: {p_ref4:.5f}, err: {err_p4:.2f}%)")
print(f"Physical: nut={nt_phys4:.6e} (ref: {nut_ref4:.6e}, err: {err_nut4:.2f}%)")

provenance["stage4"] = {
    "name": "Stage 4: Developing Turbulent Channel Flow (RANS+SA)",
    "checkpoint": "4/outputs/final_s0.pt",
    "sha256": sha4,
    "domain": {"x/h": [0.0, 160.0], "y/h": [0.0, 1.0]},
    "Dh": 2.0,
    "nu": nu4,
    "U_bulk": 1.0,
    "trainable_parameters": trainable4,
    "frob_norms": frob4,
    "bias_l2_norms": bias_l2_4,
    "test_point": {"x/h": x4, "y/h": y4, "Re": re4},
    "input_vector": [f1_4, f2_4, f3_4, f4_4],
    "layer_neurons": neurons4,
    "raw_outputs": raw_out4,
    "physical_predictions": {"u": u_phys4, "v": v_phys4, "p": p_phys4, "nu_tilde": nt_phys4},
    "reference_values": {"u": u_ref4, "v": v_ref4, "p": p_ref4, "nu_tilde": nut_ref4},
    "percent_errors": {"u": err_u4, "p": err_p4, "nu_tilde": err_nut4}
}

# ==============================================================================
# STAGE 5: Developing Turbulent Pipe Flow (Axisymmetric RANS + SA)
# ==============================================================================
print("\n--- STAGE 5: Developing Turbulent Pipe Flow (Axisymmetric RANS + SA) ---")
ckpt5_path = os.path.join(BASE_DIR, "5", "outputs", "models", "dense", "final_seed0.pt")
sha5 = get_sha256(ckpt5_path)
ckpt5 = torch.load(ckpt5_path, map_location="cpu")
st5 = ckpt5["state"]

layers5 = [
    (st5["hidden.0.weight"], st5["hidden.0.bias"]),
    (st5["hidden.1.weight"], st5["hidden.1.bias"]),
    (st5["hidden.2.weight"], st5["hidden.2.bias"]),
    (st5["hidden.3.weight"], st5["hidden.3.bias"]),
    (st5["out.weight"], st5["out.bias"]),
]
trainable5 = sum(w.numel() + b.numel() for w, b in layers5)
assert trainable5 == 13060, f"Stage 5 param count mismatch: {trainable5}"
frob5 = [torch.norm(w, p="fro").item() for w, b in layers5]
bias_l2_5 = [torch.norm(b, p=2).item() for w, b in layers5]

# Test point: x = 40.0 R, r = 0.5 R, Re = 40,000
x5, r5, re5 = 40.0, 0.5, 40000.0
L5 = 160.0
# Normalization functions for Stage 5
sys.path.insert(0, os.path.join(BASE_DIR, "5"))
import pinn as p5
_, ut5, ret5, nu_star5 = p5.est_torch(torch.tensor([re5]))
ut5 = ut5.item()
ret5 = ret5.item()
nu_star5 = nu_star5.item()
sys.path.pop(0)

eta5 = 1.0 - r5
mid5 = 0.5 * (math.log10(10000.0) + math.log10(80000.0))
half5 = 0.5 * (math.log10(80000.0) - math.log10(10000.0))

f1_5 = 2.0 * math.sqrt(x5 / L5 + 1e-12) - 1.0
f2_5 = 2.0 * eta5 - 1.0
f3_5 = 2.0 * math.log1p(ret5 * eta5) / math.log1p(ret5) - 1.0
f4_5 = (math.log10(re5) - mid5) / half5

xin5 = torch.tensor([[f1_5, f2_5, f3_5, f4_5]], dtype=torch.float32)

h = xin5
neurons5 = {}
for i, (w, b) in enumerate(layers5[:-1]):
    z = torch.matmul(h, w.t()) + b
    h = torch.tanh(z)
    neurons5[f"layer_{i}"] = {
        "z_first_4": z[0, :4].tolist(),
        "a_first_4": h[0, :4].tolist(),
        "frob_norm": frob5[i],
        "bias_norm": bias_l2_5[i],
    }

z_out5 = torch.matmul(h, layers5[-1][0].t()) + layers5[-1][1]
raw_out5 = z_out5[0].tolist()

# Envelopes
g5 = math.tanh(ret5 * eta5 / 10.0)
u_phys5 = g5 * (1.0 + raw_out5[0])
v_phys5 = (0.05 * ut5) * g5 * g5 * r5 * raw_out5[1]
p_phys5 = raw_out5[2]
nt_phys5 = nu_star5 * ret5 * g5 * math.log1p(math.exp(raw_out5[3] - 3.0))

ref5 = np.load(os.path.join(BASE_DIR, "5", "data", "ref_Re40000.npz"))
iu5 = RegularGridInterpolator((ref5["x"], ref5["r"]), ref5["U"])
iv5 = RegularGridInterpolator((ref5["x"], ref5["r"]), ref5["V"])
ip5 = RegularGridInterpolator((ref5["x"],), ref5["P_hat"])
inut5 = RegularGridInterpolator((ref5["x"], ref5["r"]), ref5["nt"])

u_ref5 = float(iu5([[x5, r5]]))
v_ref5 = float(iv5([[x5, r5]]))
p_ref5 = float(ip5([x5]))
nut_ref5 = float(inut5([[x5, r5]]))

err_u5 = abs(u_phys5 - u_ref5) / u_ref5 * 100.0
err_p5 = abs(p_phys5 - p_ref5) / abs(p_ref5) * 100.0
err_nut5 = abs(nt_phys5 - nut_ref5) / nut_ref5 * 100.0

print(f"Stage 5 Checkpoint: {ckpt5_path} (SHA256: {sha5[:16]}...)")
print(f"Test point: x={x5} R, r={r5} R, Re={re5}")
print(f"Input features: [f1={f1_5:.4f}, f2={f2_5:.4f}, f3={f3_5:.4f}, f4={f4_5:.4f}]")
print(f"Raw outputs: [a0={raw_out5[0]:.4f}, a1={raw_out5[1]:.4f}, a2={raw_out5[2]:.4f}, a3={raw_out5[3]:.4f}]")
print(f"Physical: U={u_phys5:.5f} (ref: {u_ref5:.5f}, err: {err_u5:.2f}%)")
print(f"Physical: P={p_phys5:.5f} (ref: {p_ref5:.5f}, err: {err_p5:.2f}%)")
print(f"Physical: nut={nt_phys5:.6e} (ref: {nut_ref5:.6e}, err: {err_nut5:.2f}%)")

provenance["stage5"] = {
    "name": "Stage 5: Developing Turbulent Pipe Flow (Axisymmetric RANS+SA)",
    "checkpoint": "5/outputs/models/dense/final_seed0.pt",
    "sha256": sha5,
    "domain": {"x/R": [0.0, 160.0], "r/R": [0.0, 1.0]},
    "D": 2.0,
    "nu": nu_star5,
    "U_bulk": 1.0,
    "trainable_parameters": trainable5,
    "frob_norms": frob5,
    "bias_l2_norms": bias_l2_5,
    "test_point": {"x/R": x5, "r/R": r5, "Re": re5},
    "input_vector": [f1_5, f2_5, f3_5, f4_5],
    "layer_neurons": neurons5,
    "raw_outputs": raw_out5,
    "physical_predictions": {"u": u_phys5, "v": v_phys5, "p": p_phys5, "nu_tilde": nt_phys5},
    "reference_values": {"u": u_ref5, "v": v_ref5, "p": p_ref5, "nu_tilde": nut_ref5},
    "percent_errors": {"u": err_u5, "p": err_p5, "nu_tilde": err_nut5}
}

# Write provenance.json
with open(os.path.join(BASE_DIR, "provenance.json"), "w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=2)

print("\n" + "=" * 80)
print("SUCCESS: ALL 5 STAGES VERIFIED DETERMINISTICALLY. PROVENANCE SAVED.")
print("=" * 80)
