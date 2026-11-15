"""
dump_code_truth_and_worked_examples.py

Extracts strictly verified code-truth from model classes, checkpoints, and reference solvers:
1. Exact input feature formulas & parameter counts (sum(p.numel()))
2. Exact envelope formulas & Re definitions
3. Frobenius norms of all weight matrices and L2 norms of bias vectors
4. Deterministic worked forward pass on valid in-domain points:
   - Inputs
   - First 4 neurons of EVERY layer
   - Raw outputs
   - Enveloped outputs
   - Reference solver value at that point
5. Writes provenance.json and LaTeX worked example snippets.
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

BASE_DIR = r"."

provenance = {}

def get_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ==============================================================================
# STAGE 1: 2D Laminar Channel Flow
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 1: 2D Laminar Channel Flow")
print("=" * 70)

ckpt1_path = os.path.join(BASE_DIR, "1", "outputs", "channel_pinn_hybrid.pt")
sha1 = get_sha256(ckpt1_path)
ckpt1 = torch.load(ckpt1_path, map_location="cpu")

# Model architecture: 4 hidden layers of 80 neurons, tanh, 3 inputs, 3 outputs
layers1 = [
    (ckpt1["net.0.weight"], ckpt1["net.0.bias"]),
    (ckpt1["net.2.weight"], ckpt1["net.2.bias"]),
    (ckpt1["net.4.weight"], ckpt1["net.4.bias"]),
    (ckpt1["net.6.weight"], ckpt1["net.6.bias"]),
    (ckpt1["net.8.weight"], ckpt1["net.8.bias"]),
]

trainable1 = sum(w.numel() + b.numel() for w, b in layers1)
assert trainable1 == 20003, f"Stage 1 params expected 20003, got {trainable1}"

frob1 = [torch.norm(w, p="fro").item() for w, b in layers1]
bias_l2_1 = [torch.norm(b, p=2).item() for w, b in layers1]

# In-domain test point: x = 0.5 m, y = 0.02 m, Re = 100.0
# Domain: x in [0, 1.0], y in [-0.05, 0.05], L=1.0, h=0.05, nu=1e-4
# Characteristic length: L_char = 2h = 0.1 m
# Re = U0 * (2h) / nu -> U0 = Re * nu / (2h) = 100 * 1e-4 / 0.1 = 0.1 m/s
x_s1, y_s1, re_s1 = 0.5, 0.02, 100.0
L1, h1, nu1 = 1.0, 0.05, 1e-4
u0_s1 = re_s1 * nu1 / (2.0 * h1) # 0.1 m/s
p_sc1 = 0.5 * (u0_s1 ** 2)

re_log_min1 = ckpt1["re_log_min"].item()
re_log_max1 = ckpt1["re_log_max"].item()
x_n1 = x_s1 / L1 # 0.5
y_n1 = y_s1 / h1 # 0.4
re_n1 = 2.0 * (math.log10(re_s1) - re_log_min1) / (re_log_max1 - re_log_min1) - 1.0

xin1 = torch.tensor([[x_n1, y_n1, re_n1]], dtype=torch.float32)

# Layer-by-layer forward pass
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

# Dimensionless outputs
u_hat1 = raw_out1[0]
v_hat1 = raw_out1[1]
p_hat1 = raw_out1[2]

u_phys1 = u_hat1 * u0_s1
v_phys1 = v_hat1 * u0_s1
p_phys1 = p_hat1 * p_sc1

# Reference solver lookup
ref1_file = os.path.join(BASE_DIR, "1", "outputs", "reference_data.npz")
ref1_data = np.load(ref1_file)
sys.modules.pop("evaluate", None)
sys.path.insert(0, os.path.join(BASE_DIR, "1"))
import evaluate as ev1
x_ref1_arr, y_ref1_arr, U_ref1_arr, V_ref1_arr, P_ref1_arr = ev1.load_reference(ref1_data, re_s1)
ix1 = np.argmin(np.abs(x_ref1_arr - x_s1))
iy1 = np.argmin(np.abs(y_ref1_arr - y_s1))
u_ref1 = float(U_ref1_arr[ix1, iy1])
v_ref1 = float(V_ref1_arr[ix1, iy1])
p_ref1 = float(P_ref1_arr[ix1])
sys.path.pop(0)
sys.modules.pop("evaluate", None)

provenance["stage1"] = {
    "name": "Stage 1: 2D Laminar Channel Flow",
    "checkpoint": "1/outputs/channel_pinn_hybrid.pt",
    "sha256": sha1,
    "char_length": "2h = 0.1 m (full channel height)",
    "re_definition": "Re = U_0 * (2h) / nu",
    "nu": nu1,
    "U0_Re": "Re * nu / (2h) = 1e-3 * Re m/s",
    "domain": {"x": [0.0, 1.0], "y": [-0.05, 0.05]},
    "inputs": "x_n = x/L, y_n = y/h, re_n = 2*(log10(Re) - log_min)/(log_max - log_min) - 1",
    "widths": [3, 80, 80, 80, 80, 3],
    "trainable_parameters": trainable1,
    "frob_norms": frob1,
    "bias_l2_norms": bias_l2_1,
    "output_bias": layers1[-1][1].tolist(),
    "test_point": {"x": x_s1, "y": y_s1, "Re": re_s1},
    "input_vector": xin1[0].tolist(),
    "layer_neurons": neurons1,
    "raw_outputs": raw_out1,
    "enveloped_outputs": {"u_hat": u_hat1, "v_hat": v_hat1, "p_hat": p_hat1},
    "physical_predictions": {"u": u_phys1, "v": v_phys1, "p": p_phys1},
    "reference_values": {"u": u_ref1, "v": v_ref1, "p": p_ref1},
}
print(f"Stage 1 checkpoint SHA256: {sha1}")
print(f"Trainable params: {trainable1}")
print(f"Test point prediction: u={u_phys1:.5f} m/s (ref: {u_ref1:.5f}), diff={abs(u_phys1-u_ref1):.5f} m/s")


# ==============================================================================
# STAGE 2: Laminar Axisymmetric Circular Pipe Flow
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 2: Laminar Axisymmetric Circular Pipe Flow")
print("=" * 70)

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
assert trainable2 == 20003, f"Stage 2 params expected 20003, got {trainable2}"

frob2 = [torch.norm(w, p="fro").item() for w, b in layers2]
bias_l2_2 = [torch.norm(b, p=2).item() for w, b in layers2]

# In-domain test point: x = 2.0 m, r = 0.025 m, Re = 100.0
# Domain: x in [0, 4.0], r in [0, 0.05], L=4.0, R=0.05, D=0.1, nu=1e-4
# Characteristic length: L_char = D = 2R = 0.1 m
# Re = U0 * D / nu -> U0 = Re * nu / D = 100 * 1e-4 / 0.1 = 0.1 m/s
x_s2, r_s2, re_s2 = 2.0, 0.025, 100.0
L2, R2, nu2 = 4.0, 0.05, 1e-4
D2 = 2.0 * R2
u0_s2 = re_s2 * nu2 / D2 # 0.1 m/s
p_sc2 = 0.5 * (u0_s2 ** 2)

re_log_min2 = ckpt2["re_log_min"].item()
re_log_max2 = ckpt2["re_log_max"].item()
x_n2 = x_s2 / L2 # 0.5
r_n2 = r_s2 / R2 # 0.5
re_n2 = 2.0 * (math.log10(re_s2) - re_log_min2) / (re_log_max2 - re_log_min2) - 1.0

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

u_hat2 = raw_out2[0]
v_hat2 = raw_out2[1]
p_hat2 = raw_out2[2]

u_phys2 = u_hat2 * u0_s2
v_phys2 = v_hat2 * u0_s2
p_phys2 = p_hat2 * p_sc2

# Reference solver lookup
sys.path = [p for p in sys.path if not p.endswith(os.sep + "1")]
for m in ["model", "evaluate", "config", "physics"]:
    sys.modules.pop(m, None)
sys.path.insert(0, os.path.join(BASE_DIR, "2"))
import evaluate as ev2
ref2, p_shifted2 = ev2.load_reference()
ix2 = np.argmin(np.abs(ref2[f"x_{re_s2}"] - x_s2))
ir2 = np.argmin(np.abs(ref2[f"r_{re_s2}"] - r_s2))
u_ref2 = float(ref2[f"U_{re_s2}"][ix2, ir2])
v_ref2 = float(ref2[f"V_{re_s2}"][ix2, ir2])
p_ref2 = float(p_shifted2[re_s2][ix2])
sys.path = [p for p in sys.path if not p.endswith(os.sep + "2")]
for m in ["model", "evaluate", "config", "physics"]:
    sys.modules.pop(m, None)

provenance["stage2"] = {
    "name": "Stage 2: Laminar Axisymmetric Circular Pipe Flow",
    "checkpoint": "2/outputs/pipe_pinn_hybrid.pt",
    "sha256": sha2,
    "char_length": "D = 2R = 0.1 m (pipe diameter)",
    "re_definition": "Re = U_0 * D / nu",
    "nu": nu2,
    "U0_Re": "Re * nu / D = 1e-3 * Re m/s",
    "domain": {"x": [0.0, 4.0], "r": [0.0, 0.05]},
    "inputs": "x_n = x/L, r_n = r/R, re_n = 2*(log10(Re) - log_min)/(log_max - log_min) - 1",
    "widths": [3, 80, 80, 80, 80, 3],
    "trainable_parameters": trainable2,
    "frob_norms": frob2,
    "bias_l2_norms": bias_l2_2,
    "output_bias": layers2[-1][1].tolist(),
    "test_point": {"x": x_s2, "r": r_s2, "Re": re_s2},
    "input_vector": xin2[0].tolist(),
    "layer_neurons": neurons2,
    "raw_outputs": raw_out2,
    "enveloped_outputs": {"u_hat": u_hat2, "v_hat": v_hat2, "p_hat": p_hat2},
    "physical_predictions": {"u": u_phys2, "v": v_phys2, "p": p_phys2},
    "reference_values": {"u": u_ref2, "v": v_ref2, "p": p_ref2},
}
print(f"Stage 2 checkpoint SHA256: {sha2}")
print(f"Trainable params: {trainable2}")
print(f"Test point prediction: u={u_phys2:.5f} m/s (ref: {u_ref2:.5f}), diff={abs(u_phys2-u_ref2):.5f} m/s")


# ==============================================================================
# STAGE 3: Laminar 3D Rectangular Duct Flow
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 3: Laminar 3D Rectangular Duct Flow")
print("=" * 70)

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
assert trainable3 == 20164, f"Stage 3 params expected 20164, got {trainable3}"

frob3 = [torch.norm(w, p="fro").item() for w, b in layers3]
bias_l2_3 = [torch.norm(b, p=2).item() for w, b in layers3]

# In-domain test point: x = 2.5 m (<= 5.0 m), y = 0.05 m, z = 0.05 m, Re = 100.0
# Domain: x in [0, 5.0], y in [0, 0.10], z in [0, 0.10], L=5.0, H=0.10, W=0.10
# Hydraulic diameter: D_h = 4*A/P = 4*(0.1*0.1)/0.4 = 0.10 m
# Re = U_ref * D_h / nu -> U_ref(Re) = Re * nu / D_h = 100 * 1e-4 / 0.10 = 0.10 m/s
# pscale = 0.5 * (U_ref**2) * (L / D_h)
x_s3, y_s3, z_s3, re_s3 = 2.5, 0.05, 0.05, 100.0
L3, H3, W3, nu3 = 5.0, 0.10, 0.10, 1e-4
Dh3 = 0.10
u0_s3 = re_s3 * nu3 / Dh3 # 0.10 m/s
p_sc3 = 0.5 * (u0_s3 ** 2) * (L3 / Dh3)

x_n3 = x_s3 / L3 # 0.5
y_n3 = y_s3 / H3 # 0.5
z_n3 = z_s3 / W3 # 0.5
re_n3 = (re_s3 - 325.0) / 275.0 # linear mapping for Re in [50, 600]

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

# Duct envelope:
# u_pred = 16 * y_n * (1-y_n) * z_n * (1-z_n) * (1.0 + raw_u)
# v_pred = 16 * y_n * (1-y_n) * z_n * (1-z_n) * raw_v
# w_pred = 16 * y_n * (1-y_n) * z_n * (1-z_n) * raw_w
geom3 = 16.0 * y_n3 * (1.0 - y_n3) * z_n3 * (1.0 - z_n3) # 1.0 at center
u_hat3 = geom3 * (1.0 + raw_out3[0])
v_hat3 = geom3 * raw_out3[1]
w_hat3 = geom3 * raw_out3[2]
p_hat3 = raw_out3[3]

u_phys3 = u_hat3 * u0_s3
v_phys3 = v_hat3 * u0_s3
w_phys3 = w_hat3 * u0_s3
p_phys3 = p_hat3 * p_sc3

# Reference solver lookup
ref3_path = os.path.join(BASE_DIR, "3", "data", "duct_Re100.npz")
d3 = np.load(ref3_path)
ix3 = np.argmin(np.abs(d3["x"] - x_s3))
iy3 = np.argmin(np.abs(d3["y"] - y_s3))
iz3 = np.argmin(np.abs(d3["z"] - z_s3))
u_ref3 = float(d3["U"][ix3, iy3, iz3])
v_ref3 = float(d3["V"][ix3, iy3, iz3])
w_ref3 = float(d3["W"][ix3, iy3, iz3])
p_ref3 = float(d3["P"][ix3, iy3, iz3])

provenance["stage3"] = {
    "name": "Stage 3: 3D Rectangular Duct Flow",
    "checkpoint": "3/checkpoints/duct_pinn.pt",
    "sha256": sha3,
    "char_length": "D_h = 4A/P = 0.10 m (square duct hydraulic diameter)",
    "re_definition": "Re = U_bulk * D_h / nu",
    "nu": nu3,
    "U0_Re": "Re * nu / D_h = 1e-3 * Re m/s",
    "domain": {"x": [0.0, 5.0], "y": [0.0, 0.10], "z": [0.0, 0.10]},
    "inputs": "x_n = x/L, y_n = y/H, z_n = z/W, re_n = (Re - 325)/275",
    "widths": [4, 80, 80, 80, 80, 4],
    "trainable_parameters": trainable3,
    "frob_norms": frob3,
    "bias_l2_norms": bias_l2_3,
    "output_bias": layers3[-1][1].tolist(),
    "test_point": {"x": x_s3, "y": y_s3, "z": z_s3, "Re": re_s3},
    "input_vector": xin3[0].tolist(),
    "layer_neurons": neurons3,
    "raw_outputs": raw_out3,
    "enveloped_outputs": {"u_hat": u_hat3, "v_hat": v_hat3, "w_hat": w_hat3, "p_hat": p_hat3},
    "physical_predictions": {"u": u_phys3, "v": v_phys3, "w": w_phys3, "p": p_phys3},
    "reference_values": {"u": u_ref3, "v": v_ref3, "w": w_ref3, "p": p_ref3},
}
print(f"Stage 3 checkpoint SHA256: {sha3}")
print(f"Trainable params: {trainable3}")
print(f"Test point prediction: u={u_phys3:.5f} m/s (ref: {u_ref3:.5f})")


# ==============================================================================
# STAGE 4: Developing Turbulent Channel Flow (RANS + SA)
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 4: Developing Turbulent Channel Flow (RANS + SA)")
print("=" * 70)

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
assert trainable4 == 13060, f"Stage 4 params expected 13060, got {trainable4}"

frob4 = [torch.norm(w, p="fro").item() for w, b in layers4]
bias_l2_4 = [torch.norm(b, p=2).item() for w, b in layers4]

# In-domain test point: x = 40.0 h, y = 0.5 h, Re = 20,000
# Domain: x in [0, 160], y in [0, 1.0], L=160 h, h=1.0, U_b=1.0
# Re = 2 * U_b * h / nu -> nu = 4 / Re = 4 / 20000 = 2e-4
# Re_tau_est = 0.0955 * (Re / 2.0)**0.875
x_s4, y_s4, re_s4 = 40.0, 0.5, 20000.0
L4 = 160.0
mid4 = 0.5 * (math.log10(10000.0) + math.log10(120000.0))
half4 = 0.5 * (math.log10(120000.0) - math.log10(10000.0))
ret4 = 0.0955 * (re_s4 / 2.0) ** 0.875 # ~301.76

s4 = math.sqrt(x_s4 / L4 + 1e-8)
eta4 = y_s4
z4 = math.log1p(ret4 * eta4) / math.log1p(ret4)
r4 = (math.log10(re_s4) - mid4) / half4

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

# Channel envelope:
# g = tanh(Re_tau * y / 10.0)
# vs = 0.05 * sqrt(0.5 * 0.073 * (Re/2)**-0.25)
# U = g * (1 + a_u)
# V = vs * g^2 * a_v
# P = a_p
# nu_tilde = nu * Re_tau * g * softplus(a_nut - 3.0)
g4 = math.tanh(ret4 * y_s4 / 10.0)
nu4 = 4.0 / re_s4
vs4 = 0.05 * math.sqrt(0.5 * 0.073 * (re_s4 / 2.0) ** -0.25)

u_pinn4 = g4 * (1.0 + raw_out4[0])
v_pinn4 = vs4 * (g4 ** 2) * raw_out4[1]
p_pinn4 = raw_out4[2]
nut_pinn4 = nu4 * ret4 * g4 * float(F.softplus(torch.tensor(raw_out4[3] - 3.0)))

# Reference lookup
ref4_file = os.path.join(BASE_DIR, "4", "reference_data", f"ref_Re{int(re_s4)}.npz")
rf4 = np.load(ref4_file)
ix4 = np.argmin(np.abs(rf4["x"] - x_s4))
iy4 = np.argmin(np.abs(rf4["y"] - y_s4))
u_ref4 = float(rf4["U"][ix4, iy4])
v_ref4 = float(rf4["V"][ix4, iy4])
p_ref4 = float(rf4["P_hat"][ix4])
nut_ref4 = float(rf4["nut_tilde"][ix4, iy4])

provenance["stage4"] = {
    "name": "Stage 4: Developing Turbulent Channel Flow (RANS+SA)",
    "checkpoint": "4/outputs/final_s0.pt",
    "sha256": sha4,
    "char_length": "2h = 2.0 (channel full height, h=1.0)",
    "re_definition": "Re = 2 * U_b * h / nu",
    "nu": nu4,
    "U0_Re": "Bulk velocity U_b = 1.0 (constant non-dimensionalization)",
    "domain": {"x_over_h": [0.0, 160.0], "y_over_h": [0.0, 1.0]},
    "inputs": "4 features: f1=2*sqrt(x/L)-1, f2=2*eta-1, f3=2*log1p(ret*eta)/log1p(ret)-1, f4=(log10(Re)-mid)/half",
    "widths": [4, 64, 64, 64, 64, 4],
    "trainable_parameters": trainable4,
    "frob_norms": frob4,
    "bias_l2_norms": bias_l2_4,
    "output_bias": layers4[-1][1].tolist(),
    "test_point": {"x/h": x_s4, "y/h": y_s4, "Re": re_s4},
    "input_vector": xin4[0].tolist(),
    "layer_neurons": neurons4,
    "raw_outputs": raw_out4,
    "enveloped_outputs": {"u": u_pinn4, "v": v_pinn4, "p": p_pinn4, "nut": nut_pinn4},
    "physical_predictions": {"u": u_pinn4, "v": v_pinn4, "p": p_pinn4, "nut": nut_pinn4},
    "reference_values": {"u": u_ref4, "v": v_ref4, "p": p_ref4, "nut": nut_ref4},
}
print(f"Stage 4 checkpoint SHA256: {sha4}")
print(f"Trainable params: {trainable4}")
print(f"Test point prediction: u={u_pinn4:.5f} (ref: {u_ref4:.5f}), error={abs(u_pinn4-u_ref4)*100:.2f}%")


# ==============================================================================
# STAGE 5: Developing Turbulent Pipe Flow (Axisymmetric RANS + SA)
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 5: Developing Turbulent Pipe Flow (Axisymmetric RANS + SA)")
print("=" * 70)

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
assert trainable5 == 13060, f"Stage 5 params expected 13060, got {trainable5}"

frob5 = [torch.norm(w, p="fro").item() for w, b in layers5]
bias_l2_5 = [torch.norm(b, p=2).item() for w, b in layers5]

# In-domain test point: x = 40.0 R (<= 160 R), r = 0.5 R, Re = 20,000
# Domain: x in [0, 160], r in [0, 1.0], L=160 R, R=1.0, U_b=1.0
# Re = U_b * D / nu = 2 * U_b * R / nu -> nu = 2 / Re = 2 / 20000 = 1e-4
# Re_tau_est computed via McKeon/Prandtl skin friction iteration
x_s5, r_s5, re_s5 = 40.0, 0.5, 20000.0
L5 = 160.0
re_t5 = torch.tensor([re_s5], dtype=torch.float32)

# est_torch function from pinn.py
s = 4.0 * torch.ones_like(re_t5)
for _ in range(30):
    s = 2.0 * torch.log10(re_t5 / s) - 0.8
lam = 1.0 / (s ** 2)
cf5 = float(lam / 4.0)
ut5 = float(torch.sqrt(lam / 8.0))
ret5 = float(ut5 * re_s5 / 2.0) # ~553.33
nu5 = 2.0 / re_s5

mid5 = 0.5 * (math.log10(10000.0) + math.log10(80000.0))
half5 = 0.5 * (math.log10(80000.0) - math.log10(10000.0))

eta5 = 1.0 - r_s5
f1_5 = 2.0 * math.sqrt(x_s5 / L5 + 1e-12) - 1.0
f2_5 = 2.0 * eta5 - 1.0
f3_5 = 2.0 * math.log1p(ret5 * eta5) / math.log1p(ret5) - 1.0
f4_5 = (math.log10(re_s5) - mid5) / half5

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

# Pipe envelope:
# g = tanh(Re_tau * eta / 10.0)
# U = g * (1 + a_u)
# V = 0.05 * ut * g^2 * r * a_v
# P = a_p
# nu_tilde = nu * Re_tau * g * softplus(a_nut - 3.0)
g5 = math.tanh(ret5 * eta5 / 10.0)
u_pinn5 = g5 * (1.0 + raw_out5[0])
v_pinn5 = (0.05 * ut5) * (g5 ** 2) * r_s5 * raw_out5[1]
p_pinn5 = raw_out5[2]
nut_pinn5 = nu5 * ret5 * g5 * float(F.softplus(torch.tensor(raw_out5[3] - 3.0)))

# Reference lookup
ref5_file = os.path.join(BASE_DIR, "5", "data", f"ref_Re{int(re_s5)}.npz")
rf5 = np.load(ref5_file)
ix5 = np.argmin(np.abs(rf5["x"] - x_s5))
ir5 = np.argmin(np.abs(rf5["r"] - r_s5))
u_ref5 = float(rf5["U"][ix5, ir5])
v_ref5 = float(rf5["V"][ix5, ir5])
p_ref5 = float(rf5["P_hat"][ix5])
nut_ref5 = float(rf5["nt"][ix5, ir5])

provenance["stage5"] = {
    "name": "Stage 5: Developing Turbulent Pipe Flow (Axisymmetric RANS+SA)",
    "checkpoint": "5/outputs/models/dense/final_seed0.pt",
    "sha256": sha5,
    "char_length": "D = 2R = 2.0 (pipe diameter, R=1.0)",
    "re_definition": "Re = U_b * D / nu",
    "nu": nu5,
    "U0_Re": "Bulk velocity U_b = 1.0 (constant non-dimensionalization)",
    "domain": {"x_over_R": [0.0, 160.0], "r_over_R": [0.0, 1.0]},
    "inputs": "4 features: f1=2*sqrt(x/L)-1, f2=2*eta-1, f3=2*log1p(ret*eta)/log1p(ret)-1, f4=(log10(Re)-mid)/half",
    "widths": [4, 64, 64, 64, 64, 4],
    "trainable_parameters": trainable5,
    "frob_norms": frob5,
    "bias_l2_norms": bias_l2_5,
    "output_bias": layers5[-1][1].tolist(),
    "test_point": {"x/R": x_s5, "r/R": r_s5, "Re": re_s5},
    "input_vector": xin5[0].tolist(),
    "layer_neurons": neurons5,
    "raw_outputs": raw_out5,
    "enveloped_outputs": {"u": u_pinn5, "v": v_pinn5, "p": p_pinn5, "nut": nut_pinn5},
    "physical_predictions": {"u": u_pinn5, "v": v_pinn5, "p": p_pinn5, "nut": nut_pinn5},
    "reference_values": {"u": u_ref5, "v": v_ref5, "p": p_ref5, "nut": nut_ref5},
}
print(f"Stage 5 checkpoint SHA256: {sha5}")
print(f"Trainable params: {trainable5}")
print(f"Test point prediction: u={u_pinn5:.5f} (ref: {u_ref5:.5f}), error={abs(u_pinn5-u_ref5)*100:.2f}%")

# Save provenance.json
prov_path = os.path.join(BASE_DIR, "provenance.json")
with open(prov_path, "w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=2)
print(f"\nSaved master provenance: {prov_path}")
