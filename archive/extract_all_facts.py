import os
import json
import time
import torch
import numpy as np

base = r'.'

# ==============================================================================
# STAGE 1
# ==============================================================================
print("Extracting Stage 1...")
import sys
sys.path.insert(0, os.path.join(base, '1'))
import config as cfg1
from model import ChannelPINN as M1

p1 = os.path.join(base, '1', 'outputs', 'channel_pinn_hybrid.pt')
ckpt1 = torch.load(p1, map_location='cpu')
m1 = M1(n_hidden=4, width=80)
m1.load_state_dict(ckpt1)
m1.eval()

# Measure inference time (10,000 points)
x_test = torch.rand(10000, 1)
y_test = torch.rand(10000, 1)
re_test = torch.full((10000, 1), 100.0)
t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = m1(x_test, y_test, re_test)
t_inf_1 = (time.perf_counter() - t0) / 50.0

layers1 = []
for name, p in m1.named_parameters():
    layers1.append({
        'name': name,
        'shape': list(p.shape),
        'frobenius_norm': float(p.norm().item()),
        'mean': float(p.mean().item()),
        'std': float(p.std().item()),
        'sample_values': p.flatten()[:4].tolist()
    })

# Read metrics
with open(os.path.join(base, '1', 'outputs', 'metrics_full_domain.csv'), 'r') as f:
    m1_lines = [l.strip().split(',') for l in f.readlines()]

s1 = {
    'stage': 1,
    'geometry': '2D Plane Channel Flow (Parallel Plates)',
    'dimensions': {'h': float(cfg1.h), 'L': float(cfg1.L), 'Dh': float(cfg1.Dh)},
    'fluid': {'nu': float(cfg1.NU)},
    're_sweep': {'train': cfg1.RE_TRAIN, 'test_interp': [300.0], 'test_extrap': [600.0]},
    'architecture': {
        'type': 'MLP with soft boundary conditions',
        'depth': 4,
        'width': 80,
        'activation': 'tanh',
        'initialization': 'Xavier normal (Glorot & Bengio 2010)',
        'total_parameters': sum(p.numel() for p in m1.parameters()),
        'input_features': 'xn = x/L in [0,1], yn = y/h in [-1,1], ren = (log10(Re) - log_mid)/log_half in [-1,1]',
        'output_envelopes': 'U = u_ref * (1.5*(1 - yn^2) + a_u), V = u_ref * a_v, P = p_scale * a_p',
        'layers': layers1
    },
    'loss_terms_and_weights': {
        'terms': ['data_u', 'data_v', 'data_p', 'pde_continuity', 'pde_mom_x', 'pde_mom_y', 'inlet_u', 'inlet_v', 'wall_u', 'wall_v', 'outlet_p'],
        'weights_method': 'Adaptive Wang et al. (2021) GradNorm annealing',
        'base_fixed_weights': {'data': 10.0, 'pde': 1.0, 'inlet': 5.0, 'wall': 5.0, 'outlet': 1.0}
    },
    'optimizer': {'type': 'Adam followed by L-BFGS', 'adam_lr': 0.001, 'adam_iters': 5000, 'lbfgs_iters': 1000},
    'metrics_full_domain': m1_lines,
    'inference_time_10k_points_sec': t_inf_1
}
with open(os.path.join(base, 'stage_facts_1.json'), 'w') as f:
    json.dump(s1, f, indent=2)
print("Stage 1 saved.")

# ==============================================================================
# STAGE 2
# ==============================================================================
print("Extracting Stage 2...")
for m_name in list(sys.modules.keys()):
    if m_name in ['config', 'model', 'physics', 'adaptive_weights']:
        sys.modules.pop(m_name, None)
sys.path.pop(0)
sys.path.insert(0, os.path.join(base, '2'))
import config as cfg2
from model import PipePINN as M2

p2 = os.path.join(base, '2', 'outputs', 'pipe_pinn_hybrid.pt')
ckpt2 = torch.load(p2, map_location='cpu')
m2 = M2(n_hidden=4, width=80)
m2.load_state_dict(ckpt2)
m2.eval()

t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = m2(x_test, y_test, re_test)
t_inf_2 = (time.perf_counter() - t0) / 50.0

layers2 = []
for name, p in m2.named_parameters():
    layers2.append({
        'name': name,
        'shape': list(p.shape),
        'frobenius_norm': float(p.norm().item()),
        'mean': float(p.mean().item()),
        'std': float(p.std().item()),
        'sample_values': p.flatten()[:4].tolist()
    })

with open(os.path.join(base, '2', 'outputs', 'metrics_global.csv'), 'r') as f:
    m2_lines = [l.strip().split(',') for l in f.readlines()]

s2 = {
    'stage': 2,
    'geometry': 'Axisymmetric Circular Pipe Flow',
    'dimensions': {'R': float(cfg2.R), 'D': float(2*cfg2.R), 'L': float(cfg2.L), 'L_over_D': float(cfg2.L / (2*cfg2.R))},
    'fluid': {'nu': float(cfg2.NU)},
    're_sweep': {'train': cfg2.RE_TRAIN, 'test_interp': [300.0], 'test_extrap': [600.0]},
    'architecture': {
        'type': 'MLP with soft boundary conditions',
        'depth': 4,
        'width': 80,
        'activation': 'tanh',
        'initialization': 'Xavier normal (Glorot & Bengio 2010)',
        'total_parameters': sum(p.numel() for p in m2.parameters()),
        'input_features': 'xn = x/L in [0,1], rn = r/R in [0,1], ren = (log10(Re) - log_mid)/log_half in [-1,1]',
        'output_envelopes': 'U = u_ref * (2.0*(1 - rn^2) + a_u), V = u_ref * a_v, P = p_scale * a_p',
        'layers': layers2
    },
    'loss_terms_and_weights': {
        'terms': ['data_u', 'data_v', 'data_p', 'pde_continuity', 'pde_mom_x', 'pde_mom_r', 'inlet_u', 'inlet_v', 'wall_u', 'wall_v', 'axis_v', 'outlet_p'],
        'weights_method': 'Adaptive Wang et al. (2021) GradNorm annealing',
        'base_fixed_weights': {'data': 10.0, 'pde': 1.0, 'inlet': 5.0, 'wall': 5.0, 'axis': 2.0, 'outlet': 1.0}
    },
    'optimizer': {'type': 'Adam followed by L-BFGS', 'adam_lr': 0.001, 'adam_iters': 5000, 'lbfgs_iters': 1000},
    'metrics_global': m2_lines,
    'inference_time_10k_points_sec': t_inf_2
}
with open(os.path.join(base, 'stage_facts_2.json'), 'w') as f:
    json.dump(s2, f, indent=2)
print("Stage 2 saved.")

# ==============================================================================
# STAGE 3
# ==============================================================================
print("Extracting Stage 3...")
for m_name in list(sys.modules.keys()):
    if m_name in ['config', 'model', 'physics', 'adaptive_weights', 'pinn_model', 'pinn']:
        sys.modules.pop(m_name, None)
sys.path.pop(0)
sys.path.insert(0, os.path.join(base, '3'))
import config as cfg3
from pinn_model import DuctPINN as M3

p3 = os.path.join(base, '3', 'checkpoints', 'duct_pinn.pt')
ckpt3 = torch.load(p3, map_location='cpu')
m3 = M3(cfg3.NET_DEPTH, cfg3.NET_WIDTH)
m3.load_state_dict(ckpt3)
m3.eval()

inp_test3 = torch.rand(10000, 4)
t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = m3(inp_test3[:, 0:1], inp_test3[:, 1:2], inp_test3[:, 2:3], inp_test3[:, 3:4])
t_inf_3 = (time.perf_counter() - t0) / 50.0

layers3 = []
for name, p in m3.named_parameters():
    layers3.append({
        'name': name,
        'shape': list(p.shape),
        'frobenius_norm': float(p.norm().item()),
        'mean': float(p.mean().item()),
        'std': float(p.std().item()),
        'sample_values': p.flatten()[:4].tolist()
    })

# Read summary
with open(os.path.join(base, '3', 'figures', 'global_summary.txt'), 'r') as f:
    m3_summary = [l.strip() for l in f.readlines() if l.strip()]

s3 = {
    'stage': 3,
    'geometry': '3D Rectangular Duct Flow',
    'dimensions': {'H': float(cfg3.H), 'W': float(cfg3.W), 'AR': float(cfg3.AR), 'Dh': float(cfg3.DH), 'L': float(cfg3.L_DOMAIN)},
    'fluid': {'nu': float(cfg3.NU)},
    're_sweep': {'train': cfg3.RE_TRAIN, 'test_interp': cfg3.RE_TEST_INTERP, 'test_extrap': cfg3.RE_TEST_EXTRAP},
    'architecture': {
        'type': 'MLP with soft boundary conditions',
        'depth': cfg3.NET_DEPTH,
        'width': cfg3.NET_WIDTH,
        'activation': 'tanh',
        'initialization': 'Xavier normal (Glorot & Bengio 2010)',
        'total_parameters': sum(p.numel() for p in m3.parameters()),
        'input_features': 'xn = x/L in [0,1], yn = y/H in [0,1], zn = z/W in [0,1], ren = (log10(Re) - log_mid)/log_half in [-1,1]',
        'output_envelopes': 'u = un * U_ref, v = vn * U_ref, w = wn * U_ref, p = pn * p_scale',
        'layers': layers3
    },
    'loss_terms_and_weights': {
        'terms': ['data_u', 'data_v', 'data_w', 'data_p', 'pde', 'inlet', 'wall', 'outlet'],
        'weights_found_by_search': cfg3.LOSS_WEIGHTS,
        'weights_method': 'Adaptive GradNorm (Wang et al. 2021) initialized from random search'
    },
    'optimizer': {'type': 'Adam followed by L-BFGS', 'adam_lr': cfg3.ADAM_LR, 'adam_iters': cfg3.ADAM_ITERS, 'lbfgs_iters': cfg3.LBFGS_ITERS},
    'metrics_global_summary': m3_summary,
    'inference_time_10k_points_sec': t_inf_3
}
with open(os.path.join(base, 'stage_facts_3.json'), 'w') as f:
    json.dump(s3, f, indent=2)
print("Stage 3 saved.")

# ==============================================================================
# STAGE 4
# ==============================================================================
print("Extracting Stage 4...")
for m_name in list(sys.modules.keys()):
    if m_name in ['config', 'model', 'physics', 'adaptive_weights', 'pinn_model', 'pinn']:
        sys.modules.pop(m_name, None)
sys.path.pop(0)
sys.path.insert(0, os.path.join(base, '4'))
import config as cfg4
from model import ChannelPINN as M4

p4 = os.path.join(base, '4', 'outputs', 'final_s0.pt')
ckpt4 = torch.load(p4, map_location='cpu')
arch4 = ckpt4.get('arch', {'hidden': 4, 'width': 64})
m4 = M4(hidden=arch4['hidden'], width=arch4['width'])
m4.load_state_dict(ckpt4['state'])
m4.eval()

t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = m4(x_test, y_test, re_test)
t_inf_4 = (time.perf_counter() - t0) / 50.0

layers4 = []
for name, p in m4.named_parameters():
    layers4.append({
        'name': name,
        'shape': list(p.shape),
        'frobenius_norm': float(p.norm().item()),
        'mean': float(p.mean().item()),
        'std': float(p.std().item()),
        'sample_values': p.flatten()[:4].tolist()
    })

# Read final metrics
with open(os.path.join(base, '4', 'outputs', 'final_metrics.csv'), 'r') as f:
    m4_lines = [l.strip().split(',') for l in f.readlines()]

with open(os.path.join(base, '4', 'outputs', 'best_weights.json'), 'r') as f:
    best_w4 = json.load(f)

s4 = {
    'stage': 4,
    'geometry': 'Turbulent Developing Plane Channel Flow (RANS + Spalart-Allmaras)',
    'dimensions': {'h': 1.0, 'L': float(cfg4.L_OVER_H), 'L_over_h': float(cfg4.L_OVER_H)},
    'fluid': {'nu_formula': '4.0 / Re'},
    're_sweep': {'train': cfg4.RE_TRAIN, 'val': cfg4.RE_VAL, 'test': cfg4.RE_TEST},
    'architecture': {
        'type': 'MLP with hard no-slip boundary constraints and log-wall coordinate transform',
        'depth': arch4['hidden'],
        'width': arch4['width'],
        'activation': 'tanh',
        'initialization': 'Xavier normal (Glorot & Bengio 2010)',
        'total_parameters': sum(p.numel() for p in m4.parameters()),
        'input_features': 's = 2*sqrt(x/L) - 1, eta = 2*y/h - 1, z = 2*log(1 + Re_tau*y)/log(1 + Re_tau) - 1, r = (log10(Re) - log_mid)/log_half',
        'output_envelopes': 'g = tanh(Re_tau*y/10) [exact 0 at wall]; U = g*(1 + a_u); V = 0.05*u_tau,est*g^2*a_v; P_hat = a_p; nu_tilde = nu*Re_tau*g*softplus(a_nt - 3.0)',
        'layers': layers4
    },
    'loss_terms_and_weights': {
        'meta_info_weights': ckpt4['meta']['info']['weights'],
        'best_found_weights': best_w4,
        'pde_system': 'Full Elliptic 2D RANS + Spalart-Allmaras PDE system in divergence form via PyTorch autograd'
    },
    'optimizer': {'type': 'Adam with cosine annealing', 'lr_max': 0.001, 'lr_min': 1e-5, 'epochs': 12000},
    'final_metrics': m4_lines,
    'inference_time_10k_points_sec': t_inf_4
}
with open(os.path.join(base, 'stage_facts_4.json'), 'w') as f:
    json.dump(s4, f, indent=2)
print("Stage 4 saved.")

# ==============================================================================
# STAGE 5
# ==============================================================================
print("Extracting Stage 5...")
for m_name in list(sys.modules.keys()):
    if m_name in ['config', 'model', 'physics', 'adaptive_weights', 'pinn_model', 'pinn']:
        sys.modules.pop(m_name, None)
sys.path.pop(0)
sys.path.insert(0, os.path.join(base, '5'))
import config as cfg5
from pinn import PipePINN as M5

p5 = os.path.join(base, '5', 'outputs', 'models', 'dense', 'final_seed0.pt')
ckpt5 = torch.load(p5, map_location='cpu')
m5 = M5(depth=ckpt5['depth'], width=ckpt5['width'])
m5.load_state_dict(ckpt5['state'])
m5.eval()

t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = m5(x_test.squeeze(1), y_test.squeeze(1), re_test.squeeze(1))
t_inf_5 = (time.perf_counter() - t0) / 50.0

layers5 = []
for name, p in m5.named_parameters():
    layers5.append({
        'name': name,
        'shape': list(p.shape),
        'frobenius_norm': float(p.norm().item()),
        'mean': float(p.mean().item()),
        'std': float(p.std().item()),
        'sample_values': p.flatten()[:4].tolist()
    })

with open(os.path.join(base, '5', 'outputs', 'eval_dense_final_test.json'), 'r') as f:
    m5_test = json.load(f)

with open(os.path.join(base, '5', 'outputs', 'best_weights_dense.json'), 'r') as f:
    best_w5 = json.load(f)

s5 = {
    'stage': 5,
    'geometry': 'Turbulent Developing Axisymmetric Pipe Flow (RANS + Spalart-Allmaras)',
    'dimensions': {'R': 1.0, 'D': 2.0, 'L': float(cfg5.L), 'L_over_R': float(cfg5.L)},
    'fluid': {'nu_formula': '2.0 / Re (= 1 / Re_R)'},
    're_sweep': {'train': cfg5.RE_TRAIN, 'val': cfg5.RE_VAL, 'test': cfg5.RE_TEST},
    'architecture': {
        'type': 'MLP with hard no-slip (wall) and no-penetration (axis) constraints',
        'depth': ckpt5['depth'],
        'width': ckpt5['width'],
        'activation': 'tanh',
        'initialization': 'Xavier normal (Glorot & Bengio 2010)',
        'total_parameters': sum(p.numel() for p in m5.parameters()),
        'input_features': 'f1 = 2*sqrt(x/L) - 1, f2 = 2*(1-r) - 1, f3 = 2*log(1 + Re_tau*(1-r))/log(1 + Re_tau) - 1, f4 = (log10(Re) - log_mid)/log_half',
        'output_envelopes': 'g = tanh(Re_tau*(1-r)/10) [exact 0 at wall]; U = g*(1 + a_u); V = (0.05*u_tau,est)*g^2*r*a_v [exact 0 at wall & axis]; P = a_p; nu_tilde = nu*Re_tau*g*softplus(a_nt - 3.0)',
        'layers': layers5
    },
    'loss_terms_and_weights': {
        'weights_found_by_search': best_w5,
        'pde_system': 'Full Elliptic Axisymmetric RANS + Spalart-Allmaras PDE system in divergence form via PyTorch autograd'
    },
    'optimizer': {'type': 'Adam with cosine annealing', 'lr_max': 0.001, 'lr_min': 1e-5, 'epochs': 12000},
    'final_test_metrics': m5_test,
    'inference_time_10k_points_sec': t_inf_5
}
with open(os.path.join(base, 'stage_facts_5.json'), 'w') as f:
    json.dump(s5, f, indent=2)
print("Stage 5 saved.")
print("ALL STAGE FACTS EXTRACTED SUCCESSFULLY!")
