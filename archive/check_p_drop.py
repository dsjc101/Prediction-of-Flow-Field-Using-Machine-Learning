import os, sys, torch, numpy as np
ROOT = r"."
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "3"))
import config as cfg3
from pinn_model import DuctPINN

model = DuctPINN()
sd = torch.load('3/checkpoints/duct_pinn.pt', map_location='cpu')
model.load_state_dict(sd)
model.eval()

for Re in [50, 100, 200, 300, 500, 600]:
    d_ref = np.load(f'3/data/duct_Re{Re}.npz')
    xr = d_ref['x']
    ny = d_ref['U'].shape[1]
    nz = d_ref['U'].shape[2]
    p_ref_c = d_ref['P'][:, ny//2, nz//2]
    mask = (xr >= 0.05) & (xr <= cfg3.L_DOMAIN)
    dp_ref = float(p_ref_c[mask][0] - p_ref_c[mask][-1])
    
    x_pts = np.array([0.05, cfg3.L_DOMAIN], dtype=np.float32)
    y_pts = np.full_like(x_pts, 0.5 * cfg3.H)
    z_pts = np.full_like(x_pts, 0.5 * cfg3.W)
    _, _, _, p_pred = model.predict_physical(x_pts, y_pts, z_pts, Re, device='cpu')
    dp_pinn = float(p_pred[0] - p_pred[1])
    ratio = dp_ref / dp_pinn
    print(f'Re={Re}: dp_ref={dp_ref:.5f}, dp_pinn={dp_pinn:.5f}, ratio={ratio:.2f}x')
