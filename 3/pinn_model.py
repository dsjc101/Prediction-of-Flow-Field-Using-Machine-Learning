"""
pinn_model.py
=============
Network architecture and I/O normalization for the rectangular-duct PINN.

WHY THIS IS GENUINELY 3-D (CONTRAST WITH THE PIPE'S AXISYMMETRIC REDUCTION)
-----------------------------------------------------------------------------
The pipe project's Sec 2 justified reducing (x,y,z) -> (x,r) because a
swirl-free circular pipe's exact solution is provably independent of the
azimuthal angle theta -- a rotation-invariance argument that has NO
analogue for a rectangular cross-section: rotating a rectangle by an
arbitrary angle does not map it to itself, so there is no continuous
symmetry to quotient out. The duct's exact solution genuinely depends on
BOTH cross-sectional coordinates (y,z) -- see the Fourier series in
analytical_duct.py, which is a true function of both. Consequently the
network here takes the full (x,y,z,Re) input; there is no dimensionality
reduction analogous to model.to_cartesian() in the pipe project.

What a rectangular duct DOES have -- and what the pipe/channel didn't need
in the same way -- is two DISCRETE reflection symmetries (about y=H/2 and
about z=W/2), inherited from a centered uniform inlet and geometrically
symmetric walls. This is an elementary consequence of the BCs, not a
citable result on its own; it is used here purely as a SAMPLING
efficiency device (mirror collocation points into all 4 quadrants so a
network trained on samples drawn from one quadrant sees data consistent
with the other three), not a reduction in the input dimensionality the
network must represent.

NORMALIZATION (same conditioning-device philosophy as pipe/channel
projects, deriving all scales from the Re definition itself so nothing
here leaks a solved flow field into the inputs):

  Inputs:  (x/L, y/H, z/W, rescaled log10(Re) in [-1,1])
  Outputs (normalized): (u/U_ref, v/U_ref, w/U_ref, p/p_scale)
    U_ref(Re) = Re*nu/Dh         (same functional form as pipe's U_ref)
    p_scale(Re) = 0.5*U_ref(Re)^2 * (L_DOMAIN/Dh)   -- CHANGED (bug fix):
    the dynamic-pressure scale q=0.5*U_ref^2 alone is a LOCAL/pointwise
    pressure scale; the actual pressure DROP over this project's long
    marching domain (L_DOMAIN/Dh ~ 72) is the Darcy-Weisbach friction-
    loss scale q*(L/Dh) (White, F.M. (2006), "Viscous Fluid Flow," 3rd
    ed., McGraw-Hill, Ch. 6 -- already cited elsewhere in this project),
    which is ~70x larger than q itself. Normalizing p by q alone (the
    original, pre-v/w/p-training convention, carried over from when p
    only appeared in the trivial outlet-BC term) put every p DATA target
    at O(70-90) in normalized units -- unreachable for a tanh network and
    the direct cause of the data_p loss blowing up (and, via
    adaptive_weights.py's shared normalization, destabilizing every other
    loss weight together with it). p_scale now includes the L/Dh factor
    so normalized p targets are O(1), matching u, v, w.
"""

import numpy as np
import torch
import torch.nn as nn

import config


def re_to_normalized(Re, Re_min=50, Re_max=600):
    """log10(Re) rescaled to [-1,1] -- identical convention to pipe project."""
    lo, hi = np.log10(Re_min), np.log10(Re_max)
    return 2.0 * (np.log10(Re) - lo) / (hi - lo) - 1.0


class DuctPINN(nn.Module):
    def __init__(self, depth=config.NET_DEPTH, width=config.NET_WIDTH):
        super().__init__()
        layers = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 4)]  # u, v, w, p  (normalized)
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, z, Re_norm):
        """
        x,y,z,Re_norm: (N,1) tensors, ALREADY normalized
        (x/L, y/H, z/W, log10(Re)-rescaled respectively).
        Returns (u,v,w,p) normalized outputs, each (N,1).
        """
        inp = torch.cat([x, y, z, Re_norm], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    def predict_physical(self, x_phys, y_phys, z_phys, Re, device="cpu"):
        """
        Convenience wrapper: physical (x,y,z,Re) arrays in -> physical
        (u,v,w,p) arrays out. Re may be a scalar (broadcast) or array.
        """
        x_phys = np.atleast_1d(x_phys).astype(np.float64)
        y_phys = np.atleast_1d(y_phys).astype(np.float64)
        z_phys = np.atleast_1d(z_phys).astype(np.float64)
        Re_arr = np.broadcast_to(Re, x_phys.shape).astype(np.float64)

        U_ref = Re_arr * config.NU / config.DH
        p_scale = 0.5 * U_ref ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see module docstring

        xn = torch.tensor((x_phys / config.L_DOMAIN).reshape(-1, 1), dtype=torch.float32, device=device)
        yn = torch.tensor((y_phys / config.H).reshape(-1, 1), dtype=torch.float32, device=device)
        zn = torch.tensor((z_phys / config.W).reshape(-1, 1), dtype=torch.float32, device=device)
        ren = torch.tensor(re_to_normalized(Re_arr).reshape(-1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            u, v, w, p = self.forward(xn, yn, zn, ren)
        u = u.cpu().numpy().reshape(-1) * U_ref
        v = v.cpu().numpy().reshape(-1) * U_ref
        w = w.cpu().numpy().reshape(-1) * U_ref
        p = p.cpu().numpy().reshape(-1) * p_scale
        return u, v, w, p


def mirror_quadrant_points(y, z, H, W):
    """
    Sampling-efficiency helper (see module docstring): given collocation
    points drawn in one quadrant [0,H/2]x[0,W/2], return all 4
    reflections so the training set respects the two symmetry planes by
    construction. NOT used to change the network's input dimensionality --
    only changes how collocation points are generated in train_hybrid.py.
    """
    y2, z2 = H - y, W - z
    Y = np.concatenate([y, y, y2, y2])
    Z = np.concatenate([z, z2, z, z2])
    return Y, Z