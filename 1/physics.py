"""
physics.py -- steady, incompressible 2D Navier-Stokes PDE residuals for the
hybrid PINN's physics loss.

NOTE ON PROVENANCE: like model.py, this file is referenced by
train_hybrid.py / validate_hybrid.py but was missing from the uploaded
hybrid-report file set. It is reconstructed from the hybrid report's
Sec. 4.1 ("the PINN's own physics loss uses the FULL steady Navier-Stokes
equations... the elliptic d^2u/dx^2 term is retained, computed exactly via
automatic differentiation") and from the physics-only report's own
`ns_residuals` (pinn_model.py), which implements exactly that equation set
and is reused here essentially unchanged, adapted to call the merged
model.py's forward() instead.

Governing equations (steady, incompressible, 2D, kinematic pressure):
    continuity:  u_x + v_y = 0
    x-momentum:  u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
    y-momentum:  u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
following Raissi, Perdikaris & Karniadakis (2019), J. Comput. Phys. 378,
686-707.

Note this deliberately does NOT use the parabolized (boundary-layer)
approximation that entrance_solver.py uses to generate the reference/data-
loss target (Bodoia & Osterle, 1961) -- the hybrid report is explicit that
the PDE-residual loss and the data loss are built from two different
governing-equation approximations, and that the elliptic form is the
correct one to enforce as physics (Sec. 4.1; see also known-limitations
discussion of where the two disagree, e.g. Wendel 1961, at low Re).
"""
import torch
import config as cfg
from model import u_ref


def ns_residuals(model, x, y, Re, nu=cfg.NU):
    """Physical-unit steady NS residuals (continuity, x-mom, y-mom) at the
    given (x, y, Re) points, using the model's physical forward()."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    u, v, p = model(x, y, Re)

    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=ones, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones, create_graph=True)[0]

    r_cont = u_x + v_y
    r_mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return r_cont, r_mom_x, r_mom_y


def ns_residuals_normalized(model, x, y, Re, nu=cfg.NU):
    """Same residuals, nondimensionalized by the Re-derived velocity scale
    U_ref(Re) and the channel half-height h, so residual-loss magnitude is
    comparable across the Re sweep (same float32-conditioning rationale as
    model.py's output rescaling -- physics-only report Sec. 7.2). This is
    the fixed, closed-form alternative to the adaptive gradient-norm
    balancing implemented in adaptive_weights.py."""
    r_cont, r_mx, r_my = ns_residuals(model, x, y, Re, nu=nu)
    Uref = u_ref(Re)
    cont_scale = Uref / cfg.h
    mom_scale = Uref ** 2 / cfg.h
    return r_cont / cont_scale, r_mx / mom_scale, r_my / mom_scale