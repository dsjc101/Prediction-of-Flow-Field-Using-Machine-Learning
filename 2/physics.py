"""
physics.py -- steady, incompressible, axisymmetric (no-swirl) Navier-
Stokes PDE residuals for the pipe PINN's physics loss. Circular-pipe
counterpart of the 2-D project's physics.py.

GOVERNING EQUATIONS (full elliptic form, steady, incompressible,
axisymmetric, no swirl, kinematic pressure, cylindrical coordinates
(x, r)):

    continuity:    u_x + v_r + v/r = 0
    x-momentum:    u*u_x + v*u_r + p_x - nu*(u_xx + u_rr + u_r/r) = 0
    r-momentum:    u*v_x + v*v_r + p_r - nu*(v_xx + v_rr + v_r/r - v/r^2) = 0

where u = axial velocity, v = radial velocity. This is the standard
axisymmetric reduction of the incompressible Navier-Stokes equations
(theta-derivatives dropped, u_theta = 0) -- see e.g. White, F.M. (2006),
Viscous Fluid Flow, 3rd ed., McGraw-Hill, App. B (Navier-Stokes equations
in cylindrical coordinates), or Schlichting & Gersten (2017), Boundary-
Layer Theory, 9th ed., Ch. 3. The extra v/r (continuity) and u_r/r,
v_r/r - v/r^2 (momentum) terms relative to the 2-D Cartesian project's
physics.py are exactly the curvature terms introduced by the cylindrical
metric; they vanish identically in the y -> infinity (flat) limit, which
is the correct consistency check between the two projects' governing
equations. The v/r^2 term in r-momentum is the standard "hoop stress"
term from the cylindrical Laplacian acting on a vector field (see e.g.
the derivation of the vector Laplacian in cylindrical coordinates in
Schlichting & Gersten (2017), Ch. 3, or any continuum-mechanics text
covering curvilinear tensor calculus).

Following the SAME design decision as the 2-D project's physics.py: the
PDE-residual loss uses this FULL elliptic form (not the parabolized form
entrance_solver.py uses to generate reference/data-loss targets) -- the
elliptic form is the physically correct one to enforce as a regularizer,
and Raissi, Perdikaris & Karniadakis (2019), J. Comput. Phys. 378,
686-707, is again the framework this hybrid data+physics loss follows
(their original PINN formulation, here specialized to axisymmetric flow
in the same spirit as the axisymmetric-RANS-PINN of Acta Geotechnica
(2026), https://doi.org/10.1007/s11440-026-02979-7, which likewise
enforces the full axisymmetric Navier-Stokes/RANS residuals via autodiff
rather than the parabolized/boundary-layer form used only to generate its
training data).

AXIS SINGULARITY (r=0): all three residuals above contain explicit 1/r or
1/r^2 terms, which are undefined exactly at r=0 even though the true
solution is smooth there (v=0 and du/dr=0 on the axis by symmetry cancel
the singularity in the exact solution, but not necessarily in a partially
trained network's residual). This project's design choice, following
standard axisymmetric-PINN practice (see e.g. the discussion of avoiding
exact-axis collocation points in axisymmetric PINN formulations): the PDE
RESIDUAL is only evaluated at r > 0 (interior collocation points are
sampled from r in (r_eps, R), never exactly r=0 -- see train_hybrid.py's
`sample_interior`); the axis condition itself (v=0, symmetry) is enforced
separately as its own boundary-condition loss term (`loss_axis` in
train_hybrid.py), analogous to how the wall's no-slip condition is a
boundary loss rather than something extracted from the interior PDE
residual.
"""
import torch
import config as cfg
from model import u_ref


def ns_residuals(model, x, r, Re, nu=cfg.NU):
    """Physical-unit steady axisymmetric NS residuals (continuity,
    x-momentum, r-momentum) at the given (x, r, Re) points, using the
    model's physical forward(). Caller must ensure r > 0 (see module
    docstring, 'AXIS SINGULARITY')."""
    x.requires_grad_(True)
    r.requires_grad_(True)
    u, v, p = model(x, r, Re)

    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_r = torch.autograd.grad(u, r, grad_outputs=ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True)[0]
    v_r = torch.autograd.grad(v, r, grad_outputs=ones, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=ones, create_graph=True)[0]
    p_r = torch.autograd.grad(p, r, grad_outputs=ones, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, r, grad_outputs=ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones, create_graph=True)[0]
    v_rr = torch.autograd.grad(v_r, r, grad_outputs=ones, create_graph=True)[0]

    r_cont = u_x + v_r + v / r
    r_mom_x = u * u_x + v * u_r + p_x - nu * (u_xx + u_rr + u_r / r)
    r_mom_r = u * v_x + v * v_r + p_r - nu * (v_xx + v_rr + v_r / r - v / r ** 2)
    return r_cont, r_mom_x, r_mom_r


def ns_residuals_normalized(model, x, r, Re, nu=cfg.NU):
    """Same residuals, nondimensionalized by the Re-derived velocity
    scale U_ref(Re) and the pipe radius R, so residual-loss magnitude is
    comparable across the Re sweep -- same float32-conditioning rationale
    as the 2-D project's physics.py, and as model.py's output rescaling."""
    r_cont, r_mx, r_mr = ns_residuals(model, x, r, Re, nu=nu)
    Uref = u_ref(Re)
    cont_scale = Uref / cfg.R
    mom_scale = Uref ** 2 / cfg.R
    return r_cont / cont_scale, r_mx / mom_scale, r_mr / mom_scale