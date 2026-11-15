"""
pinn.py -- axisymmetric RANS + Spalart-Allmaras PINN for developing turbulent pipe flow.

Network  (x, r, Re) -> (U, V, P_hat, nu_tilde)        U axial, V RADIAL (outward), P_hat = p/(0.5 U_b^2)
Variables are non-dimensional:  x,r in R ;  U,V in U_b ;  nu* = 2/Re (= 1/Re_R) ;  P = 0.5 P_hat (kinematic).

Hard constraints (exact in the network output form):
    wall r=1   : U = V = nu_tilde = 0        (factor g = tanh(Re_tau_est * y / 10), y = 1 - r)
    axis r=0   : V = 0                       (factor r)
Everything else (inlet, outlet, axis symmetry of U, P, nu_tilde) is a soft loss.

PDE residuals are the FULL ELLIPTIC axisymmetric RANS equations in divergence form
(nu_e = nu + nu_t varies strongly in space) + SA transport, evaluated by autograd.
"""
import math
import torch
import torch.nn as nn
import config as C

# ---- SA constants ------------------------------------------------------------------
CB1, CB2, SIG, KAP = 0.1355, 0.622, 2.0 / 3.0, 0.41
CW1 = CB1 / KAP ** 2 + (1.0 + CB2) / SIG
CW2, CW3, CV1 = 0.3, 2.0, 7.1
CV2, CV3 = 0.7, 0.9

LOG_MID = 0.5 * (math.log10(C.RE_MIN_TRAIN) + math.log10(C.RE_MAX_TRAIN))
LOG_HALF = 0.5 * (math.log10(C.RE_MAX_TRAIN) - math.log10(C.RE_MIN_TRAIN))
V_SCALE_FACTOR = 0.05          # v_s = 0.05 * u_tau_est   (V is tiny; see report Sec. 4.2)


# ----------------------------------------------------------------------------------
# Re-only scaling quantities (torch)
# ----------------------------------------------------------------------------------
def est_torch(Re):
    """returns (Cf_est, ut_est = u_tau/U_b, Re_tau_est, nu*)  -- functions of Re ONLY."""
    Re = Re.detach()
    s = 4.0 * torch.ones_like(Re)
    for _ in range(30):
        s = 2.0 * torch.log10(Re / s) - 0.8
    lam = 1.0 / s ** 2
    cf = lam / 4.0
    ut = torch.sqrt(lam / 8.0)
    ret = ut * Re / 2.0
    return cf, ut, ret, 2.0 / Re


# ----------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------
class PipePINN(nn.Module):
    def __init__(self, depth=C.NET_DEPTH, width=C.NET_WIDTH):
        super().__init__()
        layers = [nn.Linear(4, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 4)
        for m in list(self.hidden) + [self.out]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        self.depth, self.width = depth, width

    def features(self, x, r, Re, ret):
        eta = 1.0 - r
        f1 = 2.0 * torch.sqrt(torch.clamp(x, min=0.0) / C.L + 1e-12) - 1.0
        f2 = 2.0 * eta - 1.0
        f3 = 2.0 * torch.log1p(ret * eta) / torch.log1p(ret) - 1.0
        f4 = (torch.log10(Re) - LOG_MID) / LOG_HALF
        return torch.stack([f1, f2, f3, f4], dim=-1)

    def forward(self, x, r, Re):
        """x, r, Re : tensors of identical shape (n,). returns U, V, P_hat, nu_tilde (n,)"""
        _, ut, ret, nu = est_torch(Re)
        h = self.features(x, r, Re, ret)
        for lin in self.hidden:
            h = torch.tanh(lin(h))
        a = self.out(h)
        eta = 1.0 - r
        g = torch.tanh(ret * eta / 10.0)
        U = g * (1.0 + a[..., 0])
        V = (V_SCALE_FACTOR * ut) * g * g * r * a[..., 1]
        P = a[..., 2]
        nt = nu * ret * g * nn.functional.softplus(a[..., 3] - 3.0)
        return U, V, P, nt


# ----------------------------------------------------------------------------------
# SA terms (overflow-safe: see report Sec. 4.8)
# ----------------------------------------------------------------------------------
def sa_source_terms(nt, nu, Om, d):
    """returns production (cb1 S~ nt), destruction (cw1 fw (nt/d)^2), fv1.  Works in fp32/fp64."""
    chi = nt / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + CV1 ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    d2 = d * d
    Sb = nt * fv2 / (KAP ** 2 * d2)
    cond = Sb >= -CV2 * Om
    den = (CV3 - 2.0 * CV2) * Om - Sb
    den_safe = torch.where(cond, torch.ones_like(den), torch.clamp(den, min=1e-12))   # unused branch finite
    St_alt = Om + Om * (CV2 ** 2 * Om + CV3 * Sb) / den_safe
    St = torch.where(cond, Om + Sb, St_alt)
    St = torch.clamp(St, min=1e-8)
    r = torch.clamp(nt / (St * KAP ** 2 * d2), max=10.0, min=0.0)
    g = r + CW2 * (r ** 6 - r)
    c6 = CW3 ** 6
    g_lo = torch.clamp(g, max=CW3)                      # branch g < cw3 : original formula
    f_lo = g_lo * ((1.0 + c6) / (g_lo ** 6 + c6)) ** (1.0 / 6.0)
    g_hi = torch.clamp(g, min=CW3)                      # branch g >= cw3 : f_w = ((1+c6)/(1+(cw3/g)^6))^(1/6)
    f_hi = ((1.0 + c6) / (1.0 + (CW3 / g_hi) ** 6)) ** (1.0 / 6.0)
    fw = torch.where(g < CW3, f_lo, f_hi)
    prod = CB1 * St * nt
    dest = CW1 * fw * (nt / d) ** 2
    return prod, dest, fv1


def _grad(f, v):
    if not f.requires_grad:
        return torch.zeros_like(v)
    g = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(v) if g is None else g


def _grad2(f, vs):
    return [_grad(f, v) for v in vs]


def prepare(x, r, Re):
    x = x.clone().detach().requires_grad_(True)
    r = r.clone().detach().requires_grad_(True)
    return x, r, Re.detach()


def rans_sa_residuals(model, x, r, Re, scaled=True):
    """Residuals of continuity, x/r-momentum and SA at (x, r, Re). r must be > 0 (1/r terms).
    scaled=True : divided by Re-only wall-shear scales (u_tau_est, u_tau_est^2) -> 'u_tau units'."""
    x, r, Re = prepare(x, r, Re)
    U, V, Ph, nt = model(x, r, Re)
    P = 0.5 * Ph
    _, ut, ret, nu = est_torch(Re)

    Ux, Ur = _grad2(U, [x, r])
    Vx, Vr = _grad2(V, [x, r])
    Px, Pr = _grad2(P, [x, r])
    ntx, ntr = _grad2(nt, [x, r])

    chi = nt / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + CV1 ** 3)
    nue = nu + nt * fv1

    Fxx = 2.0 * nue * Ux
    Fxr = nue * (Ur + Vx)
    Frr = 2.0 * nue * Vr
    rFxr = r * Fxr
    rFrr = r * Frr

    dFxx_dx = _grad(Fxx, x)
    drFxr_dr = _grad(rFxr, r)
    dFxr_dx = _grad(Fxr, x)
    drFrr_dr = _grad(rFrr, r)

    cont = Ux + Vr + V / r
    momx = U * Ux + V * Ur + Px - (dFxx_dx + drFxr_dr / r)
    momr = U * Vx + V * Vr + Pr - (dFxr_dx + drFrr_dr / r - 2.0 * nue * V / (r * r))

    # SA
    d = torch.clamp(1.0 - r, min=1e-6)
    Om = torch.sqrt((Vx - Ur) ** 2 + 1e-10)
    prod, dest, _ = sa_source_terms(nt, nu, Om, d)
    Gx = (nu + nt) * ntx
    rGr = r * (nu + nt) * ntr
    diff = _grad(Gx, x) + _grad(rGr, r) / r + CB2 * (ntx ** 2 + ntr ** 2)
    sa = U * ntx + V * ntr - prod + dest - diff / SIG

    if scaled:
        cont, momx, momr, sa = cont / ut, momx / ut ** 2, momr / ut ** 2, sa / ut ** 2
    return dict(cont=cont, momx=momx, momr=momr, sa=sa)


# ----------------------------------------------------------------------------------
# Boundary losses
# ----------------------------------------------------------------------------------
def inlet_loss(model, r, Re):
    x = torch.zeros_like(r)
    U, V, _, nt = model(x, r, Re)
    _, ut, ret, nu = est_torch(Re)
    vs = V_SCALE_FACTOR * ut
    return (torch.mean((U - 1.0) ** 2) + torch.mean((V / vs) ** 2)
            + torch.mean(((nt / nu - 3.0) / 3.0) ** 2))


def outlet_loss(model, r, Re):
    x = torch.full_like(r, C.L)
    _, _, P, _ = model(x, r, Re)
    return torch.mean(P ** 2)


def axis_loss(model, x, Re):
    r = torch.zeros_like(x).requires_grad_(True)
    x = x.clone().detach()
    U, V, Ph, nt = model(x, r, Re)
    _, ut, ret, nu = est_torch(Re)
    Ur = _grad(U, r)
    Pr = _grad(0.5 * Ph, r)
    ntr = _grad(nt / nu, r)
    return (torch.mean((Ur / ut) ** 2) + torch.mean((Pr / ut ** 2) ** 2)
            + torch.mean((ntr / (0.1 * ret)) ** 2))


@torch.no_grad()
def predict(model, x, r, Re, batch=65536):
    outs = []
    for i in range(0, x.shape[0], batch):
        outs.append(torch.stack(model(x[i:i + batch], r[i:i + batch], Re[i:i + batch]), dim=-1))
    return torch.cat(outs, 0)


def wall_shear_cf(model, x, Re):
    """Cf = 2 nu dU/dy|wall, y = 1 - r  (exact via autograd thanks to the hard no-slip form)."""
    r = torch.ones_like(x).requires_grad_(True)
    U, _, _, _ = model(x.detach(), r, Re.detach())
    dUdr = torch.autograd.grad(U, r, torch.ones_like(U))[0]
    nu = 2.0 / Re
    return (2.0 * nu * (-dUdr)).detach()
