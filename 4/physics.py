"""
physics.py -- steady incompressible RANS + Spalart-Allmaras residuals, by autograd.

Equations (non-dimensional: h=1, U_b=1, nu=1/Re_h, kinematic pressure P = p/(rho U_b^2) = 0.5*P_hat)
    continuity :  U_x + V_y = 0
    x-momentum :  U U_x + V U_y + P_x - d/dx[ nu_e 2U_x ] - d/dy[ nu_e (U_y+V_x) ] = 0
    y-momentum :  U V_x + V V_y + P_y - d/dx[ nu_e (U_y+V_x) ] - d/dy[ nu_e 2V_y ] = 0
    nu_e = nu + nu_t,  nu_t = nu_tilde * fv1   (Boussinesq; trace part absorbed in P)
    SA         :  U nt_x + V nt_y = cb1 S~ nt - cw1 fw (nt/d)^2
                                     + (1/sigma)[ div((nu+nt) grad nt) + cb2 |grad nt|^2 ]
This is the FULL elliptic RANS system (not the parabolised one used by the reference solver),
so agreement with the reference is an independent check, exactly as in the laminar report.
Formulation: Boussinesq / SA as in Patel et al. (2024, PRFluids 9:034605) -- RANS+SA inside a
PINN; Eivazi et al. (2022, PoF 34:075117) for RANS-PINNs in general.

Residual scaling: every residual is divided by a Re-only wall-shear scale
    tau_s = u_tau,est^2 (momentum, SA)   and   u_tau,est (continuity)
so that residual MSEs are O(1) and comparable across Re (they differ by decades otherwise).
"""
import torch

import config as C

SA = C.SA
_CV2, _CV3 = 0.7, 0.9


def _d(f, *vars_):
    return torch.autograd.grad(f, vars_, grad_outputs=torch.ones_like(f), create_graph=True)


def sa_terms(nt, nu, Om, d):
    """fv1, S~, fw  (SA-noft2 with the Allmaras-Johnson-Spalart 2012 limiter).

    Written to be finite in float32 for the FIRST AND SECOND derivatives that a PINN needs
    (found by test: the textbook form gave non-finite gradients for ~40% of random networks):
      * torch.where evaluates BOTH branches, so the unused S~ branch must itself be finite
        -> its denominator is replaced by 1 where the branch is not selected;
      * fw uses g^6 with g up to ~3e5 (r clamped at 10) -> g^6 and especially its second
        derivative (~g^10) overflow float32. Equivalent overflow-free form, u = g/cw3:
            fw = (1+cw3^6)^(1/6) * u / (1+u^6)^(1/6)
               = (1+cw3^6)^(1/6) * (1+u^-6)^(-1/6)         for u >= 1
        (identity, not an approximation);
      * r = nt/(S~ kappa^2 d^2): the denominator is floored so that 1/den^2 cannot overflow
        in the backward pass (r is clamped at 10 anyway).
    """
    chi = nt / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + SA["cv1"] ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    Sbar = nt * fv2 / (SA["kappa"] ** 2 * d ** 2)
    S_reg = Om + Sbar
    use_reg = Sbar >= -_CV2 * Om
    den = (_CV3 - 2 * _CV2) * Om - Sbar
    den_safe = torch.where(use_reg, torch.ones_like(den), den - 1e-12 * torch.sign(den + 1e-30))
    S_neg = Om + Om * (_CV2 ** 2 * Om + _CV3 * Sbar) / den_safe
    St = torch.where(use_reg, S_reg, S_neg)
    St = torch.clamp(St, min=1e-12)
    den_r = torch.clamp(St * SA["kappa"] ** 2 * d ** 2, min=1e-12)
    r = torch.clamp(nt / den_r, max=10.0)
    g = r + SA["cw2"] * (r ** 6 - r)
    u = g / SA["cw3"]
    u_big = torch.clamp(u, min=1.0)
    u_small = torch.clamp(u, max=1.0)
    f_big = (1.0 + u_big ** -6) ** (-1.0 / 6.0)
    f_small = u_small * (1.0 + u_small ** 6) ** (-1.0 / 6.0)
    fw = (1.0 + SA["cw3"] ** 6) ** (1.0 / 6.0) * torch.where(u >= 1.0, f_big, f_small)
    return fv1, St, fw


def pde_residuals(model, x, y, Re):
    """Return dict of scaled residuals at (x, y, Re); x and y get requires_grad here."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    U, V, Ph, nt = model(x, y, Re)
    P = 0.5 * Ph
    nu = 4.0 / Re
    Ux, Uy = _d(U, x, y)
    Vx, Vy = _d(V, x, y)
    Px, Py = _d(P, x, y)
    ntx, nty = _d(nt, x, y)

    d = torch.clamp(y, min=1e-8)
    Om = torch.abs(Vx - Uy)
    fv1, St, fw = sa_terms(nt, nu, Om, d)
    nut = nt * fv1
    ne = nu + nut

    txx = 2.0 * ne * Ux
    txy = ne * (Uy + Vx)
    tyy = 2.0 * ne * Vy
    txx_x, = _d(txx, x)
    txy_x, txy_y = _d(txy, x, y)
    tyy_y, = _d(tyy, y)

    cont = Ux + Vy
    momx = U * Ux + V * Uy + Px - (txx_x + txy_y)
    momy = U * Vx + V * Vy + Py - (txy_x + tyy_y)

    Fx = (nu + nt) * ntx
    Fy = (nu + nt) * nty
    Fx_x, = _d(Fx, x)
    Fy_y, = _d(Fy, y)
    sa = (U * ntx + V * nty
          - (SA["cb1"] * St * nt - SA["cw1"] * fw * (nt / d) ** 2
             + (Fx_x + Fy_y + SA["cb2"] * (ntx ** 2 + nty ** 2)) / SA["sigma"]))

    ut = torch.sqrt(0.5 * 0.073 * (Re / 2.0) ** -0.25)          # u_tau,est / U_b
    return dict(cont=cont / ut, mom_x=momx / ut ** 2, mom_y=momy / ut ** 2, sa=sa / ut ** 2)


def boundary_losses(model, inlet, outlet, sym):
    """Soft BCs: inlet (U=1, V=0, nu~=3nu), outlet gauge (P=0), centreline symmetry."""
    out = {}
    xi, yi, Ri = inlet
    U, V, Ph, nt = model(xi, yi, Ri)
    vs = C.VSCALE_FRAC * torch.sqrt(0.5 * 0.073 * (Ri / 2.0) ** -0.25)
    out["inlet"] = ((U - 1.0) ** 2).mean() + ((V / vs) ** 2).mean() \
        + ((nt * Ri / 4.0 - C.SA_NUT_INLET_OVER_NU) / C.SA_NUT_INLET_OVER_NU).pow(2).mean()
    xo, yo, Ro = outlet
    _, _, Pho, _ = model(xo, yo, Ro)
    out["outlet"] = (Pho ** 2).mean()
    xs, ys, Rs = sym
    ys = ys.clone().requires_grad_(True)
    Us, Vs, Phs, nts = model(xs, ys, Rs)
    vss = C.VSCALE_FRAC * torch.sqrt(0.5 * 0.073 * (Rs / 2.0) ** -0.25)
    Uy, = _d(Us, ys)
    Py, = _d(0.5 * Phs, ys)
    nty, = _d(nts, ys)
    ut = torch.sqrt(0.5 * 0.073 * (Rs / 2.0) ** -0.25)
    out["sym"] = ((Uy / ut) ** 2).mean() + ((Vs / vss) ** 2).mean() \
        + ((Py / ut ** 2) ** 2).mean() + ((nty * Rs / 4.0 / (0.1 * 0.0955 * (Rs / 2.0) ** 0.875)) ** 2).mean()
    return out