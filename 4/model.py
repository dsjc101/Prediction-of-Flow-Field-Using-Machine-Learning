"""
model.py -- (x, y, Re) -> (U, V, P_hat, nu_tilde)

Design choices and their sources
--------------------------------
* tanh MLP, Xavier-normal init (Glorot & Bengio 2010) -- unchanged from the laminar report.
* Inputs are physics-motivated, Re-only transformations (no solved field involved):
    s    = sqrt(x/L)                 streamwise coordinate stretched towards the inlet, where the
                                     boundary layers grow ~ x^(1/2..4/5) (classical coordinate
                                     stretching, Tannehill, Anderson & Pletcher 1997, Sec. 5.6)
    eta  = y/h                       wall-normal coordinate
    z    = log(1+y+)/log(1+Re_tau)   wall-log coordinate, y+ = Re_tau,est(Re)*eta, with Re_tau,est
                                     from Dean (1978). Gives the net a coordinate in which the
                                     viscous sublayer, buffer layer and log layer all have O(1)
                                     extent (law-of-the-wall structure, Pope 2000, Ch. 7).
    r    = log10(Re) rescaled to [-1,1] over the TRAIN range (as in the laminar report).
* HARD no-slip (Lu et al. 2021 SISC 43:B1105; Sukumar & Srivastava 2022 CMAME 389:114333;
  also used with SA-PINNs by Patel et al. 2024): 
        U = g * (1 + a_u),     V = v_scale * g^2 * a_v,     nu_tilde = nu*Re_tau*g*softplus(a_n+b)
  with g = tanh(y+/10) -> U=V=nu_tilde=0 EXACTLY at the wall; V ~ y^2 and dV/dy=0 at the wall as
  continuity demands. Nothing is learnt about the wall BC, so no wall-BC weight exists to tune.
* Outputs normalised exactly like the laminar report: U/U_b, V/(0.05 u_tau,est), p/(0.5 U_b^2).
  nu_tilde (SA working variable) is returned in units U_b*h, positive by construction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class ChannelPINN(nn.Module):
    def __init__(self, hidden=C.HIDDEN, width=C.WIDTH, re_min=C.RE_MIN, re_max=C.RE_MAX,
                 length=C.L_OVER_H, features="wall"):
        super().__init__()
        self.L = float(length)
        self.mid = 0.5 * (math.log10(re_min) + math.log10(re_max))
        self.half = 0.5 * (math.log10(re_max) - math.log10(re_min))
        self.features = features
        n_in = 4 if features == "wall" else 3
        layers = [nn.Linear(n_in, width)]
        for _ in range(hidden - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 4))
        self.layers = nn.ModuleList(layers)
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        self.b_nt = -3.0

    # Re-only helpers (torch versions of config.py functions)
    @staticmethod
    def retau(Re):
        return 0.0955 * (Re / 2.0) ** 0.875

    def _features(self, x, y, Re):
        s = torch.sqrt(x / self.L + 1e-8)
        f = [2.0 * s - 1.0, 2.0 * y - 1.0]
        if self.features == "wall":
            rt = self.retau(Re)
            z = torch.log1p(rt * y) / torch.log1p(rt)
            f.append(2.0 * z - 1.0)
        f.append((torch.log10(Re) - self.mid) / self.half)
        return torch.cat(f, dim=1)

    def forward(self, x, y, Re):
        h = self._features(x, y, Re)
        for lin in self.layers[:-1]:
            h = torch.tanh(lin(h))
        a = self.layers[-1](h)
        rt = self.retau(Re)
        g = torch.tanh(rt * y / 10.0)                     # exact 0 at the wall
        nu = 4.0 / Re
        vs = C.VSCALE_FRAC * torch.sqrt(0.5 * 0.073 * (Re / 2.0) ** -0.25)   # 0.05*u_tau,est
        U = g * (1.0 + a[:, 0:1])
        V = vs * g * g * a[:, 1:2]
        Ph = a[:, 2:3]
        nt = nu * rt * g * F.softplus(a[:, 3:4] + self.b_nt)
        return U, V, Ph, nt
