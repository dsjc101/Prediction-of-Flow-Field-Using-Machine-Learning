"""
model.py -- PINN architecture for the merged hybrid data+physics
channel-flow model.

NOTE ON PROVENANCE: this file is not present in the uploaded hybrid-report
file set (channel_pinn_hybrid_report.docx and train_hybrid.py/
validate_hybrid.py both import it, but it was missing from the upload).
It is reconstructed here from three sources that must all agree, and do:
  1. The hybrid report's own "Architecture" and "Output nondimensionalization"
     tables/sections (4 hidden layers x 80 units, tanh, Xavier-normal init,
     log10(Re) input embedding, (u/U0, v/U0, p/U0^2) output).
  2. The parameter shapes actually stored in the shipped checkpoint
     (channel_pinn_hybrid.pt): net.0 is Linear(3,80), net.2/4/6 are
     Linear(80,80), net.8 is Linear(80,3) -- i.e. exactly 4 hidden layers
     of width 80, input dim 3, output dim 3, confirming (1).
  3. train_hybrid.py's own use of `model.forward_normalized(x, y, Re)` and
     `model(x, y, Re)` as two distinct calls, confirming the normalized/
     physical split implemented below.

MERGED NORMALIZATION (this is the one deliberate change from what the
missing original file most likely did): the hybrid report's own text
normalizes velocity by U0(Re)=Re*nu/Dh and pressure by U0(Re)^2 (no
0.5 factor). This project instead uses the physics-only report's version
of the same idea (Sec. 7.1-7.2 there): velocity by U_ref(Re)=Re*nu/Dh
(identical formula, same Re definition) and pressure by the STANDARD
dynamic-pressure scale q(Re)=0.5*U_ref(Re)^2 -- the same nondimensional-
ization used for pressure coefficients (Cp=p/(0.5 U^2)) throughout the
fluid-mechanics literature. The two schemes differ only by a constant
factor of 2 on pressure, so this is a cosmetic-but-principled choice, made
because q(Re) is the form that generalizes recognizably to the cylinder-
wake case this project is headed back to (U_ref becomes the freestream
velocity U_inf(Re)=Re*nu/D there; q(Re) becomes literal dynamic pressure).
Both quantities are derived from the Re definition alone -- never from a
solved flow field -- so neither is solution leakage (see physics-only
report Sec. 7.2, "Why Output Rescaling Is Still Necessary (and Why It Is
Not the Same as Leaking the Answer)").
"""
import torch
import torch.nn as nn

import config as cfg


def u_ref(Re):
    """Reference velocity implied by the Re definition, Re = U_ref*Dh/nu.
    Known before anything is solved -- not a fitted or solved-for quantity.
    Identical formula to config.inlet_velocity(); kept as a separate name
    here because in this file its role is a NORMALIZATION SCALE, not a
    boundary condition."""
    return Re * cfg.NU / cfg.Dh


def p_scale(Re):
    """Dynamic-pressure scale q(Re) = 0.5 * U_ref(Re)^2 -- the standard
    Reynolds-number nondimensionalization for pressure (Cp = p/(0.5 U^2)).
    See module docstring for why this replaces the hybrid report's raw
    U_ref(Re)^2 scale."""
    return 0.5 * u_ref(Re) ** 2


class ChannelPINN(nn.Module):
    """(x, y, Re) -> (u, v, p). Re enters as log10(Re), rescaled to
    [-1, 1] over the training range, per the log-Reynolds embedding
    used in both source reports (Sec. 4.2 hybrid report / Sec. 5.4,7
    physics-only report), which keeps tanh activations away from
    saturation across a 10x Re range."""

    def __init__(self, n_hidden=4, width=80, re_log_min=None, re_log_max=None):
        super().__init__()
        if re_log_min is None:
            re_log_min = float(torch.log10(torch.tensor(float(min(cfg.RE_TRAIN)))))
        if re_log_max is None:
            re_log_max = float(torch.log10(torch.tensor(float(max(cfg.RE_TRAIN)))))

        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]  # u~, v~, p~ (normalized outputs)
        self.net = nn.Sequential(*layers)

        # Xavier-normal init, standard for tanh PINNs (Raissi et al. 2019
        # practice; explicitly called out in the hybrid report's
        # Architecture table).
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Stored as buffers so they save/load with the checkpoint.
        self.register_buffer("re_log_min", torch.tensor(float(re_log_min)))
        self.register_buffer("re_log_max", torch.tensor(float(re_log_max)))

    def _normalize_inputs(self, x, y, Re):
        xn = x / cfg.L          # x in [0,1]
        yn = y / cfg.h           # y in [-1,1]
        logRe = torch.log10(Re)
        ren = 2 * (logRe - self.re_log_min) / (self.re_log_max - self.re_log_min) - 1
        return torch.cat([xn, yn, ren], dim=1)

    def forward_normalized(self, x, y, Re):
        """Returns dimensionless (u/U_ref, v/U_ref, p/q) -- what the data
        and boundary losses are computed against directly in train_hybrid.py
        (avoids the float32 round-off problem discussed in the physics-only
        report Sec. 7.2, since these ratios are all O(1) regardless of Re)."""
        inp = self._normalize_inputs(x, y, Re)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def forward(self, x, y, Re):
        """Returns physical (u, v, p), rescaled by U_ref(Re) and q(Re)."""
        a, b, c = self.forward_normalized(x, y, Re)
        Uref = u_ref(Re)
        return a * Uref, b * Uref, c * p_scale(Re)