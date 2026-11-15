"""
model.py -- PINN architecture for the axisymmetric circular-pipe hybrid
data+physics model. Direct extension of the 2-D project's model.py: same
depth/width, same Xavier-normal init, same log10(Re) embedding, same
Re-derived normalization philosophy -- only the second spatial input
switches from y (Cartesian, wall-to-wall) to r (radial, axis-to-wall),
and a `to_cartesian` helper is added so the axisymmetric field can be
queried at any real 3-D point.

NORMALIZATION: identical convention to the 2-D project (kept for
consistency and, per that file's own reasoning, because this is the
scale that will generalize recognizably to other Re-parameterized flows):
velocity by U_ref(Re) = Re*nu/D (from the Re definition alone), pressure
by the standard dynamic-pressure scale q(Re) = 0.5*U_ref(Re)^2. Both are
known before anything is solved -- never derived from a solved field, so
neither is solution leakage (2-D project's model.py docstring, citing
that project's own physics-only report Sec. 7.2, has the full argument;
it applies unchanged here).

AXISYMMETRIC OUTPUT: the network predicts (u_x, u_r, p) as a function of
(x, r, Re) -- the two independent spatial coordinates the swirl-free,
axisymmetric problem actually depends on (see config.py's docstring for
why this is exact, not approximate, for this geometry/BC combination).
`to_cartesian` reconstructs the full 3-D velocity vector
(u_x, u_y, u_z) at an arbitrary (x, y, z) by evaluating at
r = sqrt(y^2+z^2) and rotating the predicted radial velocity onto the
(y, z) plane -- u_y = u_r*(y/r), u_z = u_r*(z/r) -- which is exactly how
an axisymmetric-no-swirl vector field is reassembled in 3-D (standard
vector decomposition in cylindrical coordinates, e.g. White 2006, App. B;
used the same way in the axisymmetric-PINN literature, e.g. the
axisymmetric RANS-PINN for jet flow of Acta Geotechnica (2026),
https://doi.org/10.1007/s11440-026-02979-7, which likewise keeps
separate (U, V) networks over (x, r) and reconstructs the full field from
them).
"""
import torch
import torch.nn as nn

import config as cfg


def u_ref(Re):
    """Reference velocity implied by the Re definition, Re = U_ref*D/nu.
    Known before anything is solved. Identical role to the 2-D project's
    u_ref(), with D (pipe diameter) replacing Dh (channel hydraulic
    diameter) -- both reduce to the same formula, Re*nu/(hydraulic
    diameter), since Dh = D for a circular pipe."""
    return Re * cfg.NU / cfg.D


def p_scale(Re):
    """Pressure normalization scale -- the TOTAL viscous pressure drop over
    the domain length, |dp/dx(Re)| * L, NOT dynamic pressure 0.5*U^2.

    CHANGE, and why: dynamic pressure (the earlier version of this
    function, q=0.5*U_ref^2, the standard external/aerodynamic Cp
    convention) is the wrong characteristic scale for INTERNAL,
    viscous-dominated duct flow. The ratio of the actual physical
    pressure drop to that dynamic-pressure scale is, for Hagen-Poiseuille
    flow (config.poiseuille_dpdx, itself derived from the exact profile):
        Delta_p_total / (0.5*U_ref^2) = 32*L / (R*Re)
    which for this project's geometry (L=4m, R=0.05m) is ~51 at Re=50
    down to ~4 at Re=600 -- i.e. the network's normalized-space pressure
    TARGET was never O(1) like u/v's targets are; it was 4x-51x too
    large, worse at low Re. That single mismatch (not a loss-weighting
    problem) is what produced a fully-developed p error stuck at ~90%
    across every weight configuration tried in compare_weighting.py's
    sweep, monotonically improving with Re -- exactly the ratio above.
    Normalizing by the actual total viscous pressure drop instead makes
    the target O(1) for every Re (Hagen-Poiseuille/Darcy-Weisbach
    friction-driven pressure drop for internal pipe flow; White, F.M.
    (2006), Viscous Fluid Flow, 3rd ed., McGraw-Hill, Ch. 6)."""
    return abs(cfg.poiseuille_dpdx(Re)) * cfg.L


class PipePINN(nn.Module):
    """(x, r, Re) -> (u_x, u_r, p) for axisymmetric circular-pipe flow.
    Architecture identical to the 2-D project's ChannelPINN (4 hidden
    layers, width 80, tanh, Xavier-normal init) so any accuracy
    differences between the two projects are attributable to the physics/
    geometry, not to a different network capacity. Re enters as
    log10(Re) rescaled to [-1,1] over the training range -- same
    log-Reynolds embedding and same rationale (keeps tanh away from
    saturation across a 10x Re range) as the 2-D project's model.py."""

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

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.register_buffer("re_log_min", torch.tensor(float(re_log_min)))
        self.register_buffer("re_log_max", torch.tensor(float(re_log_max)))

    def _normalize_inputs(self, x, r, Re):
        xn = x / cfg.L              # x in [0,1]
        rn = r / cfg.R               # r in [0,1] (axis to wall; no sign
                                     # ambiguity, unlike the channel's y
                                     # in [-1,1], since r is a radius)
        logRe = torch.log10(Re)
        ren = 2 * (logRe - self.re_log_min) / (self.re_log_max - self.re_log_min) - 1
        return torch.cat([xn, rn, ren], dim=1)

    def forward_normalized(self, x, r, Re):
        """Returns dimensionless (u_x/U_ref, u_r/U_ref, p/q)."""
        inp = self._normalize_inputs(x, r, Re)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def forward(self, x, r, Re):
        """Returns physical (u_x, u_r, p) at axisymmetric coordinate
        (x, r), rescaled by U_ref(Re) and q(Re)."""
        a, b, c = self.forward_normalized(x, r, Re)
        Uref = u_ref(Re)
        return a * Uref, b * Uref, c * p_scale(Re)

    def to_cartesian(self, x, y, z, Re, eps=1e-8):
        """Full 3-D field reconstruction at an arbitrary Cartesian point
        (x, y, z), y and z measured from the pipe axis. r = sqrt(y^2+z^2);
        the predicted radial velocity u_r is rotated onto the (y,z) plane
        (u_y = u_r*y/r, u_z = u_r*z/r) -- the standard cylindrical-to-
        Cartesian vector reconstruction for a swirl-free axisymmetric
        field (White 2006, App. B). Returns (u_x, u_y, u_z, p)."""
        r = torch.sqrt(y ** 2 + z ** 2 + eps ** 2)  # eps avoids a 0/0 exactly on the axis
        u_x, u_r, p = self.forward(x, r, Re)
        u_y = u_r * (y / r)
        u_z = u_r * (z / r)
        return u_x, u_y, u_z, p