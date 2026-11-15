"""
data.py -- reference data handling, data regimes (dense / sparse), VAL / evaluation point sets and
the random samplers for collocation and boundary points.
"""
import math
import numpy as np
import torch
import config as C
import pinn as P

_CACHE = {}


def load_ref(Re):
    if Re not in _CACHE:
        _CACHE[Re] = dict(np.load(C.ref_path(Re)))
    return _CACHE[Re]


def vs_np(Re):
    return P.V_SCALE_FACTOR * C.est_np(Re)[1]


def in_bands(x):
    m = np.zeros_like(x, dtype=bool)
    for a, b in C.HOLDOUT_BANDS:
        m |= (x >= a) & (x <= b)
    return m


def _pack(Re, ix, iy, dev):
    """flatten selected stations ix x cells iy of reference Re into torch tensors (float32)."""
    d = load_ref(Re)
    X, Y = np.meshgrid(d["x"][ix], d["r"][iy], indexing="ij")
    out = dict(
        x=X.ravel(), r=Y.ravel(),
        U=d["U"][np.ix_(ix, iy)].ravel(), V=d["V"][np.ix_(ix, iy)].ravel(),
        P=np.repeat(d["P_hat"][ix], len(iy)),
        nt=d["nt"][np.ix_(ix, iy)].ravel(),
        Re=np.full(X.size, float(Re)), ix=np.repeat(ix, len(iy)))
    out = {k: torch.as_tensor(v, dtype=torch.float32 if k != "ix" else torch.long, device=dev) for k, v in out.items()}
    out["Vh"] = out["V"] / float(vs_np(Re))
    return out


def _cat(parts):
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


# ----------------------------------------------------------------------------------
# training data
# ----------------------------------------------------------------------------------
def training_pool(regime, dev="cpu", re_list=None):
    re_list = re_list or C.RE_TRAIN
    parts = []
    for Re in re_list:
        d = load_ref(Re)
        x = d["x"]
        iy = np.arange(len(d["r"]))
        if regime == "dense":
            ix = np.where((x >= C.X_DATA_MIN) & (~in_bands(x)))[0]
        elif regime == "sparse":
            ix = np.array(sorted({int(np.argmin(np.abs(x - s))) for s in C.SPARSE_STATIONS}))
        else:
            raise ValueError(regime)
        parts.append(_pack(Re, ix, iy, dev))
    return _cat(parts)


# ----------------------------------------------------------------------------------
# VAL / evaluation point sets
# ----------------------------------------------------------------------------------
def val_set(Re, dev="cpu"):
    """cheap score set used every few hundred iterations: every 8th station x every 4th cell."""
    d = load_ref(Re)
    ix = np.arange(len(d["x"]))
    ix = ix[d["x"] >= C.X_DATA_MIN][::8]
    iy = np.arange(len(d["r"]))[::4]
    pts = _pack(Re, ix, iy, dev)
    # Cf stations: every 8th station with x >= 1
    ic = np.arange(len(d["x"]))
    ic = ic[d["x"] >= C.CF_X_MIN][::8]
    cf = dict(x=torch.as_tensor(d["x"][ic], dtype=torch.float32, device=dev),
              Re=torch.full((len(ic),), float(Re), device=dev),
              cf_ref=torch.as_tensor(d["Cf"][ic], dtype=torch.float32, device=dev))
    return pts, cf


def eval_set(Re, dev="cpu"):
    """full evaluation set: every 2nd station (x >= 0.05) x all cells; Cf for every 2nd station x >= 1."""
    d = load_ref(Re)
    ix = np.arange(len(d["x"]))
    ix = ix[d["x"] >= C.X_DATA_MIN][::2]
    iy = np.arange(len(d["r"]))
    pts = _pack(Re, ix, iy, dev)
    ic = np.arange(len(d["x"]))
    ic = ic[d["x"] >= C.CF_X_MIN][::2]
    cf = dict(x=torch.as_tensor(d["x"][ic], dtype=torch.float32, device=dev),
              Re=torch.full((len(ic),), float(Re), device=dev),
              cf_ref=torch.as_tensor(d["Cf"][ic], dtype=torch.float32, device=dev))
    return pts, cf, ix, iy


# ----------------------------------------------------------------------------------
# random samplers (fresh points every call)
# ----------------------------------------------------------------------------------
def _loguniform(n, lo, hi, dev):
    return torch.exp(torch.rand(n, device=dev) * (math.log(hi) - math.log(lo)) + math.log(lo))


def sample_re(n, dev):
    return _loguniform(n, C.RE_MIN_TRAIN, C.RE_MAX_TRAIN, dev)


def sample_x(n, dev, xmin=0.02):
    h = n // 2
    x1 = torch.rand(h, device=dev) * C.L
    x2 = C.L * torch.rand(n - h, device=dev) ** 2
    return torch.clamp(torch.cat([x1, x2]), min=xmin)


def sample_y(n, ret, dev, y_plus_min=0.5, y_max=1.0 - C.R_AXIS_MIN):
    """half uniform in (0, y_max], half log-uniform in y+ in [y_plus_min, ret*y_max]"""
    h = n // 2
    y1 = torch.rand(h, device=dev) * y_max
    ypl = torch.exp(torch.rand(n - h, device=dev) * (torch.log(ret[h:] * y_max) - math.log(y_plus_min)) + math.log(y_plus_min))
    y2 = ypl / ret[h:]
    y = torch.cat([y1, y2])
    return torch.clamp(y, min=1e-6, max=y_max)


def sample_pde(n, dev):
    Re = sample_re(n, dev)
    ret = P.est_torch(Re)[2]
    x = sample_x(n, dev)
    y = sample_y(n, ret, dev)
    return x, 1.0 - y, Re


def sample_inlet(n, dev):
    Re = sample_re(n, dev)
    ret = P.est_torch(Re)[2]
    h = n // 2
    y1 = (5.0 / ret[:h]) + torch.rand(h, device=dev) * (1.0 - 5.0 / ret[:h])
    ypl = torch.exp(torch.rand(n - h, device=dev) * (torch.log(ret[h:]) - math.log(5.0)) + math.log(5.0))
    y = torch.cat([y1, ypl / ret[h:]])
    return 1.0 - torch.clamp(y, max=1.0), Re


def sample_outlet(n, dev):
    Re = sample_re(n, dev)
    ret = P.est_torch(Re)[2]
    y = sample_y(n, ret, dev, y_max=1.0)
    return 1.0 - y, Re


def sample_axis(n, dev):
    Re = sample_re(n, dev)
    return sample_x(n, dev, xmin=0.0), Re


def fresh_residual_points(Re, n=6000, seed=777, dev="cpu"):
    """independent evaluation points for the PDE residual (fixed seed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ret = float(C.est_np(Re)[2])
    x = torch.clamp(C.L * torch.rand(n, generator=g) ** 2, min=0.05)
    ypl = torch.exp(torch.rand(n, generator=g) * (math.log(ret * (1 - C.R_AXIS_MIN)) - math.log(0.5)) + math.log(0.5))
    y = torch.clamp(ypl / ret, max=1 - C.R_AXIS_MIN)
    Rev = torch.full((n,), float(Re))
    return x.to(dev), (1.0 - y).to(dev), Rev.to(dev)
