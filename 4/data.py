"""
data.py -- reference-data stores, point samplers and evaluation grids.

ANTI-OVERFITTING STRUCTURE (read this first)
--------------------------------------------
1.  Re split:    TRAIN Re have data; VAL Re are used only for early stopping and the
                 weight search; TEST Re are only loaded when final=True is passed
                 (evaluate.py --final). `RefStore` refuses to load a TEST Re otherwise.
2.  x-band hold-out: at every TRAIN Re, three streamwise bands (config.HOLDOUT_X_BANDS)
                 are removed from the DATA loss. They are scored separately.
3.  Fresh points every iteration: collocation/BC points are re-drawn at every Adam
                 step (DeepXDE-style resampling, Lu et al. 2021), so the network cannot
                 memorise a fixed point set; data points are drawn at random from the
                 allowed grid at every step.
4.  Collocation Re is drawn continuously in [Re_min, Re_max] of the TRAIN range, so the
                 (label-free) physics constrains the in-between Re. Extrapolation Re
                 (> Re_max) never see ANY loss term.
"""
import numpy as np
import torch

import config as C
from build_reference import load_reference

_TEST_SET = set(float(r) for r in C.RE_TEST)
_VAL_SET = set(float(r) for r in C.RE_VAL)


def _guard(Re, final):
    if float(Re) in _TEST_SET and not final:
        raise PermissionError(
            f"Re={Re} is a TEST Reynolds number. Test data may only be loaded with "
            f"final=True (evaluate.py --final) so it can never leak into tuning.")


def _tt(a, device, dtype):
    return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)


# --------------------------------------------------------------- data store
class TrainStore:
    """Flattened (x, y, Re) -> targets for all TRAIN Re, with hold-out bands removed."""

    def __init__(self, re_list, device, dtype, supervise_nut=False):
        xs, ys, res, us, vs, ps, ns = [], [], [], [], [], [], []
        self.n_holdout = 0
        for Re in re_list:
            assert float(Re) not in _TEST_SET and float(Re) not in _VAL_SET, \
                "TrainStore must only contain TRAIN Re"
            r = load_reference(Re)
            x, y = r["x"], r["y"]
            if C.SPARSE:                                   # profiles at a few stations only
                ix = np.unique([int(np.argmin(np.abs(x - s))) for s in C.SPARSE_STATIONS])
            else:
                allowed_x = x >= C.X_DATA_MIN
                for (a, b) in C.HOLDOUT_X_BANDS:
                    allowed_x &= ~((x >= a) & (x <= b))
                self.n_holdout += int(np.sum(~allowed_x & (x >= C.X_DATA_MIN)))
                ix = np.where(allowed_x)[0]
            X, Y = np.meshgrid(x[ix], y, indexing="ij")
            xs.append(X.ravel()); ys.append(Y.ravel()); res.append(np.full(X.size, float(Re)))
            us.append(r["U"][ix].ravel())
            vs.append((r["V"][ix] / C.v_scale(Re)).ravel())
            ps.append(np.repeat(r["P_hat"][ix][:, None], len(y), axis=1).ravel())
            ns.append((r["nut_tilde"][ix] * r["Reh"]).ravel())      # chi = nut_tilde/nu
        cat = lambda l: _tt(np.concatenate(l), device, dtype)
        self.x, self.y, self.Re = cat(xs), cat(ys), cat(res)
        self.u, self.v, self.p, self.chi = cat(us), cat(vs), cat(ps), cat(ns)
        self.N = self.x.numel()
        self.device = device

    def sample(self, n):
        idx = torch.randint(0, self.N, (n,), device=self.device)
        col = lambda t: t[idx].unsqueeze(1)
        return dict(x=col(self.x), y=col(self.y), Re=col(self.Re),
                    u=col(self.u), v=col(self.v), p=col(self.p), chi=col(self.chi))


# --------------------------------------------------- collocation / BC points
def _eta_sampler(n, retau, device, dtype, eta_min_yplus=0.5, frac_log=0.5):
    """Wall-normal coordinate in (0,1]: half uniform, half log-uniform in y+ in
    [eta_min_yplus, Re_tau] (clusters points in the viscous sublayer / buffer layer,
    where RANS gradients are steepest; Tannehill et al. 1997, Sec. 5.6)."""
    u = torch.rand(n, device=device, dtype=dtype)
    is_log = torch.rand(n, device=device) < frac_log
    ymin = eta_min_yplus / retau
    lg = ymin * (1.0 / ymin) ** torch.rand(n, device=device, dtype=dtype)   # log-uniform in [ymin, 1]
    return torch.where(is_log, lg, u).clamp(min=1e-7, max=1.0)


def _x_sampler(n, device, dtype, x_min):
    """Streamwise coordinate: half uniform in x, half ~ s^2 (s uniform) which matches the
    inlet clustering of the reference grid and of the network's sqrt(x) feature."""
    L = C.L_OVER_H
    xu = torch.rand(n, device=device, dtype=dtype) * L
    xc = L * torch.rand(n, device=device, dtype=dtype) ** 2
    pick = torch.rand(n, device=device) < 0.5
    return torch.where(pick, xc, xu).clamp(min=x_min, max=L)


def _re_sampler(n, device, dtype):
    lo, hi = np.log(C.RE_MIN), np.log(C.RE_MAX)
    return torch.exp(lo + (hi - lo) * torch.rand(n, device=device, dtype=dtype))


def retau_t(Re):
    return 0.0955 * (Re / 2.0) ** 0.875


def sample_colloc(n, device, dtype, x_min=0.02):
    Re = _re_sampler(n, device, dtype)
    x = _x_sampler(n, device, dtype, x_min)
    y = _eta_sampler(n, retau_t(Re), device, dtype)
    return x.unsqueeze(1), y.unsqueeze(1), Re.unsqueeze(1)


def sample_inlet(n, device, dtype):
    Re = _re_sampler(n, device, dtype)
    rt = retau_t(Re)
    eta_c = 5.0 / rt                                    # exclude y+<5: hard wall factor cannot follow the corner
    u = torch.rand(n, device=device, dtype=dtype)
    lg = eta_c * (1.0 / eta_c) ** torch.rand(n, device=device, dtype=dtype)
    y = torch.where(torch.rand(n, device=device) < 0.5, eta_c + (1 - eta_c) * u, lg)
    return torch.zeros(n, 1, device=device, dtype=dtype), y.unsqueeze(1), Re.unsqueeze(1)


def sample_outlet(n, device, dtype):
    Re = _re_sampler(n, device, dtype)
    y = _eta_sampler(n, retau_t(Re), device, dtype)
    return torch.full((n, 1), C.L_OVER_H, device=device, dtype=dtype), y.unsqueeze(1), Re.unsqueeze(1)


def sample_sym(n, device, dtype, x_min=0.02):
    Re = _re_sampler(n, device, dtype)
    x = _x_sampler(n, device, dtype, x_min)
    return x.unsqueeze(1), torch.ones(n, 1, device=device, dtype=dtype), Re.unsqueeze(1)


# ----------------------------------------------------------- evaluation sets
class EvalSet:
    """A (sub-sampled) reference grid for one Re with every needed normalisation."""

    def __init__(self, Re, device, dtype, stride_x=4, stride_y=2, final=False, full=False):
        _guard(Re, final)
        r = load_reference(Re)
        self.Re = float(Re)
        x, y = r["x"], r["y"]
        ix = np.where(x >= C.X_DATA_MIN)[0]
        if not full:
            ix = ix[::stride_x]
            if ix[-1] != len(x) - 1:
                ix = np.append(ix, len(x) - 1)
            iy = np.arange(0, len(y), stride_y)
        else:
            iy = np.arange(len(y))
        self.ix, self.iy = ix, iy
        X, Y = np.meshgrid(x[ix], y[iy], indexing="ij")
        self.shape = X.shape
        self.x = _tt(X.ravel()[:, None], device, dtype)
        self.y = _tt(Y.ravel()[:, None], device, dtype)
        self.Re_col = torch.full_like(self.x, self.Re)
        self.u = _tt(r["U"][np.ix_(ix, iy)].ravel(), device, dtype)
        self.v = _tt((r["V"][np.ix_(ix, iy)] / C.v_scale(Re)).ravel(), device, dtype)
        self.p = _tt(np.repeat(r["P_hat"][ix][:, None], len(iy), axis=1).ravel(), device, dtype)
        self.chi = _tt((r["nut_tilde"][np.ix_(ix, iy)] * r["Reh"]).ravel(), device, dtype)
        xv = X.ravel()
        L = C.L_OVER_H
        self.masks = {
            "all": np.ones_like(xv, bool),
            "inlet(x<10h)": xv < 10.0,
            "fully-dev(x>=0.9L)": xv >= C.X_FD_FRACTION * L,
            "x<3h(upstream of data)": xv < 3.0,
        }
        self.holdout_mask = np.zeros_like(xv, bool)
        for (a, b) in C.HOLDOUT_X_BANDS:
            self.holdout_mask |= (xv >= a) & (xv <= b)
        self.x_np = xv
        self.ref = r
        # wall shear coefficient Cf(x) is scored on the x grid (x >= 1h: the parabolic reference is
        # least reliable in the first ~1h and Cf is singular at x->0)
        xc = x[ix]
        self.cf_ref = r["Cf"][ix]
        self.cf_xt = _tt(xc[:, None], device, dtype)
        self.cf_Re = torch.full_like(self.cf_xt, self.Re)
        okx = xc >= 1.0
        self.cf_masks = {"all": okx, "inlet(x<10h)": okx & (xc < 10.0), "fully-dev(x>=0.9L)": xc >= C.X_FD_FRACTION * L,
                         "x<3h(upstream of data)": okx & (xc < 3.0)}
        hb = np.zeros_like(xc, bool)
        for (a_, b_) in C.HOLDOUT_X_BANDS:
            hb |= (xc >= a_) & (xc <= b_)
        self.cf_hold = hb


@torch.no_grad()
def predict(model, es, chunk=40000):
    outs = []
    for i in range(0, es.x.shape[0], chunk):
        U, V, Ph, nt = model(es.x[i:i + chunk], es.y[i:i + chunk], es.Re_col[i:i + chunk])
        outs.append(torch.cat([U, V / C.v_scale(es.Re), Ph, nt * (es.Re / 4.0)], dim=1))
    return torch.cat(outs, dim=0)


def rel_l2(pred, exact, mask=None):
    """100*sqrt(mean((pred-exact)^2)) / scale.  Fields are already divided by their
    GLOBAL, Re-only scales (U_b, 0.05*u_tau,est, 0.5*U_b^2), so scale == 1 here --
    same philosophy as Section 5 of the laminar report (never the local exact value)."""
    d = (pred - exact) ** 2
    if mask is not None:
        d = d[torch.as_tensor(mask, device=d.device)]
    return 100.0 * torch.sqrt(d.mean()).item()


def model_cf(model, es):
    """Cf(x) = 2 nu dU/dy|wall of the network (hard no-slip -> exact derivative at y=0)."""
    with torch.enable_grad():
        y = torch.zeros_like(es.cf_xt, requires_grad=True)
        U = model(es.cf_xt, y, es.cf_Re)[0]
        Uy, = torch.autograd.grad(U.sum(), y)
    return (2.0 * (4.0 / es.Re) * Uy).detach().cpu().numpy().ravel()


@torch.no_grad()
def score_set(model, es):
    """Dict {region: {u, v, p, chi, cf}} of relL2 (%).  cf is a RELATIVE error of Cf(x)
    (Cf > 0 everywhere, so a relative measure is safe here, unlike u/v at the wall)."""
    P = predict(model, es)
    tgt = [es.u, es.v, es.p, es.chi]
    out = {}
    for name, m in es.masks.items():
        out[name] = {f: rel_l2(P[:, k], tgt[k], m) for k, f in enumerate(["u", "v", "p", "chi"])}
    if es.holdout_mask.any():
        out["x-holdout-bands"] = {f: rel_l2(P[:, k], tgt[k], es.holdout_mask)
                                  for k, f in enumerate(["u", "v", "p", "chi"])}
    rel = (model_cf(model, es) - es.cf_ref) / es.cf_ref
    cfe = lambda m: 100.0 * float(np.sqrt(np.mean(rel[m] ** 2))) if m.any() else float("nan")
    for name, m in es.cf_masks.items():
        out[name]["cf"] = cfe(m)
    if "x-holdout-bands" in out:
        out["x-holdout-bands"]["cf"] = cfe(es.cf_hold)
    return out


def headline(score_dict):
    """Selection score: mean of u, p and Cf errors on the 'all' region. V and the latent chi are
    reported but not scored (see train.val_score)."""
    s = score_dict["all"]
    return (s["u"] + s["p"] + s["cf"]) / 3.0