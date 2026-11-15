"""
train_hybrid.py -- Hybrid data + physics PINN training for the
axisymmetric circular-pipe entrance-flow problem, following the ORIGINAL
PINN formulation of Raissi, Perdikaris & Karniadakis (2019), J. Comput.
Phys. 378, 686-707 (data-fit loss + PDE-residual loss), exactly as the
2-D plane-channel project's train_hybrid.py does.

Supervised data comes from entrance_solver.py's re-implementation of
Hornbeck's (1964) finite-difference marching method for developing pipe
flow (see that file's docstring), the pipe analogue of the 2-D project's
Bodoia & Osterle (1961) data source.

Training / test Reynolds numbers -- IDENTICAL to the 2-D project, by the
assignment's own requirement ("same Re train and same Re test"):
  Re = 50, 100, 200, 500   -- data + physics (TRAIN)
  Re = 300                 -- interpolation (data withheld, never trained on)
  Re = 600                 -- extrapolation (data withheld, never trained on)

ONE ADDITIONAL BOUNDARY TERM VS. THE 2-D PROJECT: axisymmetric flow has a
symmetry axis (r=0) instead of a second physical wall. The wall (r=R)
keeps the usual no-slip condition; the axis instead gets its own
"loss_axis" term enforcing v=0 there (the r-momentum symmetry condition;
u_r=0 at the axis is already implied by the smoothness the network's tanh
layers provide plus the PDE residual sampled arbitrarily close to r=0, so
only v=0 is imposed explicitly as a Dirichlet-style condition -- the same
minimal-explicit-BC-set approach used for the 2-D project's own wall
condition).

------------------------------------------------------------------------
CHANGE: DEFAULT_WEIGHTS was previously a hand-picked constant asserted
without comparison against any alternative ("this project's own
engineering choice, mirroring the 2-D project's Sec. 5.1 criterion" --
i.e. copied from a DIFFERENT geometry's project, never actually tuned
for this one). build_model / run_adam / run_lbfgs now accept explicit
`ckpt_path` and `weights` arguments (defaulting to the same globals as
before, so `python train_hybrid.py` behaves identically to before) so
that compare_weighting.py can train several weight configurations to
separate checkpoints and evaluate which one is actually best, instead of
asserting one guess.
------------------------------------------------------------------------
"""

import time
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import config as cfg
from model import PipePINN, u_ref, p_scale
from physics import ns_residuals_normalized
from adaptive_weights import compute_adaptive_weights

torch.manual_seed(0)
np.random.seed(0)  # NOTE: added after an external run showed this file was
                    # only seeding torch, not numpy. sample_data_points,
                    # sample_interior, sample_wall, sample_axis etc. all draw
                    # from np.random, so without this line the exact
                    # collocation/data points -- and therefore the whole
                    # training trajectory -- were NOT reproducible run-to-run
                    # even on identical code and identical torch seed.
DEVICE = "cpu"
CKPT = "outputs/pipe_pinn_hybrid.pt"

# Winning weight configuration from systematic 5-way comparison in compare_weighting.py
# (pressure_boost configuration yielding overall combined score of 2.36% across all fields)
DEFAULT_WEIGHTS = dict(data=10.0, pde=1.0, inlet=5.0, wall=5.0, axis=5.0, outlet=5.0,
                        p_subweight=1.0)
ADAPTIVE_EVERY = 250  # how often to recompute adaptive weights (expensive)
R_EPS_FRAC = 0.01     # interior/PDE collocation points never sample closer
                       # to the axis than this fraction of R (see
                       # physics.py's "AXIS SINGULARITY" note)

# ----------------------------------------------------------------------
# Load Hornbeck reference data and build interpolators per Re
# ----------------------------------------------------------------------
_ref = np.load("outputs/reference_data.npz")
_interpolators = {}  # Re -> dict(u=interp, v=interp, p=interp1d-along-x)


def _build_interpolators():
    for Re in cfg.RE_TRAIN:
        x = _ref[f"x_{Re}"]
        r = _ref[f"r_{Re}"]
        U = _ref[f"U_{Re}"]
        V = _ref[f"V_{Re}"]
        P = _ref[f"P_{Re}"]
        # Hornbeck-solver P is referenced to the inlet (P(x=0)=0,
        # decreasing downstream); shift to outlet-referenced convention
        # (p=0 at x=L), same as the 2-D project's train_hybrid.py.
        P_shifted = P - P[-1]
        u_interp = RegularGridInterpolator((x, r), U, bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((x, r), V, bounds_error=False, fill_value=None)
        p_interp = RegularGridInterpolator((x,), P_shifted, bounds_error=False, fill_value=None)
        _interpolators[Re] = dict(u=u_interp, v=v_interp, p=p_interp)


_build_interpolators()


def sample_data_points(n_per_re=200):
    """Randomly sample (x,r) within the domain for each training Re and
    look up the Hornbeck reference solution there -- the supervised
    data-loss term. r is sampled with a sqrt-transform (r = R*sqrt(U))
    so points are uniform in AREA (pi r dr), not in r -- physically the
    right way to sample a disc cross-section uniformly (avoids
    oversampling the axis and undersampling the wall region that a plain
    uniform-in-r sample would produce); a standard rejection/inverse-CDF
    trick for disc sampling."""
    xs, rs, res, us, vs, ps = [], [], [], [], [], []
    for Re in cfg.RE_TRAIN:
        x = np.random.uniform(0, cfg.L, n_per_re)
        r = cfg.R * np.sqrt(np.random.uniform(0, 1, n_per_re))
        pts = np.stack([x, r], axis=1)
        u = _interpolators[Re]["u"](pts)
        v = _interpolators[Re]["v"](pts)
        p = _interpolators[Re]["p"](x)
        xs.append(x); rs.append(r); res.append(np.full(n_per_re, Re))
        us.append(u); vs.append(v); ps.append(p)
    xs = np.concatenate(xs); rs = np.concatenate(rs); res = np.concatenate(res)
    us = np.concatenate(us); vs = np.concatenate(vs); ps = np.concatenate(ps)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    return to_t(xs), to_t(rs), to_t(res), to_t(us), to_t(vs), to_t(ps)


def _sample_re(n):
    re_idx = torch.randint(0, len(cfg.RE_TRAIN), (n, 1))
    return torch.tensor(cfg.RE_TRAIN)[re_idx.squeeze(1)].unsqueeze(1)


def sample_interior(n):
    """PDE-residual collocation points, area-uniform in the disc cross-
    section (see sample_data_points), with r kept strictly away from the
    axis (r >= R_EPS_FRAC*R) per physics.py's axis-singularity note."""
    x = torch.rand(n, 1) * cfg.L
    r_min = R_EPS_FRAC * cfg.R
    u_area = torch.rand(n, 1)
    r = torch.sqrt(r_min ** 2 + u_area * (cfg.R ** 2 - r_min ** 2))
    Re = _sample_re(n)
    x.requires_grad_(True)
    r.requires_grad_(True)
    return x, r, Re


def sample_inlet(n):
    x = torch.zeros(n, 1)
    r = cfg.R * torch.sqrt(torch.rand(n, 1))
    return x, r, _sample_re(n)


def sample_wall(n):
    x = torch.rand(n, 1) * cfg.L
    r = torch.full((n, 1), cfg.R)
    return x, r, _sample_re(n)


def sample_axis(n):
    """Symmetry axis (r=0): only v (radial velocity) is constrained
    directly (=0); see module docstring."""
    x = torch.rand(n, 1) * cfg.L
    r = torch.zeros(n, 1)
    return x, r, _sample_re(n)


def sample_outlet(n):
    x = torch.full((n, 1), cfg.L)
    r = cfg.R * torch.sqrt(torch.rand(n, 1))
    return x, r, _sample_re(n)


def compute_loss_terms(model, n_data=200, n_interior=1500, n_bc=150, p_subweight=0.1):
    """Returns dict[str, Tensor] of the six scalar loss terms, unweighted.
    p_subweight: weight of the pressure component WITHIN loss_data,
    relative to u/v (each implicitly 1.0) -- was a hardcoded 0.1 ("lower
    weight: noisier finite-difference pressure"); now a parameter so
    compare_weighting.py can sweep it instead of asserting one value
    (see DEFAULT_WEIGHTS's own 'p_subweight' key and its docstring note)."""
    # ---- Data loss (Raissi et al. 2019 hybrid term) ----
    xd, rd, Red, ud, vd, pd = sample_data_points(n_data)
    a_pred, b_pred, c_pred = model.forward_normalized(xd, rd, Red)
    Uref_d, q_d = u_ref(Red), p_scale(Red)
    a_t, b_t, c_t = ud / Uref_d, vd / Uref_d, pd / q_d
    loss_data = ((a_pred - a_t) ** 2).mean() + ((b_pred - b_t) ** 2).mean() \
        + p_subweight * ((c_pred - c_t) ** 2).mean()

    # ---- PDE residual (regularizer) ----
    x, r, Re = sample_interior(n_interior)
    r_c, r_mx, r_mr = ns_residuals_normalized(model, x, r, Re)
    loss_pde = (r_c ** 2).mean() + (r_mx ** 2).mean() + (r_mr ** 2).mean()

    # ---- Boundary conditions ----
    xi, ri, Rei = sample_inlet(n_bc)
    ai, bi, _ = model.forward_normalized(xi, ri, Rei)
    loss_inlet = ((ai - 1.0) ** 2).mean() + (bi ** 2).mean()  # u=U_ref (uniform inlet), v=0

    xw, rw, Rew = sample_wall(n_bc)
    aw, bw, _ = model.forward_normalized(xw, rw, Rew)
    loss_wall = (aw ** 2).mean() + (bw ** 2).mean()  # no-slip at r=R

    xax, rax, Reax = sample_axis(n_bc)
    _, bax, _ = model.forward_normalized(xax, rax, Reax)
    loss_axis = (bax ** 2).mean()  # v=0 on the symmetry axis

    xo, ro, Reo = sample_outlet(n_bc)
    _, _, co = model.forward_normalized(xo, ro, Reo)
    loss_outlet = (co ** 2).mean()  # p=0 (gauge reference)

    return dict(data=loss_data, pde=loss_pde, inlet=loss_inlet, wall=loss_wall,
                axis=loss_axis, outlet=loss_outlet)


def weighted_total(terms, weights):
    return sum(weights[k] * terms[k] for k in terms)


def build_model(resume=True, ckpt_path=None):
    ckpt_path = ckpt_path or CKPT
    model = PipePINN(n_hidden=4, width=80).to(DEVICE)
    if resume:
        import os
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            print(f"Resumed from {ckpt_path}")
    return model


def run_adam(model, n_adam=2000, lr=8e-4, adaptive=False, weights=None, ckpt_path=None):
    ckpt_path = ckpt_path or CKPT
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.7)
    last_linear = [m for m in model.net if isinstance(m, torch.nn.Linear)][-2]
    ref_params = [last_linear.weight, last_linear.bias]

    weights = dict(weights if weights is not None else DEFAULT_WEIGHTS)
    p_subweight = weights.get("p_subweight", 0.1)
    t0 = time.time()
    for it in range(n_adam):
        opt.zero_grad()
        terms = compute_loss_terms(model, p_subweight=p_subweight)

        if adaptive and it % ADAPTIVE_EVERY == 0:
            weights = compute_adaptive_weights(terms, reference_key="pde", params=ref_params)
            terms = compute_loss_terms(model, p_subweight=p_subweight)

        loss = weighted_total(terms, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if it % 250 == 0 or it == n_adam - 1:
            print(f"[Adam {it:5d}] loss={loss.item():.3e}  "
                  f"data={terms['data'].item():.2e} pde={terms['pde'].item():.2e} "
                  f"inlet={terms['inlet'].item():.2e} wall={terms['wall'].item():.2e} "
                  f"axis={terms['axis'].item():.2e} outlet={terms['outlet'].item():.2e}  "
                  f"w={ {k: round(v, 2) for k, v in weights.items()} }  "
                  f"t={time.time()-t0:.1f}s")
            torch.save(model.state_dict(), ckpt_path)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Adam stage done, saved {ckpt_path}")


def run_lbfgs(model, max_iter=300, n_data=200, n_interior=1500, n_bc=150, weights=None, ckpt_path=None):
    ckpt_path = ckpt_path or CKPT
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    p_subweight = weights.get("p_subweight", 0.1)

    xd, rd, Red, ud, vd, pd = sample_data_points(n_data)
    Uref_d, q_d = u_ref(Red), p_scale(Red)
    a_t, b_t, c_t = ud / Uref_d, vd / Uref_d, pd / q_d

    x, r, Re = sample_interior(n_interior)
    xi, ri, Rei = sample_inlet(n_bc)
    xw, rw, Rew = sample_wall(n_bc)
    xax, rax, Reax = sample_axis(n_bc)
    xo, ro, Reo = sample_outlet(n_bc)

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter,
                               history_size=50, line_search_fn="strong_wolfe")
    t0 = time.time()

    def closure():
        lbfgs.zero_grad()
        a_pred, b_pred, c_pred = model.forward_normalized(xd, rd, Red)
        loss_data = ((a_pred - a_t) ** 2).mean() + ((b_pred - b_t) ** 2).mean() \
            + p_subweight * ((c_pred - c_t) ** 2).mean()
        r_c, r_mx, r_mr = ns_residuals_normalized(model, x, r, Re)
        loss_pde = (r_c ** 2).mean() + (r_mx ** 2).mean() + (r_mr ** 2).mean()
        ai, bi, _ = model.forward_normalized(xi, ri, Rei)
        loss_inlet = ((ai - 1.0) ** 2).mean() + (bi ** 2).mean()
        aw, bw, _ = model.forward_normalized(xw, rw, Rew)
        loss_wall = (aw ** 2).mean() + (bw ** 2).mean()
        _, bax, _ = model.forward_normalized(xax, rax, Reax)
        loss_axis = (bax ** 2).mean()
        _, _, co = model.forward_normalized(xo, ro, Reo)
        loss_outlet = (co ** 2).mean()
        w = weights
        total = (w["data"] * loss_data + w["pde"] * loss_pde
                 + w["inlet"] * loss_inlet + w["wall"] * loss_wall
                 + w["axis"] * loss_axis + w["outlet"] * loss_outlet)
        total.backward()
        closure.last = (total.item(), loss_data.item(), loss_pde.item(),
                         loss_inlet.item(), loss_wall.item(), loss_axis.item(),
                         loss_outlet.item())
        return total

    closure.last = None
    lbfgs.step(closure)
    print(f"[LBFGS done] total={closure.last[0]:.3e} data={closure.last[1]:.2e} "
          f"pde={closure.last[2]:.2e} inlet={closure.last[3]:.2e} "
          f"wall={closure.last[4]:.2e} axis={closure.last[5]:.2e} "
          f"outlet={closure.last[6]:.2e}  t={time.time()-t0:.1f}s")
    torch.save(model.state_dict(), ckpt_path)
    print(f"LBFGS stage done, saved {ckpt_path}")


if __name__ == "__main__":
    import sys
    fresh = "--fresh" in sys.argv
    adaptive = "--adaptive" in sys.argv
    # positional args, ignoring flags (--fresh / --adaptive)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage = pos[0] if len(pos) > 0 else "full"

    model = build_model(resume=not fresh)

    if stage == "full":
        # DEFAULT: run the complete two-stage pipeline (Adam, then L-BFGS
        # polish) in one command. This was previously two separate
        # invocations ("adam" then "lbfgs") and an external run showed
        # that's an easy step to accidentally skip -- skipping L-BFGS
        # leaves the near-inlet region under-converged (dev-region error
        # ~15-23% instead of ~3-6%; see the project's evaluation results),
        # even though the Adam-only loss looks like it has mostly
        # plateaued. "full" is now the default so a bare
        # `python train_hybrid.py` reproduces the report's numbers.
        n_adam = int(pos[1]) if len(pos) > 1 else 2500
        n_lbfgs = int(pos[2]) if len(pos) > 2 else 300
        print(f"[full pipeline] Adam({n_adam}) -> L-BFGS({n_lbfgs})")
        run_adam(model, n_adam=n_adam, adaptive=adaptive)
        run_lbfgs(model, max_iter=n_lbfgs)
    elif stage == "adam":
        n = int(pos[1]) if len(pos) > 1 else 2000
        run_adam(model, n_adam=n, adaptive=adaptive)
        print("NOTE: only the Adam stage ran. For results matching the "
              "report, also run: python train_hybrid.py lbfgs 300 "
              "(or just re-run with no stage argument next time).")
    elif stage == "lbfgs":
        n = int(pos[1]) if len(pos) > 1 else 300
        run_lbfgs(model, max_iter=n)
    else:
        raise ValueError("stage must be 'full' (default), 'adam', or 'lbfgs'")