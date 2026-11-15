"""
train_hybrid.py -- Hybrid data + physics PINN training, following the
ORIGINAL PINN formulation of Raissi, Perdikaris & Karniadakis (2019),
J. Comput. Phys. 378, 686-707, which combines a supervised data-fit loss
with the PDE-residual loss (the zero-data variant is a special case of
this framework with the data-loss weight set to zero, not the default).

This is the winning architecture identified by comparing the two source
reports: the hybrid report's own results show 0.4-0.9% error against the
exact fully-developed profile and 2.3-8.0% mean error against the full
developing-region reference (including for both held-out Re), vs. the
physics-only report's 7-10%+ error and complete failure to reproduce
Re-dependent entrance-region development. See channel_pinn_hybrid_report
(Sec. 7, "Comparison with the Earlier Zero-Data Model") for the full
comparison table.

Supervised data comes from entrance_solver.py's re-implementation of the
Bodoia & Osterle (1961) finite-difference marching method (see that
file's docstring), NOT from this project's own OpenFOAM CFD.

Training Reynolds numbers: Re = 50, 100, 200, 500 (data + physics).
Test Reynolds numbers (data withheld, never seen in training):
  Re = 300 -- interpolation (inside [50, 500])
  Re = 600 -- extrapolation (outside the training range)

CHANGE FROM THE ORIGINAL (missing) FILE: pressure targets are now
normalized by model.p_scale(Re) = 0.5*U_ref(Re)^2 (the standard dynamic-
pressure scale) rather than raw U_ref(Re)^2, to match the merged model.py
-- see model.py's docstring for why. This is purely an internal-constant
change; it does not add or remove any information the network sees.

RUNNING: `python train_hybrid.py` with NO arguments now runs the full
default pipeline described in the README -- Adam for 3000 iterations from
scratch, then an LBFGS(200) polish -- so you don't have to type the two
commands separately. Pass explicit args (`adam 3000 --fresh`, `lbfgs 200`,
etc.) to run a single stage instead.

UPDATE: x-sampling for both the data loss and the PDE-residual loss is
now inlet-clustered rather than uniform (see _clustered_x below) to
address the high developing-region error reported near the entrance --
see that function's docstring for the citation trail.

UPDATE: DEFAULT_WEIGHTS is fixed to the winning candidate from the weight
search (outputs/weight_search_results.csv: data=20, pde=1, inlet=5, wall=5,
outlet=1), which outperformed the Wang et al. (2021) adaptive weighting scheme
and alternative fixed ratios on whole-domain relative L2 error across train and
unseen Reynolds numbers. The no-argument `python train_hybrid.py` runs Adam
(3000 iters) followed by LBFGS (200 iters) with these optimal weights.
Pass explicit args (`adam 3000 --adaptive`, etc.) to run with adaptive weighting.
"""

import time
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------
# Inlet-clustered x-sampling (fix for the high developing-region error
# reported in evaluate.py's Bodoia-Osterle comparison, concentrated near
# x=0). Rationale: a uniform inlet profile meeting a no-slip wall at x=0
# is a classical corner singularity for this problem (see e.g. Dennis &
# Smith (1973) on the parallel-plate entrance corner; also the
# elliptic-vs-parabolic mismatch this project's own entrance_solver.py
# docstring flags via Wendel (1961)), so the sharpest gradients the
# network has to fit are concentrated right at the inlet. Sampling x
# UNIFORMLY (as before) under-represents exactly that region. Biasing
# collocation/data points toward the inlet is standard practice for
# regions of known sharp gradients -- see the grid-clustering treatment
# in Tannehill, Anderson & Pletcher (1997), Computational Fluid
# Mechanics and Heat Transfer, 2nd ed., Sec. 5.6, and, specifically for
# PINNs, the residual-concentration rationale in Lu, Meng, Mao &
# Karniadakis (2021), "DeepXDE: A Deep Learning Library for Solving
# Differential Equations," SIAM Review 63(1):208-228, Sec. 3.4. This
# project uses a fixed a-priori bias (we already know where the
# singularity is) rather than DeepXDE's residual-adaptive resampling
# (which discovers it); that is a legitimate simplification, not a
# shortcut around the underlying method.
# ----------------------------------------------------------------------
CLUSTER_FRACTION = 0.5     # fraction of x-samples drawn from the biased
                            # (inlet-clustered) distribution; the rest are
                            # uniform over the whole domain so far-field /
                            # fully-developed accuracy is not sacrificed
CLUSTER_LENGTH_SCALE = 0.15  # [m] decay length of the biased density;
                              # ~= the smallest Le/Dh scale of interest
                              # (Re=50's Le is already only ~0.14 m)


def _clustered_x(n, L=None):
    """n x-samples in [0, L]: CLUSTER_FRACTION of them drawn from an
    exponential density concentrated near x=0 (via inverse-CDF sampling
    of a truncated exponential), the remainder uniform over [0, L]."""
    if L is None:
        L = cfg.L
    n_clustered = int(round(CLUSTER_FRACTION * n))
    n_uniform = n - n_clustered

    u = torch.rand(n_clustered, 1)
    cdf_at_L = 1.0 - np.exp(-L / CLUSTER_LENGTH_SCALE)
    x_clustered = -CLUSTER_LENGTH_SCALE * torch.log(1.0 - u * cdf_at_L)
    x_clustered = x_clustered.clamp(max=L)

    x_uniform = torch.rand(n_uniform, 1) * L
    x = torch.cat([x_clustered, x_uniform], dim=0)
    return x[torch.randperm(n)]

import config as cfg
from model import ChannelPINN, u_ref, p_scale
from physics import ns_residuals_normalized
from adaptive_weights import compute_adaptive_weights

torch.manual_seed(0)
DEVICE = "cpu"
CKPT = "outputs/channel_pinn_hybrid.pt"

# Fixed weights, now used ONLY as the starting point for the adaptive
# scheme's first EMA update (see adaptive_weights.py) and as the fallback
# if --fixed is explicitly passed. They are no longer what trains by
# default -- see the RUNNING note above and the __main__ block below.
DEFAULT_WEIGHTS = dict(data=20.0, pde=1.0, inlet=5.0, wall=5.0, outlet=1.0)
ADAPTIVE_EVERY = 250  # how often to recompute adaptive weights (expensive)

# ----------------------------------------------------------------------
# Load Bodoia-Osterle reference data and build interpolators per Re
# ----------------------------------------------------------------------
_ref = np.load("outputs/reference_data.npz")
_interpolators = {}  # Re -> dict(u=interp, v=interp, p=interp1d-along-x)


def _build_interpolators():
    for Re in cfg.RE_TRAIN:
        x = _ref[f"x_{Re}"]
        y = _ref[f"y_{Re}"]
        U = _ref[f"U_{Re}"]
        V = _ref[f"V_{Re}"]
        P = _ref[f"P_{Re}"]
        # Bodoia-Osterle solver references P to the inlet (P(x=0)=0,
        # decreasing downstream); this project's PINN uses an
        # outlet-referenced convention (p=0 at x=L), matching the
        # OpenFOAM-style outlet BC used elsewhere in the project. Shift
        # so P_shifted(x=L) = 0.
        P_shifted = P - P[-1]
        u_interp = RegularGridInterpolator((x, y), U, bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((x, y), V, bounds_error=False, fill_value=None)
        p_interp = RegularGridInterpolator((x,), P_shifted, bounds_error=False, fill_value=None)
        _interpolators[Re] = dict(u=u_interp, v=v_interp, p=p_interp)


_build_interpolators()


def sample_data_points(n_per_re=200):
    """Randomly sample (x,y) within the domain for each training Re and
    look up the Bodoia-Osterle reference solution there -- the supervised
    data-loss term of the hybrid PINN loss."""
    xs, ys, res, us, vs, ps = [], [], [], [], [], []
    for Re in cfg.RE_TRAIN:
        x = _clustered_x(n_per_re).numpy().flatten()  # inlet-clustered, not uniform
        y = np.random.uniform(-cfg.h, cfg.h, n_per_re)
        pts = np.stack([x, y], axis=1)
        u = _interpolators[Re]["u"](pts)
        v = _interpolators[Re]["v"](pts)
        p = _interpolators[Re]["p"](x)
        xs.append(x); ys.append(y); res.append(np.full(n_per_re, Re))
        us.append(u); vs.append(v); ps.append(p)
    xs = np.concatenate(xs); ys = np.concatenate(ys); res = np.concatenate(res)
    us = np.concatenate(us); vs = np.concatenate(vs); ps = np.concatenate(ps)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    return to_t(xs), to_t(ys), to_t(res), to_t(us), to_t(vs), to_t(ps)


def _sample_re(n):
    re_idx = torch.randint(0, len(cfg.RE_TRAIN), (n, 1))
    return torch.tensor(cfg.RE_TRAIN)[re_idx.squeeze(1)].unsqueeze(1)


def sample_interior(n):
    x = _clustered_x(n)  # inlet-clustered, not uniform -- see module header
    y = (torch.rand(n, 1) * 2 - 1) * cfg.h
    Re = _sample_re(n)
    x.requires_grad_(True)
    y.requires_grad_(True)
    return x, y, Re


def sample_inlet(n):
    x = torch.zeros(n, 1)
    y = (torch.rand(n, 1) * 2 - 1) * cfg.h
    return x, y, _sample_re(n)


def sample_walls(n):
    x = torch.rand(n, 1) * cfg.L
    sign = torch.where(torch.rand(n, 1) > 0.5, 1.0, -1.0)
    y = sign * cfg.h
    return x, y, _sample_re(n)


def sample_outlet(n):
    x = torch.full((n, 1), cfg.L)
    y = (torch.rand(n, 1) * 2 - 1) * cfg.h
    return x, y, _sample_re(n)


def compute_loss_terms(model, n_data=200, n_interior=1500, n_bc=150):
    """Returns dict[str, Tensor] of the five scalar loss terms, unweighted
    -- kept separate (rather than pre-summed) so both the fixed-weight
    path and the adaptive-weight path can consume the same terms."""
    # ---- Data loss (Raissi et al. 2019 hybrid term) ----
    xd, yd, Red, ud, vd, pd = sample_data_points(n_data)
    a_pred, b_pred, c_pred = model.forward_normalized(xd, yd, Red)
    Uref_d, q_d = u_ref(Red), p_scale(Red)
    a_t, b_t, c_t = ud / Uref_d, vd / Uref_d, pd / q_d
    loss_data = ((a_pred - a_t) ** 2).mean() + ((b_pred - b_t) ** 2).mean() \
        + 0.1 * ((c_pred - c_t) ** 2).mean()  # lower weight: noisier finite-difference pressure

    # ---- PDE residual (regularizer, per hybrid PINN framework) ----
    x, y, Re = sample_interior(n_interior)
    r_c, r_mx, r_my = ns_residuals_normalized(model, x, y, Re)
    loss_pde = (r_c ** 2).mean() + (r_mx ** 2).mean() + (r_my ** 2).mean()

    # ---- Boundary conditions ----
    xi, yi, Rei = sample_inlet(n_bc)
    ai, bi, _ = model.forward_normalized(xi, yi, Rei)
    loss_inlet = ((ai - 1.0) ** 2).mean() + (bi ** 2).mean()  # u=U_ref (uniform inlet), v=0

    xw, yw, Rew = sample_walls(n_bc)
    aw, bw, _ = model.forward_normalized(xw, yw, Rew)
    loss_wall = (aw ** 2).mean() + (bw ** 2).mean()  # no-slip

    xo, yo, Reo = sample_outlet(n_bc)
    _, _, co = model.forward_normalized(xo, yo, Reo)
    loss_outlet = (co ** 2).mean()  # p=0 (gauge reference)

    return dict(data=loss_data, pde=loss_pde, inlet=loss_inlet, wall=loss_wall, outlet=loss_outlet)


def weighted_total(terms, weights):
    return sum(weights[k] * terms[k] for k in terms)


def build_model(resume=True, ckpt_path=CKPT):
    model = ChannelPINN(n_hidden=4, width=80).to(DEVICE)
    if resume:
        import os
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            print(f"Resumed from {ckpt_path}")
    return model


def run_adam(model, n_adam=2000, lr=8e-4, adaptive=False, ckpt_path=CKPT, verbose=True):
    """adaptive=True (the default): weights are learned via the Wang et
    al. (2021) gradient-statistics scheme in adaptive_weights.py, starting
    from DEFAULT_WEIGHTS and updated every ADAPTIVE_EVERY iterations.
    adaptive=False: DEFAULT_WEIGHTS are used unchanged for the whole run.
    Returns the final weights dict, so the caller (see __main__) can carry
    the learned weights into the LBFGS polish stage instead of dropping
    back to DEFAULT_WEIGHTS there."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.7)
    # Reference layer for adaptive weighting: the last hidden Linear layer
    # (Wang et al. 2021 measure against a single representative layer,
    # not the whole network).
    last_linear = [m for m in model.net if isinstance(m, torch.nn.Linear)][-2]
    ref_params = [last_linear.weight, last_linear.bias]

    weights = dict(DEFAULT_WEIGHTS)
    t0 = time.time()
    for it in range(n_adam):
        opt.zero_grad()
        terms = compute_loss_terms(model)

        if adaptive and it % ADAPTIVE_EVERY == 0:
            weights = compute_adaptive_weights(terms, reference_key="pde", params=ref_params)
            # compute_adaptive_weights consumes the graph with retain_graph
            # calls; recompute the terms fresh for the actual backward pass
            terms = compute_loss_terms(model)

        loss = weighted_total(terms, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if it % 250 == 0 or it == n_adam - 1:
            if verbose:
                print(f"[Adam {it:5d}] loss={loss.item():.3e}  "
                      f"data={terms['data'].item():.2e} pde={terms['pde'].item():.2e} "
                      f"inlet={terms['inlet'].item():.2e} wall={terms['wall'].item():.2e} "
                      f"outlet={terms['outlet'].item():.2e}  "
                      f"w={ {k: round(v, 2) for k, v in weights.items()} }  "
                      f"t={time.time()-t0:.1f}s")
            torch.save(model.state_dict(), ckpt_path)
    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f"Adam stage done, saved {ckpt_path}")
    return weights


def run_lbfgs(model, max_iter=300, n_data=200, n_interior=1500, n_bc=150, weights=None,
               ckpt_path=CKPT, verbose=True):
    """weights: the loss weights to use for the whole LBFGS run (fixed
    for the duration -- LBFGS's strong-Wolfe line search calls closure()
    many times per step, so recomputing adaptive weights mid-step the way
    run_adam does isn't well-defined here). Defaults to DEFAULT_WEIGHTS
    if not given; pass the dict returned by run_adam(adaptive=True) to
    carry the learned weights into this polish stage instead."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    xd, yd, Red, ud, vd, pd = sample_data_points(n_data)
    Uref_d, q_d = u_ref(Red), p_scale(Red)
    a_t, b_t, c_t = ud / Uref_d, vd / Uref_d, pd / q_d

    x, y, Re = sample_interior(n_interior)
    xi, yi, Rei = sample_inlet(n_bc)
    xw, yw, Rew = sample_walls(n_bc)
    xo, yo, Reo = sample_outlet(n_bc)

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter,
                               history_size=50, line_search_fn="strong_wolfe")
    t0 = time.time()

    def closure():
        lbfgs.zero_grad()
        a_pred, b_pred, c_pred = model.forward_normalized(xd, yd, Red)
        loss_data = ((a_pred - a_t) ** 2).mean() + ((b_pred - b_t) ** 2).mean() \
            + 0.1 * ((c_pred - c_t) ** 2).mean()
        r_c, r_mx, r_my = ns_residuals_normalized(model, x, y, Re)
        loss_pde = (r_c ** 2).mean() + (r_mx ** 2).mean() + (r_my ** 2).mean()
        ai, bi, _ = model.forward_normalized(xi, yi, Rei)
        loss_inlet = ((ai - 1.0) ** 2).mean() + (bi ** 2).mean()
        aw, bw, _ = model.forward_normalized(xw, yw, Rew)
        loss_wall = (aw ** 2).mean() + (bw ** 2).mean()
        _, _, co = model.forward_normalized(xo, yo, Reo)
        loss_outlet = (co ** 2).mean()
        w = weights
        total = (w["data"] * loss_data + w["pde"] * loss_pde
                 + w["inlet"] * loss_inlet + w["wall"] * loss_wall
                 + w["outlet"] * loss_outlet)
        total.backward()
        closure.last = (total.item(), loss_data.item(), loss_pde.item(),
                         loss_inlet.item(), loss_wall.item(), loss_outlet.item())
        return total

    closure.last = None
    lbfgs.step(closure)
    if verbose:
        print(f"[LBFGS done] total={closure.last[0]:.3e} data={closure.last[1]:.2e} "
              f"pde={closure.last[2]:.2e} inlet={closure.last[3]:.2e} "
              f"wall={closure.last[4]:.2e} outlet={closure.last[5]:.2e}  "
              f"w={ {k: round(v, 2) for k, v in weights.items()} }  t={time.time()-t0:.1f}s")
    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f"LBFGS stage done, saved {ckpt_path}")


def train_config(weights=None, adaptive=False, n_adam=3000, n_lbfgs=200,
                  ckpt_path=CKPT, fresh=True, verbose=True):
    """Convenience wrapper used by weight_search.py to train one candidate
    weighting scheme end-to-end (Adam then LBFGS-polish) into its own
    checkpoint file, without touching the CLI pipeline above.
      weights=None, adaptive=True  -> Wang et al. (2021) adaptive scheme,
                                       starting from DEFAULT_WEIGHTS
      weights=<dict>, adaptive=False -> that fixed dict, unchanged, for
                                       both stages (a "fixed-weight
                                       candidate" in the weight search)
    Returns the weights dict actually used for the LBFGS stage (so the
    caller can record exactly what a fixed candidate was, or what the
    adaptive scheme converged to).
    """
    model = build_model(resume=not fresh, ckpt_path=ckpt_path)
    if adaptive:
        used_weights = run_adam(model, n_adam=n_adam, adaptive=True,
                                 ckpt_path=ckpt_path, verbose=verbose)
    else:
        global DEFAULT_WEIGHTS
        saved_default = DEFAULT_WEIGHTS
        DEFAULT_WEIGHTS = dict(weights)
        try:
            run_adam(model, n_adam=n_adam, adaptive=False,
                      ckpt_path=ckpt_path, verbose=verbose)
        finally:
            DEFAULT_WEIGHTS = saved_default
        used_weights = dict(weights)
    if n_lbfgs > 0:
        run_lbfgs(model, max_iter=n_lbfgs, weights=used_weights,
                   ckpt_path=ckpt_path, verbose=verbose)
    return used_weights


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No CLI args at all -> run the full default pipeline documented
        # in the README (`adam 3000 --fresh` followed by `lbfgs 200`).
        print("No stage given on the command line -- running the default "
              "pipeline: Adam(3000 iters, fresh, fixed weights) then "
              "LBFGS(200 iters, using the fixed weights).")
        model = build_model(resume=False)
        fixed_weights = run_adam(model, n_adam=3000, adaptive=False)
        run_lbfgs(model, max_iter=200, weights=fixed_weights)
    else:
        stage = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else (2000 if stage == "adam" else 300)
        fresh = "--fresh" in sys.argv
        # Adaptive is now the default for the "adam" stage; --fixed opts
        # back into the old constant DEFAULT_WEIGHTS behavior. --adaptive
        # is still accepted (now a no-op) so old invocations don't break.
        adaptive = "--fixed" not in sys.argv

        model = build_model(resume=not fresh)
        if stage == "adam":
            run_adam(model, n_adam=n, adaptive=adaptive)
        elif stage == "lbfgs":
            # Standalone `lbfgs` invocation has no just-converged Adam
            # weights to inherit, so it uses DEFAULT_WEIGHTS unless
            # --fixed/--adaptive says otherwise -- there is nothing to
            # switch here, LBFGS always needs a single fixed dict; this
            # branch exists so old commands still run.
            run_lbfgs(model, max_iter=n)
        else:
            raise ValueError("stage must be 'adam' or 'lbfgs'")