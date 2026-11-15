"""
train_hybrid.py
=================
Hybrid data + physics PINN training for the rectangular duct, following
the SAME framework as the channel and pipe projects:

  Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019), "Physics-informed
  neural networks: A deep learning framework for solving forward and
  inverse problems involving nonlinear partial differential equations,"
  J. Comput. Phys., 378, 686-707, https://doi.org/10.1016/j.jcp.2018.10.045

PHYSICS-LOSS EQUATIONS: full elliptic 3-D incompressible Navier-Stokes
(NOT the parabolized form used only to generate reference/data-loss
targets, same design decision as both earlier projects), evaluated via
automatic differentiation at randomly sampled interior (x,y,z,Re) points:

  continuity:  u_x + v_y + w_z = 0
  x-momentum:  u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx+u_yy+u_zz) = 0
  y-momentum:  u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx+v_yy+v_zz) = 0
  z-momentum:  u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx+w_yy+w_zz) = 0

These are the standard Cartesian incompressible Navier-Stokes equations
with no curvature terms (contrast with the pipe project's v/r, u_r/r,
v_r/r-v/r^2 cylindrical terms, Sec 7 of the pipe report) -- a rectangular
duct in Cartesian coordinates has no metric curvature to introduce extra
terms, which is itself a clean cross-check between the two 3-D
extensions: setting the duct's height and width both -> the same value
and comparing against a very-high-aspect-ratio limit should recover the
plane-channel physics with no curvature-term artifacts, unlike the
pipe-vs-channel comparison which necessarily differs by those terms.

BOUNDARY CONDITIONS:
  inlet:  u=U_ref(Re), v=w=0            at x=0
  wall:   u=v=w=0                        at y in {0,H} and z in {0,W}
          (FOUR wall faces, vs the pipe's one wall + one axis, and the
          channel's two walls -- no axis/singularity term exists here,
          contrast with pipe Sec 7.1)
  outlet: p=0                            at x=L (gauge reference)
  [optional, if --quarter-domain] symmetry planes: v=0 at y=H/2, w=0 at
          z=W/2 -- the duct's own analogue of the pipe's axis condition,
          but here from discrete reflection symmetry, not a coordinate
          singularity.

LOSS WEIGHTS: config.LOSS_WEIGHTS, now 8 terms -- data_u, data_v, data_w,
data_p (CHANGED: v, w, p now have their own reference data via
duct_solver.py and so their own data-loss terms, previously only u did),
pde, inlet, wall, outlet. Starting values documented in config.py;
weight_search.py (see that file) performs an explicit grid search over
this space, scored on held-out-Re validation error, rather than asserting
a weighting by hand.

TRAINING SCHEDULE: Adam (config.ADAM_ITERS, lr=config.ADAM_LR, StepLR
decay) followed by L-BFGS polish (config.LBFGS_ITERS, strong-Wolfe line
search) -- identical two-stage schedule to both earlier projects.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.optim as optim

import config
from pinn_model import DuctPINN, re_to_normalized, mirror_quadrant_points
import adaptive_weights as aw


def load_reference_data():
    data = {}
    for f in sorted(glob.glob(os.path.join(config.DATA_DIR, "duct_Re*.npz"))):
        d = np.load(f)
        data[int(d["Re"])] = dict(
            x=d["x"], U=d["U"], V=d["V"], W=d["W"], P=d["P"],
            y=d["y"], z=d["z"], U_mean=float(d["U_mean"]),
            # first station index whose v, w, p are past the top-hat-inlet
            # singularity (see duct_solver.py / generate_data.py) -- data
            # points are never sampled below this station for v, w, p.
            crossflow_valid_from=int(d["crossflow_valid_from"])
            if "crossflow_valid_from" in d else 3,
        )
    return data


def sample_data_points(ref, Re, n_samples, quarter_domain=False):
    """
    Sample (x,y,z) points + reference u, v, w, p from the marching
    solution for Re (TRAIN Re only, per config.RE_TRAIN). Sampling
    excludes stations below crossflow_valid_from (the known top-hat-inlet
    singularity in v, w, p, see duct_solver.py/generate_data.py) so the
    data-loss reference values for v, w, p are never drawn from that
    contaminated region; u itself has no such exclusion (its own,
    separate, much smaller grid artifact is confined to station 0 only,
    see duct_solver.py's self_consistency_report).
    """
    d = ref[Re]
    Nx, Ny, Nz = d["U"].shape
    x0 = d["crossflow_valid_from"]
    ix = np.random.randint(x0, Nx, n_samples)
    iy = np.random.randint(0, Ny, n_samples)
    iz = np.random.randint(0, Nz, n_samples)
    x = d["x"][ix]
    y = d["y"][iy]
    z = d["z"][iz]
    u = d["U"][ix, iy, iz]
    v = d["V"][ix, iy, iz]
    w = d["W"][ix, iy, iz]
    p = d["P"][ix, iy, iz]
    return x, y, z, u, v, w, p


def pde_residual(model, x, y, z, Re_norm, nu):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)
    u, v, w, p = model(x, y, z, Re_norm)

    def grad(f, v_):
        return torch.autograd.grad(f, v_, grad_outputs=torch.ones_like(f),
                                    create_graph=True)[0]

    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)
    p_x, p_y, p_z = grad(p, x), grad(p, y), grad(p, z)

    u_xx, u_yy, u_zz = grad(u_x, x), grad(u_y, y), grad(u_z, z)
    v_xx, v_yy, v_zz = grad(v_x, x), grad(v_y, y), grad(v_z, z)
    w_xx, w_yy, w_zz = grad(w_x, x), grad(w_y, y), grad(w_z, z)

    cont = u_x + v_y + w_z
    mom_x = u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    mom_y = u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    mom_z = u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)
    return (cont ** 2).mean() + (mom_x ** 2).mean() + (mom_y ** 2).mean() + (mom_z ** 2).mean()


def to_norm(x_phys, y_phys, z_phys, Re_phys, device):
    xn = torch.tensor((x_phys / config.L_DOMAIN).reshape(-1, 1), dtype=torch.float32, device=device)
    yn = torch.tensor((y_phys / config.H).reshape(-1, 1), dtype=torch.float32, device=device)
    zn = torch.tensor((z_phys / config.W).reshape(-1, 1), dtype=torch.float32, device=device)
    ren = torch.tensor(re_to_normalized(Re_phys).reshape(-1, 1), dtype=torch.float32, device=device)
    return xn, yn, zn, ren


def train(quarter_domain=False, adaptive=None, device="cpu", n_data=2000,
          n_pde=4000, n_bc=1000, adaptive_warmup=100, adaptive_every=100):
    """
    adaptive: if None (default), falls back to config.PDE_ADAPTIVE_WEIGHTS
    (True by default -- see config.py). Pass adaptive=False (or run with
    --fixed-weights on the CLI) to use the fixed config.LOSS_WEIGHTS for
    the entire run instead, unmodified. NOTE: config.LOSS_WEIGHTS is now
    an 8-term dict (data_u, data_v, data_w, data_p, pde, inlet, wall,
    outlet) since v, w, p each have their own data-loss term (see
    duct_solver.py); weight_search.py searches this same 8-term space and
    its result should be treated as the reported starting weights, with
    --adaptive (Wang, Teng & Perdikaris 2021, see adaptive_weights.py)
    still available and still the recommended default for the run itself.

    With adaptive=True, weights start at config.LOSS_WEIGHTS (iteration 0)
    and are re-estimated every `adaptive_every` iterations, starting at
    iteration `adaptive_warmup`, via adaptive_weights.gradient_norm_weights
    -- i.e. each per-term loss's weight is set inversely proportional to
    its own parameter-gradient norm (smoothed with an EMA against the
    previous weight, alpha=0.9), so no single term's gradient can
    dominate or vanish relative to the others. This *replaces* the fixed
    10:1:5:5:1 guess with weights the training run finds for itself.
    """
    if adaptive is None:
        adaptive = config.PDE_ADAPTIVE_WEIGHTS
    ref = load_reference_data()
    model = DuctPINN().to(device)
    opt = optim.Adam(model.parameters(), lr=config.ADAM_LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.7)

    weights = dict(config.LOSS_WEIGHTS)
    loss_hist = []
    weight_hist = []

    for it in range(config.ADAM_ITERS):
        opt.zero_grad()
        losses = {}

        # ---- data loss (TRAIN Re only), one term per predicted quantity
        # (u, v, w, p) -- CHANGED from the single "data" term, since v, w,
        # p now have their own reference values (see duct_solver.py); each
        # is normalized the SAME way pinn_model.py now normalizes the
        # network's own outputs (U_ref(Re)=Re*nu/Dh for velocities,
        # p_scale(Re)=0.5*U_ref^2*(L_DOMAIN/Dh) for pressure -- see
        # pinn_model.py module docstring for why q alone is the wrong
        # scale here), so all four loss terms are on comparable,
        # dimensionless O(1) scales before weighting. ----
        data_loss_u = data_loss_v = data_loss_w = data_loss_p = 0.0
        for Re in config.RE_TRAIN:
            x, y, z, u_ref, v_ref, w_ref, p_ref = sample_data_points(
                ref, Re, n_data // len(config.RE_TRAIN))
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            u_pred, v_pred, w_pred, p_pred = model(xn, yn, zn, ren)
            U_ref_scale = Re * config.NU / config.DH
            p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py
            u_ref_t = torch.tensor((u_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
            v_ref_t = torch.tensor((v_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
            w_ref_t = torch.tensor((w_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
            p_ref_t = torch.tensor((p_ref / p_scale).reshape(-1, 1), dtype=torch.float32, device=device)
            data_loss_u = data_loss_u + ((u_pred - u_ref_t) ** 2).mean()
            data_loss_v = data_loss_v + ((v_pred - v_ref_t) ** 2).mean()
            data_loss_w = data_loss_w + ((w_pred - w_ref_t) ** 2).mean()
            data_loss_p = data_loss_p + ((p_pred - p_ref_t) ** 2).mean()
        losses["data_u"] = data_loss_u / len(config.RE_TRAIN)
        losses["data_v"] = data_loss_v / len(config.RE_TRAIN)
        losses["data_w"] = data_loss_w / len(config.RE_TRAIN)
        losses["data_p"] = data_loss_p / len(config.RE_TRAIN)

        # ---- pde residual loss (ALL Re, including held-out) ----
        pde_loss = 0.0
        for Re in config.RE_ALL:
            x = np.random.uniform(0, config.L_DOMAIN, n_pde)
            y = np.random.uniform(0, config.H, n_pde)
            z = np.random.uniform(0, config.W, n_pde)
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            pde_loss = pde_loss + pde_residual(model, xn, yn, zn, ren, config.NU)
        losses["pde"] = pde_loss / len(config.RE_ALL)

        # ---- inlet BC: u=U_ref, v=w=0 at x=0 ----
        inlet_loss = 0.0
        for Re in config.RE_ALL:
            y = np.random.uniform(0, config.H, n_bc)
            z = np.random.uniform(0, config.W, n_bc)
            x = np.zeros_like(y)
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            u_p, v_p, w_p, _ = model(xn, yn, zn, ren)
            inlet_loss = inlet_loss + ((u_p - 1.0) ** 2).mean() + (v_p ** 2).mean() + (w_p ** 2).mean()
        losses["inlet"] = inlet_loss / len(config.RE_ALL)

        # ---- wall BC: u=v=w=0 on all 4 faces ----
        wall_loss = 0.0
        for Re in config.RE_ALL:
            n_each = n_bc // 4
            x = np.random.uniform(0, config.L_DOMAIN, n_each * 4)
            y = np.concatenate([np.zeros(n_each), np.full(n_each, config.H),
                                 np.random.uniform(0, config.H, n_each),
                                 np.random.uniform(0, config.H, n_each)])
            z = np.concatenate([np.random.uniform(0, config.W, n_each),
                                 np.random.uniform(0, config.W, n_each),
                                 np.zeros(n_each), np.full(n_each, config.W)])
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            u_p, v_p, w_p, _ = model(xn, yn, zn, ren)
            wall_loss = wall_loss + (u_p ** 2).mean() + (v_p ** 2).mean() + (w_p ** 2).mean()
        losses["wall"] = wall_loss / len(config.RE_ALL)

        # ---- outlet BC: p=0 at x=L ----
        outlet_loss = 0.0
        for Re in config.RE_ALL:
            y = np.random.uniform(0, config.H, n_bc)
            z = np.random.uniform(0, config.W, n_bc)
            x = np.full_like(y, config.L_DOMAIN)
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            _, _, _, p_p = model(xn, yn, zn, ren)
            outlet_loss = outlet_loss + (p_p ** 2).mean()
        losses["outlet"] = outlet_loss / len(config.RE_ALL)

        if adaptive and it >= adaptive_warmup and it % adaptive_every == 0:
            # NOTE: gradient_norm_weights needs retain_graph=True per-term
            # backward passes (see adaptive_weights.py); it is called BEFORE
            # total.backward() below and does not consume the graph needed
            # for the subsequent weighted backward pass.
            #
            # CHANGED (bug fix): only the 4 physics/BC terms (pde, inlet,
            # wall, outlet) are passed to gradient_norm_weights now -- the
            # 4 data_* terms (data_u, data_v, data_w, data_p) are left
            # FIXED at whatever they were set to (weight_search.py's
            # result, or config.LOSS_WEIGHTS). Reason: gradient_norm_weights
            # renormalizes so ALL weights it's given average to 1 (see
            # adaptive_weights.py), so when it was applied to all 8 terms
            # together, a pathological gradient blowup in ONE term (which
            # is exactly what the p-normalization bug above caused in
            # data_p before it was fixed) dragged every OTHER weight down
            # with it each iteration -- including data_v and data_w, which
            # is very likely why they failed to converge even though the
            # bug was specifically in data_p. u, v, w, p are four
            # independent SUPERVISED targets with genuinely different
            # natural magnitudes (v, w are much smaller than u by physics,
            # not by a fittable scale error), which is not the kind of
            # physics-vs-BC gradient-pathology imbalance Wang, Teng &
            # Perdikaris (2021) designed this scheme to correct -- so the
            # data terms are kept out of it, and only the 4 PDE/BC terms
            # (which ARE the kind of physics-residual-vs-boundary-
            # condition balance the method targets) are adaptively
            # rebalanced among themselves.
            bc_losses = {k: losses[k] for k in ("pde", "inlet", "wall", "outlet")}
            bc_prev = {k: weights[k] for k in bc_losses}
            weights.update(aw.gradient_norm_weights(bc_losses, model, prev_weights=bc_prev))

        total = sum(weights[k] * losses[k] for k in losses)
        total.backward()
        opt.step()
        sched.step()
        loss_hist.append({k: float(v.item()) for k, v in losses.items()} | {"total": float(total.item())})
        weight_hist.append(dict(weights))

        if it % 100 == 0:
            print(f"[Adam {it:4d}/{config.ADAM_ITERS}] total={total.item():.4e}  "
                  + "  ".join(f"{k}={v.item():.3e}" for k, v in losses.items()))
            if adaptive:
                print(f"    weights: " + "  ".join(f"{k}={w:.3f}" for k, w in weights.items()))

    # ---- L-BFGS polish ----
    print(f"\nSwitching to L-BFGS polish ({config.LBFGS_ITERS} iterations)...")
    lbfgs = optim.LBFGS(model.parameters(), max_iter=config.LBFGS_ITERS,
                         line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        # reuse a fresh sample each closure call (cheap, standard practice)
        Re = config.RE_TRAIN[np.random.randint(len(config.RE_TRAIN))]
        x, y, z, u_ref, v_ref, w_ref, p_ref = sample_data_points(ref, Re, n_data)
        Re_arr = np.full_like(x, Re, dtype=np.float64)
        xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
        u_pred, v_pred, w_pred, p_pred = model(xn, yn, zn, ren)
        U_ref_scale = Re * config.NU / config.DH
        p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py
        u_ref_t = torch.tensor((u_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
        v_ref_t = torch.tensor((v_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
        w_ref_t = torch.tensor((w_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32, device=device)
        p_ref_t = torch.tensor((p_ref / p_scale).reshape(-1, 1), dtype=torch.float32, device=device)
        data_loss_u = ((u_pred - u_ref_t) ** 2).mean()
        data_loss_v = ((v_pred - v_ref_t) ** 2).mean()
        data_loss_w = ((w_pred - w_ref_t) ** 2).mean()
        data_loss_p = ((p_pred - p_ref_t) ** 2).mean()

        Re_pde = config.RE_ALL[np.random.randint(len(config.RE_ALL))]
        x = np.random.uniform(0, config.L_DOMAIN, n_pde)
        y = np.random.uniform(0, config.H, n_pde)
        z = np.random.uniform(0, config.W, n_pde)
        Re_arr = np.full_like(x, Re_pde, dtype=np.float64)
        xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
        pde_loss = pde_residual(model, xn, yn, zn, ren, config.NU)

        total = (weights["data_u"] * data_loss_u + weights["data_v"] * data_loss_v
                 + weights["data_w"] * data_loss_w + weights["data_p"] * data_loss_p
                 + weights["pde"] * pde_loss)
        total.backward()
        return total

    lbfgs.step(closure)
    final_loss = closure().item()
    print(f"Final loss after L-BFGS polish: {final_loss:.4e}")

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.CKPT_DIR, "duct_pinn.pt"))
    np.save(os.path.join(config.CKPT_DIR, "loss_history.npy"), loss_hist)
    np.save(os.path.join(config.CKPT_DIR, "weight_history.npy"), weight_hist)
    print(f"Final loss weights used (last value, {'adaptive' if adaptive else 'fixed'}): "
          + "  ".join(f"{k}={w:.3f}" for k, w in weights.items()))
    print(f"Saved model to {config.CKPT_DIR}/duct_pinn.pt, "
          f"loss history to {config.CKPT_DIR}/loss_history.npy, "
          f"and per-iteration weight history to {config.CKPT_DIR}/weight_history.npy")
    print("-> send checkpoints/loss_history.npy (or a printout of its first/"
          "last entries) back so the report's training-trajectory numbers "
          "are the actual run, not projected.")
    return model, loss_hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-weights", action="store_true",
                     help="Disable adaptive gradient-norm weight balancing "
                          "(Wang, Teng & Perdikaris 2021) and use the fixed "
                          "config.LOSS_WEIGHTS (10:1:5:5:1) for the whole "
                          "run instead -- this reproduces the OLD default "
                          "behaviour of this project, useful for an A/B "
                          "comparison against the new adaptive default.")
    ap.add_argument("--quarter-domain", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(quarter_domain=args.quarter_domain,
          adaptive=(False if args.fixed_weights else None),
          device=args.device)