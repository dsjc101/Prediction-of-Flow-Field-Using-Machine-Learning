"""
weight_search.py
==================
Searches config.LOSS_WEIGHTS (data_u, data_v, data_w, data_p, pde, inlet,
wall, outlet) instead of asserting a hand-picked set, per the same
"every decision backed by a citable choice, not an arbitrary one"
discipline as the rest of this project.

METHOD: RANDOM search over the weight space, each candidate trained for a
SHORT run (config.ADAM_ITERS // search_iter_fraction Adam iterations, no
L-BFGS polish -- a short run is a cheap but consistent relative ranking
signal, the same "short-run proxy" idea used throughout hyperparameter
search practice), then SCORED on VALIDATION error at the two held-out Re
(config.RE_TEST_INTERP, config.RE_TEST_EXTRAP) -- i.e. Re the candidate
was never shown a single data point from during that short run -- using
the SAME relL2 definition and the SAME (u,v,w,p) equal-weighted mean used
as the headline metric in evaluate.py/evaluate_global.py, so "best
weights" means "best generalization to unseen Re," not "best fit to
training Re" (which would trivially favor whichever candidate simply
overweights the data term).

Random search, not grid search, is used because grid search's cost grows
as (points per dimension)^8 for this 8-weight space, while random search
is known to find comparably good or better hyperparameters at a fraction
of the evaluations in exactly this kind of moderate-to-high-dimensional
setting, because a grid wastes evaluations on unimportant dimensions
while random sampling still explores the important ones densely:

  Bergstra, J. & Bengio, Y. (2012), "Random Search for Hyper-Parameter
  Optimization," Journal of Machine Learning Research, 13, 281-305.
  https://jmlr.org/papers/v13/bergstra12a.html

SEARCH SPACE: each weight is drawn log-uniformly (standard practice for
scale/magnitude-type hyperparameters, also discussed in Bergstra & Bengio
2012) from [0.1, 50] for the four data_* terms and pde, and from
[0.5, 20] for inlet/wall/outlet (BC terms are conventionally kept within
a narrower band since they must remain enforced strongly enough to pin
down the solution but a value of 1000x the data term serves no purpose
here). config.LOSS_WEIGHTS itself is always included as one of the
candidates (a "seeded" search, so the result can never be worse than the
existing hand-set starting point).

USAGE:
    python weight_search.py --n-candidates 20 --short-iters 400

Writes checkpoints/weight_search_results.npy (every candidate + its
validation score) and prints the best candidate's weights -- COPY THOSE
into config.LOSS_WEIGHTS (or pass them straight to train_hybrid.train()
via a custom weights dict) before the final full run.
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim

import config
from pinn_model import DuctPINN, re_to_normalized
from train_hybrid import load_reference_data, sample_data_points, pde_residual, to_norm


def sample_candidate(rng):
    def loguniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return dict(
        data_u=loguniform(0.1, 50), data_v=loguniform(0.1, 50),
        data_w=loguniform(0.1, 50), data_p=loguniform(0.1, 50),
        pde=loguniform(0.1, 50),
        inlet=loguniform(0.5, 20), wall=loguniform(0.5, 20), outlet=loguniform(0.5, 20),
    )


def short_train(weights, ref, short_iters, device="cpu", n_data=1000, n_pde=2000, n_bc=500):
    """One short, fixed-weight training run (no adaptive re-weighting, no
    L-BFGS polish -- this function is a cheap RELATIVE-ranking signal for
    the search, not a substitute for the final full run in train_hybrid.py)."""
    model = DuctPINN().to(device)
    opt = optim.Adam(model.parameters(), lr=config.ADAM_LR)
    for it in range(short_iters):
        opt.zero_grad()
        losses = {}
        dlu = dlv = dlw = dlp = 0.0
        for Re in config.RE_TRAIN:
            x, y, z, u_ref, v_ref, w_ref, p_ref = sample_data_points(
                ref, Re, n_data // len(config.RE_TRAIN))
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            u_p, v_p, w_p, p_p = model(xn, yn, zn, ren)
            U_ref_scale = Re * config.NU / config.DH
            p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py
            dlu = dlu + ((u_p - torch.tensor((u_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32)) ** 2).mean()
            dlv = dlv + ((v_p - torch.tensor((v_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32)) ** 2).mean()
            dlw = dlw + ((w_p - torch.tensor((w_ref / U_ref_scale).reshape(-1, 1), dtype=torch.float32)) ** 2).mean()
            dlp = dlp + ((p_p - torch.tensor((p_ref / p_scale).reshape(-1, 1), dtype=torch.float32)) ** 2).mean()
        losses["data_u"], losses["data_v"] = dlu / len(config.RE_TRAIN), dlv / len(config.RE_TRAIN)
        losses["data_w"], losses["data_p"] = dlw / len(config.RE_TRAIN), dlp / len(config.RE_TRAIN)

        pde_loss = 0.0
        for Re in config.RE_ALL:
            x = np.random.uniform(0, config.L_DOMAIN, n_pde)
            y = np.random.uniform(0, config.H, n_pde)
            z = np.random.uniform(0, config.W, n_pde)
            Re_arr = np.full_like(x, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            pde_loss = pde_loss + pde_residual(model, xn, yn, zn, ren, config.NU)
        losses["pde"] = pde_loss / len(config.RE_ALL)

        inlet_loss = wall_loss = outlet_loss = 0.0
        for Re in config.RE_ALL:
            y = np.random.uniform(0, config.H, n_bc)
            z = np.random.uniform(0, config.W, n_bc)
            x0 = np.zeros_like(y)
            Re_arr = np.full_like(x0, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(x0, y, z, Re_arr, device)
            u_p, v_p, w_p, _ = model(xn, yn, zn, ren)
            inlet_loss = inlet_loss + ((u_p - 1.0) ** 2).mean() + (v_p ** 2).mean() + (w_p ** 2).mean()

            n_each = n_bc // 4
            xw = np.random.uniform(0, config.L_DOMAIN, n_each * 4)
            yw = np.concatenate([np.zeros(n_each), np.full(n_each, config.H),
                                  np.random.uniform(0, config.H, n_each), np.random.uniform(0, config.H, n_each)])
            zw = np.concatenate([np.random.uniform(0, config.W, n_each), np.random.uniform(0, config.W, n_each),
                                  np.zeros(n_each), np.full(n_each, config.W)])
            Re_arr = np.full_like(xw, Re, dtype=np.float64)
            xn, yn, zn, ren = to_norm(xw, yw, zw, Re_arr, device)
            u_p, v_p, w_p, _ = model(xn, yn, zn, ren)
            wall_loss = wall_loss + (u_p ** 2).mean() + (v_p ** 2).mean() + (w_p ** 2).mean()

            xo = np.full_like(y, config.L_DOMAIN)
            xn, yn, zn, ren = to_norm(xo, y, z, Re_arr, device)
            _, _, _, p_p = model(xn, yn, zn, ren)
            outlet_loss = outlet_loss + (p_p ** 2).mean()
        losses["inlet"] = inlet_loss / len(config.RE_ALL)
        losses["wall"] = wall_loss / len(config.RE_ALL)
        losses["outlet"] = outlet_loss / len(config.RE_ALL)

        total = sum(weights[k] * losses[k] for k in losses)
        total.backward()
        opt.step()
    return model


def validation_score(model, ref, device="cpu", n_val=4000):
    """
    Mean relL2 over u, v, w, p at the TWO held-out Re
    (config.RE_TEST_INTERP, config.RE_TEST_EXTRAP), each computed as

        relL2(field) = || field_pinn - field_ref ||_2 / || field_ref ||_2

    over n_val random (x,y,z) points sampled from that Re's marching
    reference (same relL2 definition used in evaluate.py/
    evaluate_global.py; see those files for why this norm is used rather
    than a pointwise-relative one). Lower is better; this is the score
    the random search maximizes (minimizes).
    """
    scores = []
    for Re in [config.RE_TEST_INTERP, config.RE_TEST_EXTRAP]:
        d = ref[Re]
        x0 = d["crossflow_valid_from"]
        Nx, Ny, Nz = d["U"].shape
        ix = np.random.randint(x0, Nx, n_val)
        iy = np.random.randint(0, Ny, n_val)
        iz = np.random.randint(0, Nz, n_val)
        x, y, z = d["x"][ix], d["y"][iy], d["z"][iz]
        u_ref, v_ref, w_ref, p_ref = d["U"][ix, iy, iz], d["V"][ix, iy, iz], d["W"][ix, iy, iz], d["P"][ix, iy, iz]
        Re_arr = np.full_like(x, Re, dtype=np.float64)
        with torch.no_grad():
            xn, yn, zn, ren = to_norm(x, y, z, Re_arr, device)
            u_p, v_p, w_p, p_p = model(xn, yn, zn, ren)
        U_ref_scale = Re * config.NU / config.DH
        p_scale = 0.5 * U_ref_scale ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see pinn_model.py
        u_p, v_p, w_p, p_p = (t.cpu().numpy().reshape(-1) for t in (u_p, v_p, w_p, p_p))
        for pred, ref_arr, scale in [(u_p, u_ref, U_ref_scale), (v_p, v_ref, U_ref_scale),
                                      (w_p, w_ref, U_ref_scale), (p_p, p_ref, p_scale)]:
            pred_phys = pred * scale
            num = np.linalg.norm(pred_phys - ref_arr)
            den = np.linalg.norm(ref_arr) + 1e-8 * scale  # floor for fields (v,w) that can be near-zero
            scores.append(num / den)
    return float(np.mean(scores))


def run_search(n_candidates=20, short_iters=400, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    ref = load_reference_data()
    candidates = [dict(config.LOSS_WEIGHTS)] + [sample_candidate(rng) for _ in range(n_candidates - 1)]
    results = []
    for i, w in enumerate(candidates):
        model = short_train(w, ref, short_iters, device=device)
        score = validation_score(model, ref, device=device)
        results.append(dict(weights=w, score=score))
        print(f"[{i+1}/{len(candidates)}] score(mean relL2, held-out Re, u+v+w+p)={score:.4f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in w.items()))

    best = min(results, key=lambda r: r["score"])
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    np.save(os.path.join(config.CKPT_DIR, "weight_search_results.npy"), results)
    print(f"\nBEST weights (score={best['score']:.4f}):")
    for k, v in best["weights"].items():
        print(f"    {k} = {v:.3f}")
    print(f"\nSaved all {len(results)} candidates to "
          f"{config.CKPT_DIR}/weight_search_results.npy")
    print("-> copy the BEST weights above into config.LOSS_WEIGHTS (or pass "
          "them as the starting point to train_hybrid.train()'s adaptive "
          "scheme) before the final full training run, and report this "
          "search (candidate count, short_iters, and the resulting best "
          "set) in the final report next to the fixed-vs-adaptive A/B "
          "comparison.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=20)
    ap.add_argument("--short-iters", type=int, default=400,
                     help="Adam iterations per candidate's short training run "
                          "(a fraction of config.ADAM_ITERS, e.g. 400 vs 2500).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_search(n_candidates=args.n_candidates, short_iters=args.short_iters,
               seed=args.seed, device=args.device)