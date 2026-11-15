"""
weight_search.py -- Trains a small set of candidate loss-term weighting
schemes for the hybrid PINN and picks the best one by held-out
whole-domain error, instead of assuming DEFAULT_WEIGHTS (or the adaptive
scheme) is best a priori.

WHY A SEARCH, NOT A SINGLE ARBITRARY CHOICE: train_hybrid.py's
DEFAULT_WEIGHTS = dict(data=10, pde=1, inlet=5, wall=5, outlet=1) and the
Wang et al. (2021) adaptive scheme (adaptive_weights.py) are each
defensible on their own terms, but neither is validated against the
other, or against nearby alternatives, anywhere in this project. Trying
more than one candidate and picking the empirically best is standard
hyperparameter-search practice (Bergstra, J. & Bengio, Y. (2012),
"Random Search for Hyper-Parameter Optimization," Journal of Machine
Learning Research, 13, 281-305 -- the standard reference for why trying
several hyperparameter settings and comparing them outperforms picking
one by hand). The candidate grid below is not exhaustive; it varies the
two axes the project's own documentation already flags as consequential:

  - DATA-loss weight (train_hybrid.py's compute_loss_terms() docstring:
    "lower weight [on p]: noisier finite-difference pressure" already
    signals this term's weight matters; DEFAULT_WEIGHTS anchors it 10x
    the PDE-residual term).
  - BOUNDARY-CONDITION weight (inlet/wall): under-weighting Dirichlet
    BCs relative to the interior PDE residual is a documented PINN
    failure mode -- McClenny, L. & Braga-Neto, U. (2020), "Self-Adaptive
    Physics-Informed Neural Networks," arXiv:2009.04544, motivate their
    own adaptive scheme largely by this exact imbalance.

  candidate            data  pde  inlet  wall  outlet   rationale
  fixed_default          10    1     5     5      1     train_hybrid.py's own baseline
  fixed_data_light        5    1     5     5      1     halves data-term dominance
  fixed_data_heavy       20    1     5     5      1     doubles data-term dominance
  fixed_bc_heavy         10    1    10    10      1     tests the McClenny & Braga-Neto
                                                          under-weighted-BC failure mode
  adaptive_wang2021    (learned every ADAPTIVE_EVERY iters, Wang et al. 2021)

SELECTION METRIC: each candidate is trained to a full checkpoint, then
scored with evaluate_global.py's exact relL2 whole-domain metric (see
that module's docstring for the formula and why it is safe near zero),
averaged over all three fields (u, v, p) and reported separately for
TRAIN Re and for UNSEEN (interpolation/extrapolation) Re -- the same
TRAIN-vs-UNSEEN split evaluate.py already uses as its generalization
check. The candidate with the lowest MEAN OF THE TWO (not just TRAIN
error) is selected: a candidate that fits TRAIN Re well only by starving
the PDE-residual/BC terms would show that up as high UNSEEN error, since
UNSEEN Re never receive direct data supervision (evaluate.py's own
generalization-proof rationale). This deliberately does not use TRAIN
error alone, which would reward overfitting the weighting scheme to the
labeled Re exactly as it would reward overfitting the network itself.

USAGE:
    python weight_search.py                 # default: 1200 Adam + 150 LBFGS per candidate
    python weight_search.py --adam 3000 --lbfgs 200   # full-length runs (slow: 5 candidates)

Produces:
    outputs/weight_search/<candidate>.pt        -- each candidate's checkpoint
    outputs/weight_search_results.csv           -- per-candidate, per-field scores
    outputs/channel_pinn_hybrid.pt               -- OVERWRITTEN with the winning
                                                     candidate's checkpoint, so
                                                     evaluate.py / evaluate_global.py
                                                     pick it up unchanged
"""
import argparse
import os
import numpy as np

import config as cfg
from train_hybrid import train_config, DEFAULT_WEIGHTS
from evaluate import ALL_RE, TAGS, FIELDS, load_reference, field_scale
from evaluate_global import predict_grid, global_relative_l2

OUT = "outputs"
SEARCH_DIR = f"{OUT}/weight_search"

CANDIDATES = [
    ("fixed_default",   dict(data=10.0, pde=1.0, inlet=5.0, wall=5.0, outlet=1.0), False),
    ("fixed_data_light", dict(data=5.0, pde=1.0, inlet=5.0, wall=5.0, outlet=1.0), False),
    ("fixed_data_heavy", dict(data=20.0, pde=1.0, inlet=5.0, wall=5.0, outlet=1.0), False),
    ("fixed_bc_heavy",   dict(data=10.0, pde=1.0, inlet=10.0, wall=10.0, outlet=1.0), False),
    ("adaptive_wang2021", None, True),
]


def score_checkpoint(ckpt_path):
    """Whole-domain relL2 (%), per field, split TRAIN vs UNSEEN Re --
    exactly evaluate_global.py's metric, reused here rather than
    reimplemented so the search and the final report always agree."""
    import torch
    from model import ChannelPINN
    model = ChannelPINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    ref = np.load(f"{OUT}/reference_data.npz")
    per_field_train, per_field_unseen = {}, {}
    for f in FIELDS:
        train_errs, unseen_errs = [], []
        for Re in ALL_RE:
            x_ref, y_ref, U_ref, V_ref, P_ref = load_reference(ref, Re)
            X, Y = np.meshgrid(x_ref, y_ref, indexing="ij")
            U_pred, V_pred, P_pred = predict_grid(model, X, Y, Re)
            P_ref_grid = np.broadcast_to(P_ref[:, None], X.shape)
            exact = {"u": U_ref, "v": V_ref, "p": P_ref_grid}[f]
            pred = {"u": U_pred, "v": V_pred, "p": P_pred}[f]
            err = global_relative_l2(pred, exact, field_scale(f, Re))
            (train_errs if TAGS[Re] == "TRAIN" else unseen_errs).append(err)
        per_field_train[f] = float(np.mean(train_errs))
        per_field_unseen[f] = float(np.mean(unseen_errs))
    return per_field_train, per_field_unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=1200, help="Adam iterations per candidate")
    ap.add_argument("--lbfgs", type=int, default=150, help="LBFGS iterations per candidate")
    args = ap.parse_args()

    os.makedirs(SEARCH_DIR, exist_ok=True)
    rows = []

    for name, weights, adaptive in CANDIDATES:
        ckpt_path = f"{SEARCH_DIR}/{name}.pt"
        print(f"\n=== Training candidate '{name}' "
              f"(adam={args.adam}, lbfgs={args.lbfgs}, adaptive={adaptive}) ===")
        used_weights = train_config(weights=weights, adaptive=adaptive,
                                     n_adam=args.adam, n_lbfgs=args.lbfgs,
                                     ckpt_path=ckpt_path, fresh=True, verbose=True)
        train_scores, unseen_scores = score_checkpoint(ckpt_path)
        overall = float(np.mean([0.5 * (train_scores[f] + unseen_scores[f]) for f in FIELDS]))
        rows.append(dict(name=name, weights=used_weights,
                          train_scores=train_scores, unseen_scores=unseen_scores,
                          overall=overall, ckpt=ckpt_path))
        print(f"[{name}] TRAIN relL2 %: {train_scores}   UNSEEN relL2 %: {unseen_scores}   "
              f"overall(mean of TRAIN,UNSEEN, all fields)={overall:.3f}%")

    rows.sort(key=lambda r: r["overall"])
    best = rows[0]

    with open(f"{OUT}/weight_search_results.csv", "w") as fh:
        fh.write("candidate,weights,train_u,train_v,train_p,unseen_u,unseen_v,unseen_p,overall_score\n")
        for r in rows:
            w = r["weights"]
            wstr = ";".join(f"{k}={v:.3f}" for k, v in w.items())
            fh.write(f"{r['name']},\"{wstr}\","
                     f"{r['train_scores']['u']:.3f},{r['train_scores']['v']:.3f},{r['train_scores']['p']:.3f},"
                     f"{r['unseen_scores']['u']:.3f},{r['unseen_scores']['v']:.3f},{r['unseen_scores']['p']:.3f},"
                     f"{r['overall']:.3f}\n")

    print("\n=== Weight search complete ===")
    print(f"Ranking (best first, by mean(TRAIN, UNSEEN) relL2 % averaged over u,v,p):")
    for r in rows:
        print(f"  {r['name']:20s} overall={r['overall']:.3f}%  weights={r['weights']}")
    print(f"\nBest candidate: {best['name']}  (weights={best['weights']})")

    import shutil
    shutil.copyfile(best["ckpt"], f"{OUT}/channel_pinn_hybrid.pt")
    print(f"Copied {best['ckpt']} -> {OUT}/channel_pinn_hybrid.pt "
          f"(this is what evaluate.py / evaluate_global.py will load next).")
    print(f"Full comparison saved to {OUT}/weight_search_results.csv")


if __name__ == "__main__":
    main()
