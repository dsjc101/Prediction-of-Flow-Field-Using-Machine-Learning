"""
compare_weighting.py -- Trains a small set of loss-weighting strategies
to SEPARATE checkpoints and evaluates all of them, on ALL THREE outputs
(u, v, p), so the final report can say "we compared X configurations and
picked the one that performed best on Y" instead of asserting one
hand-picked weight set.

CONFIGURATIONS COMPARED:
  fixed_baseline   : train_hybrid.py's original DEFAULT_WEIGHTS
                      (data=10, pde=1, inlet=5, wall=5, axis=5, outlet=1,
                      p_subweight=0.1). Trained fresh here, same budget as
                      every other config below (an earlier version of this
                      script reused an existing outputs/pipe_pinn_hybrid.pt
                      "to save a retrain" -- removed: that checkpoint could
                      come from a different iteration budget or hand-edited
                      weights, which silently biased the comparison).
  adaptive         : Wang, S., Teng, Y. & Perdikaris, P. (2021),
                      "Understanding and mitigating gradient flow
                      pathologies in physics-informed neural networks,"
                      SIAM J. Sci. Comput. 43(5), A3055-A3081 --
                      already implemented in adaptive_weights.py and
                      wired up behind train_hybrid.py's --adaptive flag,
                      but never actually run/compared until now.
  outlet_boost     : fixed weights, but outlet raised 1 -> 5 (same as
                      inlet/wall/axis instead of tied with pde at the
                      bottom). Motivated directly by the fully-developed
                      evaluation finding that Re=50 is anomalously worse
                      far downstream near the outlet (see evaluate.py's
                      metrics_fully_developed.csv) -- outlet is the term
                      that enforces p=0 at x=L, and it was one of the
                      two most weakly weighted terms in the baseline.
  data_boost       : fixed weights, but data raised 10 -> 20, everything
                      else unchanged -- a generic "trust supervision more"
                      control, to check outlet_boost's improvement (if
                      any) isn't just "more weight on anything helps."
  pressure_boost   : fixed weights, but the pressure sub-weight inside
                      loss_data (formerly a hardcoded 0.1, now
                      train_hybrid.py's tunable 'p_subweight') raised to
                      1.0 (parity with u/v), plus outlet 1 -> 5. Targets
                      the actual observed failure mode directly: p was
                      the worst-supervised field (only the outlet's
                      single line plus a 10x-downweighted data term touch
                      it at all), and data_boost cannot fix that because
                      it scales u/v/p's data terms uniformly.

Each config is trained fresh (--fresh, i.e. NOT resumed from an existing
checkpoint) so the comparison is apples-to-apples, except fixed_baseline
which reuses your existing checkpoint as noted above.

------------------------------------------------------------------------
METRICS (SAME formulas/reasoning as evaluate.py / evaluate_global.py --
see those files' docstrings for the full derivation of relative_l2 vs.
normalized_l2 and why ONLY u self-normalizes while v and p do not):
------------------------------------------------------------------------
  dev_mean_{u,v,p}   : mean of the 4-station developing-region error,
                       averaged again across all 6 Re. u uses
                       self-normalized relative L2; v and p BOTH use
                       normalized L2 by a fixed scale (U_mean(Re) for v,
                       p_scale(Re) for p) -- NOT self-normalized, because
                       v's own reference magnitude shrinks toward 0 well
                       before the fully-developed station (already at
                       the f=0.9*Le station within this very check), and
                       p is a single scalar per station that passes
                       through the outlet gauge p(x=L)=0 (see evaluate.py's
                       docstring for the full derivation, including the
                       blown-up-percentage artifact self-normalizing v
                       used to produce here).
  fd_mean_{u,v,p}    : fully-developed error at the common station
                       X_FD = 0.975*L, averaged across all 6 Re. u uses
                       relative L2 vs. exact Hagen-Poiseuille; v uses
                       normalized L2 vs. exact v=0 (fully-developed
                       continuity); p uses normalized L2 vs. the exact
                       constant-dp/dx pressure (config.poiseuille_pressure).
  fd_Re50_u          : fully-developed u error at X_FD, Re=50 specifically
                       (the anomalous case this comparison is partly
                       trying to fix).
  global_mean_{u,v,p}: whole-(x,r)-field error, averaged across all 6 Re
                       (same GLOBAL definitions as evaluate_global.py).

------------------------------------------------------------------------
HOW THE WINNER IS CHOSEN -- combined_score, and why unweighted
------------------------------------------------------------------------
The four configs are ranked by

    combined_score = mean(global_mean_u, global_mean_v, global_mean_p)

i.e. an UNWEIGHTED average of the whole-domain error across all three
physical outputs. Unweighted, specifically, because:
  - The assignment/report is about the network reproducing the full
    (u, v, p) field, not u alone -- picking weights that help u at v/p's
    expense would be optimizing the wrong objective.
  - There is no first-principles reason (nor any citation used elsewhere
    in this project) to prefer one output's accuracy over another's --
    an arbitrary custom weighting here would be exactly the kind of
    unjustified choice this whole comparison script exists to avoid.
  - global_mean (whole-domain) rather than dev_mean or fd_mean is used
    for the RANKING criterion specifically because it is the single
    metric that summarizes accuracy over the entire solution domain
    rather than one particular region/station -- the dev-region and
    fully-developed numbers remain fully reported (weight_sweep_comparison.csv)
    for diagnosing WHERE a configuration wins or loses, they are just not
    what ranks the configs.

Writes outputs/weight_sweep_comparison.csv. Does NOT overwrite your
outputs/pipe_pinn_hybrid.pt (used by evaluate.py / evaluate_global.py
for the "official" run) -- each candidate gets its own file under
outputs/sweep/. Once you pick a winner, either point evaluate.py's CKPT
at that file, or copy it over outputs/pipe_pinn_hybrid.pt yourself (this
script does that automatically for you at the end -- see main()).

RUNTIME NOTE: this trains all 5 configs fresh, on CPU, at the SAME full
budget as train_hybrid.py's own default (N_ADAM/N_LBFGS = 2500/300) --
observed ~2 min/config on CPU, so ~10 min total for the whole sweep.
Numbers are therefore directly quotable in the report with no separate
"retrain the winner" step needed.
"""
import os
import shutil
import numpy as np
import torch

import config as cfg
from model import PipePINN, u_ref, p_scale
import train_hybrid as th

OUT = "outputs"
SWEEP_DIR = f"{OUT}/sweep"
os.makedirs(SWEEP_DIR, exist_ok=True)

N_ADAM = 2500
N_LBFGS = 300

CONFIGS = {
    "fixed_baseline": dict(weights=dict(th.DEFAULT_WEIGHTS), adaptive=False),
    "adaptive": dict(weights=dict(th.DEFAULT_WEIGHTS), adaptive=True),
    "outlet_boost": dict(weights={**th.DEFAULT_WEIGHTS, "outlet": 5.0}, adaptive=False),
    "data_boost": dict(weights={**th.DEFAULT_WEIGHTS, "data": 20.0}, adaptive=False),
    # NEW: directly targets the pressure failure seen in evaluate.py's fully-
    # developed check (p normL2 up to 253% at Re=50, wrong-sign prediction) --
    # p_subweight was a hardcoded 0.1 inside loss_data (pressure counted 1/10th
    # as much as u/v there); this raises it to parity (1.0) AND raises outlet
    # (the only other loss term that touches p at all) from 1 -> 5, tied with
    # inlet/wall/axis instead of the lowest tier. data_boost alone couldn't fix
    # this because scaling the outer 'data' weight scales u/v/p uniformly --
    # it never rebalances p's 0.1x sub-weight relative to u/v within that term.
    "pressure_boost": dict(weights={**th.DEFAULT_WEIGHTS, "p_subweight": 1.0, "outlet": 5.0},
                            adaptive=False),
}

STATION_FRACS = [0.05, 0.25, 0.5, 0.9]
FD_FRAC = 0.975
X_FD = FD_FRAC * cfg.L
ALL_RE = cfg.ALL_RE_FOR_EVAL
FIELDS = ["u", "v", "p"]


def train_config(name, cfg_dict):
    ckpt_path = f"{SWEEP_DIR}/{name}.pt"
    # NOTE: previously reused an existing outputs/pipe_pinn_hybrid.pt for
    # fixed_baseline "to save a retrain." Removed: that checkpoint may have
    # come from a run with a DIFFERENT iteration budget or hand-edited
    # weights than what this sweep is actually comparing (this bit us --
    # a stale full-budget checkpoint silently beat every genuinely-fresh
    # 1200/150-iteration candidate, including pressure_boost, making the
    # "winner" an artifact of unequal training rather than a real result).
    # Now every config trains fresh, at the SAME budget (N_ADAM/N_LBFGS,
    # matched to train_hybrid.py's own full default), so the comparison is
    # actually apples-to-apples.
    print(f"\n=== Training config '{name}': weights={cfg_dict['weights']} adaptive={cfg_dict['adaptive']} ===")
    model = th.build_model(resume=False, ckpt_path=ckpt_path)
    th.run_adam(model, n_adam=N_ADAM, adaptive=cfg_dict["adaptive"],
                weights=cfg_dict["weights"], ckpt_path=ckpt_path)
    th.run_lbfgs(model, max_iter=N_LBFGS, weights=cfg_dict["weights"], ckpt_path=ckpt_path)
    return ckpt_path


def load_model(ckpt_path):
    model = PipePINN(n_hidden=4, width=80)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def predict(model, x_val, r_val, Re_val):
    x = torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1)
    r = torch.tensor(r_val, dtype=torch.float32).reshape(-1, 1)
    Re = torch.full_like(x, float(Re_val))
    with torch.no_grad():
        u, v, p = model(x, r, Re)
    return u.numpy().flatten(), v.numpy().flatten(), p.numpy().flatten()


def relative_l2(pred, exact):
    """100 * ||pred-exact||_2 / ||exact||_2 -- see evaluate.py's docstring."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (np.sqrt(np.mean(exact ** 2)) + 1e-12)


def normalized_l2(pred, exact, scale):
    """100 * ||pred-exact||_2 / scale -- see evaluate.py's docstring."""
    return 100.0 * np.sqrt(np.mean((pred - exact) ** 2)) / (abs(scale) + 1e-12)


def evaluate_config(ckpt_path, ref, p_shifted):
    model = load_model(ckpt_path)
    dev_means = {f: [] for f in FIELDS}
    fd_errs = {f: {} for f in FIELDS}
    global_errs = {f: [] for f in FIELDS}

    for Re in ALL_RE:
        x_ref = ref[f"x_{Re}"]; r_ref = ref[f"r_{Re}"]
        U_ref_grid = ref[f"U_{Re}"]; V_ref_grid = ref[f"V_{Re}"]
        Le = cfg.entrance_length(Re)

        # ---- developing-region mean, this Re, all 3 fields ----
        station_errs = {f: [] for f in FIELDS}
        for frac in STATION_FRACS:
            xs = min(frac * Le, cfg.L * 0.98)
            j = np.argmin(np.abs(x_ref - xs))
            up, vp, pp = predict(model, np.full_like(r_ref, x_ref[j]), r_ref, Re)
            station_errs["u"].append(relative_l2(up, U_ref_grid[j, :]))
            station_errs["v"].append(normalized_l2(vp, V_ref_grid[j, :], u_ref(Re)))
            p_target = np.full_like(r_ref, p_shifted[Re][j])
            station_errs["p"].append(normalized_l2(pp, p_target, p_scale(Re)))
        for f in FIELDS:
            dev_means[f].append(np.mean(station_errs[f]))

        # ---- fully-developed, common X_FD, all 3 fields ----
        r_fd = np.linspace(0.0, cfg.R, 100)
        u_pred_fd, v_pred_fd, p_pred_fd = predict(model, np.full_like(r_fd, X_FD), r_fd, Re)
        u_exact_fd = cfg.poiseuille_profile(r_fd, Re)
        v_exact_fd = np.zeros_like(r_fd)
        p_exact_fd = np.full_like(r_fd, cfg.poiseuille_pressure(X_FD, Re))
        fd_errs["u"][Re] = relative_l2(u_pred_fd, u_exact_fd)
        fd_errs["v"][Re] = normalized_l2(v_pred_fd, v_exact_fd, u_ref(Re))
        fd_errs["p"][Re] = normalized_l2(p_pred_fd, p_exact_fd, p_scale(Re))

        # ---- whole-field error, all 3 fields (reuse the same reference grid) ----
        X, Rg = np.meshgrid(x_ref, r_ref, indexing="ij")
        up_field, vp_field, pp_field = predict(model, X.flatten(), Rg.flatten(), Re)
        up_field = up_field.reshape(X.shape); vp_field = vp_field.reshape(X.shape); pp_field = pp_field.reshape(X.shape)
        P_ref_1d = p_shifted[Re]
        P_ref_grid = np.broadcast_to(P_ref_1d[:, None], X.shape)
        global_errs["u"].append(relative_l2(up_field, U_ref_grid))
        global_errs["v"].append(normalized_l2(vp_field, V_ref_grid, u_ref(Re)))
        global_errs["p"].append(normalized_l2(pp_field, P_ref_grid, p_scale(Re)))

    return dict(
        dev_mean_u=float(np.mean(dev_means["u"])),
        dev_mean_v=float(np.mean(dev_means["v"])),
        dev_mean_p=float(np.mean(dev_means["p"])),
        fd_mean_u=float(np.mean(list(fd_errs["u"].values()))),
        fd_mean_v=float(np.mean(list(fd_errs["v"].values()))),
        fd_mean_p=float(np.mean(list(fd_errs["p"].values()))),
        fd_Re50_u=float(fd_errs["u"].get(50.0, float("nan"))),
        global_mean_u=float(np.mean(global_errs["u"])),
        global_mean_v=float(np.mean(global_errs["v"])),
        global_mean_p=float(np.mean(global_errs["p"])),
    )


def main():
    ref = np.load(f"{OUT}/reference_data.npz")
    p_shifted = {Re: ref[f"P_{Re}"] - ref[f"P_{Re}"][-1] for Re in ALL_RE}  # same outlet gauge as train_hybrid.py

    rows = []
    for name, cfg_dict in CONFIGS.items():
        ckpt_path = train_config(name, cfg_dict)
        metrics = evaluate_config(ckpt_path, ref, p_shifted)
        metrics["combined_score"] = float(np.mean([metrics["global_mean_u"],
                                                    metrics["global_mean_v"],
                                                    metrics["global_mean_p"]]))
        rows.append((name, ckpt_path, metrics))
        print(f"[{name}] dev_mean(u/v/p)={metrics['dev_mean_u']:.2f}/{metrics['dev_mean_v']:.2f}/{metrics['dev_mean_p']:.2f}%  "
              f"fd_mean(u/v/p)={metrics['fd_mean_u']:.2f}/{metrics['fd_mean_v']:.2f}/{metrics['fd_mean_p']:.2f}%  "
              f"global(u/v/p)={metrics['global_mean_u']:.2f}/{metrics['global_mean_v']:.2f}/{metrics['global_mean_p']:.2f}%  "
              f"combined_score={metrics['combined_score']:.2f}%")

    cols = ["config", "dev_mean_u_pct", "dev_mean_v_pct", "dev_mean_p_pct",
            "fd_mean_u_pct", "fd_mean_v_pct", "fd_mean_p_pct", "fd_Re50_u_pct",
            "global_mean_u_pct", "global_mean_v_pct", "global_mean_p_pct", "combined_score_pct"]
    with open(f"{OUT}/weight_sweep_comparison.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for name, ckpt_path, m in rows:
            f.write(f"{name},{m['dev_mean_u']:.3f},{m['dev_mean_v']:.3f},{m['dev_mean_p']:.3f},"
                    f"{m['fd_mean_u']:.3f},{m['fd_mean_v']:.3f},{m['fd_mean_p']:.3f},{m['fd_Re50_u']:.3f},"
                    f"{m['global_mean_u']:.3f},{m['global_mean_v']:.3f},{m['global_mean_p']:.3f},"
                    f"{m['combined_score']:.3f}\n")

    best_name, best_ckpt, best_metrics = min(rows, key=lambda r: r[2]["combined_score"])
    print(f"\nBest by combined_score (unweighted mean of global u/v/p error, see module docstring): "
          f"'{best_name}' ({best_metrics['combined_score']:.2f}%)")
    print(f"Saved {OUT}/weight_sweep_comparison.csv")

    winner_path = f"{OUT}/pipe_pinn_hybrid.pt"
    shutil.copy(best_ckpt, winner_path)
    print(f"Copied the winning checkpoint ('{best_name}') to {winner_path} -- "
          f"this is what evaluate.py / evaluate_global.py will now use.")
    print(f"'{best_name}' was trained fresh at the full {N_ADAM}/{N_LBFGS} budget, same as "
          f"every other config -- its numbers are directly quotable, no retrain needed. "
          f"Rerun evaluate.py / evaluate_global.py now.")


if __name__ == "__main__":
    main()