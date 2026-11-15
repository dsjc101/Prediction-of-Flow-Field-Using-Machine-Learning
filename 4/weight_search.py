"""
weight_search.py -- find the loss weights by search, not assumption.

    python weight_search.py --trials 40 --budget quick
    python weight_search.py --trials 40 --budget quick --resume      # continue the same study

Method
------
* Search space: log-uniform weights for  data, p_sub (pressure share inside the data term),
  cont, mom, sa, inlet, outlet, sym   (+ learning rate).  Wall BCs have NO weight (hard-imposed).
* Sampler: TPE (Bergstra et al. 2011, NIPS) via Optuna (Akiba et al. 2019, KDD) -- more
  sample-efficient than grid/random search (Bergstra & Bengio 2012 showed random >> grid;
  TPE improves on random).
* Pruner: Hyperband (Li et al. 2018, JMLR 18:1-52) -- bad configurations are stopped early.
* Objective: mean relL2 (u,v,p) on the VAL Reynolds numbers ONLY (config.RE_VAL). TEST Re are
  not loadable here (data.py raises PermissionError), so selection bias against the final
  test numbers is avoided (Cawley & Talbot 2010).
* Trial seeds differ from the final-training seeds, and the top-3 configs are RE-TRAINED with
  2 fresh seeds before one is declared the winner (guards against a lucky seed).
* A baseline with the old fixed weights and an equal-weights baseline are scored in the same
  way, so the report can state how much the search actually bought.
Outputs: outputs/weight_search_trials.csv, outputs/best_weights.json
"""
import argparse
import json
import os

import numpy as np
import optuna

import config as C
import train as T

# Ranges were WIDENED after search #1: its top-10 trials sat on the upper edge for data (1e3), the lower
# edges for mom (1e-4) and inlet (0.1), and near the top for lr -- i.e. the bounds were binding.
SPACE = dict(                    # (low, high), all log-uniform
    data=(1.0, 1e4), v_sub=(1e-3, 1.0), p_sub=(1e-2, 1.0), cont=(1e-2, 1e3), mom=(1e-5, 1e1),
    sa=(1e-5, 1e1), inlet=(1e-2, 1e2), outlet=(1e-2, 1e1), sym=(1e-2, 1e1))


def suggest(trial):
    w = {k: trial.suggest_float(k, lo, hi, log=True) for k, (lo, hi) in SPACE.items()}
    lr = trial.suggest_float("lr", 3e-4, 1e-2, log=True)
    return w, lr


def make_objective(budget, hidden, width):
    def objective(trial):
        w, lr = suggest(trial)

        def cb(it, val):
            print(f"  [trial {trial.number}] it {it}  VAL {val:.2f}%", flush=True)
            trial.report(val, it)
            return trial.should_prune()

        try:
            _, info = T.train(w, budget, seed=1000 + trial.number, lr=lr, hidden=hidden, width=width,
                              verbose=False, report_cb=cb, tag=f"trial{trial.number}")
        except (FloatingPointError, RuntimeError):
            raise optuna.TrialPruned()
        if not np.isfinite(info["best_val"]):
            raise optuna.TrialPruned()
        if trial.should_prune():
            raise optuna.TrialPruned()
        return info["best_val"]
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--budget", default=None, choices=list(T.BUDGETS), help="default: quick (GPU) / cpu_search (CPU)")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="(ignored: the study is ALWAYS resumed if it exists)")
    ap.add_argument("--fresh", action="store_true", help="delete the existing study first")
    ap.add_argument("--search_only", action="store_true", help="run trials only (use for parallel workers)")
    ap.add_argument("--finalize", action="store_true", help="skip trials; confirm top-3 and write best_weights.json")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--confirm_budget", default=None, help="budget for the top-3 re-training (default = --budget)")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    a.budget = a.budget or T.default_budget("search")
    dh, dw = T.default_arch()
    a.hidden, a.width = a.hidden or dh, a.width or dw
    dbfile = os.path.join(C.OUT_DIR, "weight_search.db")
    if a.fresh and os.path.exists(dbfile):
        os.remove(dbfile)
        print("deleted old study")
    db = f"sqlite:///{dbfile}"
    print(f"budget={a.budget}  net={a.hidden}x{a.width}  device={T.get_device()}", flush=True)
    study = optuna.create_study(
        study_name="weights", storage=db, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=T.BUDGETS[a.budget]["val_every"] * 2,
                                              max_resource=T.BUDGETS[a.budget]["adam"], reduction_factor=3))
    # enqueue the two baselines so they are scored under identical conditions
    if len(study.trials) == 0 and not a.finalize:     # baselines only once, by whichever worker starts first
        prev = os.path.join(C.OUT_DIR, "best_weights.json")
        if os.path.exists(prev):                       # re-score the previous winner under the current objective
            pj = json.load(open(prev))
            study.enqueue_trial(dict({k: pj["weights"][k] for k in SPACE}, lr=pj["lr"]))
        study.enqueue_trial(dict(T.DEFAULT_WEIGHTS, lr=1e-3))
        study.enqueue_trial(dict(T.DEFAULT_WEIGHTS, v_sub=1.0, lr=1e-3))   # = weights of the sanity run
        study.enqueue_trial(dict(data=1.0, v_sub=1.0, p_sub=1.0, cont=1.0, mom=1.0, sa=1.0, inlet=1.0,
                                 outlet=1.0, sym=1.0, lr=1e-3))
    if not a.finalize:
        study.optimize(make_objective(a.budget, a.hidden, a.width), n_trials=a.trials,
                       show_progress_bar=False, catch=(Exception,))
        if a.search_only:
            return
    df = study.trials_dataframe()
    df.to_csv(os.path.join(C.OUT_DIR, "weight_search_trials.csv"), index=False)

    done = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    done.sort(key=lambda t: t.value)
    print("\nTop trials (VAL relL2 %, lower is better):")
    for t in done[:5]:
        print(f"  #{t.number:3d} {t.value:6.2f}  " + "  ".join(f"{k}={v:.3g}" for k, v in t.params.items()))

    # confirmation: retrain top-3 with 2 new seeds -> pick best MEAN
    cb = a.confirm_budget or a.budget
    results = []
    for t in done[:3]:
        w = {k: t.params[k] for k in SPACE}
        vals = [t.value]
        for s in (5001, 5002):
            _, info = T.train(w, cb, seed=s, lr=t.params["lr"], hidden=a.hidden, width=a.width, verbose=False)
            vals.append(info["best_val"])
        results.append((float(np.mean(vals)), float(np.std(vals)), t))
        print(f"  confirm #{t.number}: VAL = {np.mean(vals):.2f} +- {np.std(vals):.2f}  ({vals})")
    results.sort(key=lambda r: r[0])
    m, sd, win = results[0]
    out = dict(weights={k: win.params[k] for k in SPACE}, lr=win.params["lr"], val_mean=m, val_std=sd,
               trial=win.number, budget=a.budget)
    base = [t for t in study.trials if t.number == 0 and t.value is not None]
    if base:
        out["baseline_old_default_val"] = base[0].value
    json.dump(out, open(os.path.join(C.OUT_DIR, "best_weights.json"), "w"), indent=1)
    print("\nWINNER:", json.dumps(out, indent=1))


if __name__ == "__main__":
    main()