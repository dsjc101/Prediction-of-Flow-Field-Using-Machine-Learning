"""
search_weights.py -- loss-weight search by TPE (Optuna), VAL Reynolds numbers ONLY.

  python search_weights.py --regime dense  --trials 48 --workers 6
  python search_weights.py --regime sparse --trials 48 --workers 6

Search space (log-uniform): data 1-1e4, v_sub 1e-3-1, p_sub 1e-2-1, cont 1e-2-1e3, mom 1e-5-10, sa 1e-5-10,
inlet 1e-2-1e2, outlet 1e-2-10, sym 1e-2-10, lr 3e-4-1e-2.  Wall BC has no weight (exact).
Each trial = one network trained with the cpu_search budget (seed 1000 + trial number); objective = VAL score
(mean over VAL Re of [relL2(U)+relL2(P_hat)+relL2(Cf)]/3).  The 3 best trials are re-trained with 2 fresh seeds
(5001, 5002); the best mean of three wins -> outputs/best_weights_<regime>.json
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import config as C

SPACE = dict(data=(1, 1e4), v_sub=(1e-3, 1), p_sub=(1e-2, 1), cont=(1e-2, 1e3), mom=(1e-5, 10), sa=(1e-5, 10),
             inlet=(1e-2, 1e2), outlet=(1e-2, 10), sym=(1e-2, 10), lr=(3e-4, 1e-2))

# starting points: the winners of the turbulent CHANNEL report (same loss structure) + generic baselines
CHANNEL_DENSE = dict(data=6769, v_sub=0.0064, p_sub=0.169, cont=225, mom=3.4e-5, sa=3.3e-5, inlet=0.12, outlet=5.5, sym=1.3, lr=4.4e-3)
CHANNEL_SPARSE = dict(data=529, v_sub=0.122, p_sub=0.214, cont=82.7, mom=6.5e-5, sa=8.9e-5, inlet=0.07, outlet=5.3, sym=1.8, lr=8.7e-3)
DEFAULTS = dict(data=10, v_sub=0.05, p_sub=0.1, cont=0.1, mom=0.01, sa=0.05, inlet=5, outlet=1, sym=1, lr=1e-3)
EQUAL = dict(data=1, v_sub=1, p_sub=1, cont=1, mom=1, sa=1, inlet=1, outlet=1, sym=1, lr=1e-3)

_G = {}


def _setup(regime, budget, threads=1):
    import torch, train as T, data as D
    torch.set_num_threads(threads)
    if "pool" not in _G:
        _G["pool"] = D.training_pool(regime, "cpu")
        _G["val"] = T.make_val_sets("cpu")
    return T


def objective_factory(regime, budget):
    def objective(trial):
        T = _setup(regime, budget)
        w = {k: trial.suggest_float(k, lo, hi, log=True) for k, (lo, hi) in SPACE.items()}
        seed = 1000 + trial.number
        m, info = T.train_model(regime, w, budget, seed, 1.0, "searched", device="cpu", pool=_G["pool"],
                                val_sets=_G["val"], verbose=False, label=f"trial{trial.number}")
        s = info["val_score"]
        print(f"  trial {trial.number:3d}  VAL {s:7.3f}  rollbacks {info['rollbacks']}  {info['seconds']:.0f}s", flush=True)
        return s if np.isfinite(s) else 1e6
    return objective


def worker(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(C.OUT_DIR, f'optuna_{a.regime}.db')}"
    study = optuna.load_study(study_name=f"pipe_{a.regime}", storage=storage)
    study.optimize(objective_factory(a.regime, a.budget), n_trials=a.n_per_worker)


def _confirm_job(args):
    regime, budget, w, seed = args
    T = _setup(regime, budget)
    m, info = T.train_model(regime, w, budget, seed, 1.0, "searched", device="cpu", pool=_G["pool"],
                            val_sets=_G["val"], verbose=False)
    return info["val_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["dense", "sparse"], required=True)
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", default="cpu_search", choices=list(C.BUDGETS))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n_per_worker", type=int, default=8)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = os.path.join(C.OUT_DIR, f"optuna_{a.regime}.db")
    if os.path.exists(db):
        os.remove(db)
    storage = f"sqlite:///{db}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=0)
    study = optuna.create_study(study_name=f"pipe_{a.regime}", storage=storage, sampler=sampler, direction="minimize")
    base = [CHANNEL_DENSE if a.regime == "dense" else CHANNEL_SPARSE, DEFAULTS, dict(DEFAULTS, v_sub=1.0), EQUAL]
    prev = os.path.join(C.OUT_DIR, "best_weights_dense.json")
    if a.regime == "sparse" and os.path.exists(prev):
        base.insert(0, {k: v for k, v in json.load(open(prev)).items() if k in SPACE})
    if a.regime == "sparse":
        base.append(CHANNEL_DENSE)
    for b in base:
        study.enqueue_trial({k: float(np.clip(b[k], *SPACE[k])) for k in SPACE})
    per = int(np.ceil(a.trials / a.workers))
    print(f"search {a.regime}: {a.workers} workers x {per} trials, budget {a.budget}", flush=True)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", "--regime", a.regime, "--budget", a.budget,
                               "--n_per_worker", str(per)]) for _ in range(a.workers)]
    for p in procs:
        p.wait()
    study = optuna.load_study(study_name=f"pipe_{a.regime}", storage=storage)
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value)
    print(f"\n{len(done)} trials in {(time.time() - t0) / 60:.1f} min; best 5:")
    for t in done[:5]:
        print(f"  trial {t.number:3d} VAL {t.value:7.3f}  " + " ".join(f"{k}={v:.3g}" for k, v in t.params.items()))
    # ---- confirmation of the top 3 with 2 fresh seeds ---------------------------------------
    from multiprocessing import Pool
    top = done[:3]
    jobs = [(a.regime, a.budget, t.params, s) for t in top for s in (5001, 5002)]
    with Pool(min(a.workers, len(jobs))) as pool:
        scores = pool.map(_confirm_job, jobs)
    rows = []
    for i, t in enumerate(top):
        sc = [t.value, scores[2 * i], scores[2 * i + 1]]
        rows.append((float(np.mean(sc)), float(np.std(sc)), t))
        print(f"  confirm trial {t.number}: {np.mean(sc):.3f} ± {np.std(sc):.3f}  (seeds: {np.round(sc, 2)})")
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    out = dict(best[2].params)
    path = os.path.join(C.OUT_DIR, f"best_weights_{a.regime}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(C.OUT_DIR, f"search_summary_{a.regime}.json"), "w") as f:
        json.dump(dict(trials=[dict(n=t.number, value=t.value, params=t.params) for t in done],
                       confirm=[dict(n=r[2].number, mean=r[0], std=r[1]) for r in rows]), f, indent=1)
    print(f"\nwinner: trial {best[2].number}  confirmed VAL {best[0]:.3f} ± {best[1]:.3f}\nsaved {path}")


if __name__ == "__main__":
    main()
