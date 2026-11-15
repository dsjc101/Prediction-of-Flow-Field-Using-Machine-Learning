"""
run_all.py -- one entry point for the whole pipeline (mirrors the turbulent-channel report, Sec. 12).

  python run_all.py                          DENSE: selftest -> reference -> verify -> search -> final(3 seeds) -> evaluate(TRAIN+VAL)
  python run_all.py --sparse                 SPARSE: search -> ablation -> sweep -> final -> evaluate
  python run_all.py --stage reference        a single stage (reference | verify | selftest | search | ablation | sweep | final | evaluate | test)
  python run_all.py --sparse --stage test    TEST Re (28k, 96k, 120k): run ONCE per model
  python run_all.py --quick                  tiny budgets (~minutes) just to check that everything runs

Common flags:  --workers N (parallel processes, default = CPU cores // 2)   --budget NAME   --physics_factor F
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import config as C

PY = sys.executable
HERE = C.HERE


def sh(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def _train_job(args):
    (regime, weights, budget, seed, factor, variant, tag, depth, width, threads, device) = args
    import torch, train as T
    torch.set_num_threads(threads)
    m, info = T.train_model(regime, weights, budget, seed, factor, variant, depth, width, device,
                            label=f"{regime}/{tag}/s{seed}", verbose=(threads is not None))
    T.save_model(m, info, os.path.join(T.model_dir(regime), f"{tag}_seed{seed}.pt"))
    return tag, seed, info["val_score"], info["seconds"]


def parallel_train(jobs, workers, device="cpu"):
    from multiprocessing import Pool
    threads = max(1, (os.cpu_count() or 2) // max(1, min(workers, len(jobs))))
    jobs = [j + (threads, device) for j in jobs]
    if workers <= 1 or len(jobs) == 1:
        return [_train_job(j) for j in jobs]
    with Pool(min(workers, len(jobs)), maxtasksperchild=1) as p:
        return p.map(_train_job, jobs, chunksize=1)


def best_factor(regime, default):
    p = os.path.join(C.OUT_DIR, f"sweep_{regime}.json")
    if os.path.exists(p):
        return json.load(open(p))["best_factor"]
    return default


def stage_final(a):
    import train as T
    regime = "sparse" if a.sparse else "dense"
    w = T.load_weights(regime)
    f = a.physics_factor if a.physics_factor is not None else (best_factor(regime, 100.0) if a.sparse else 1.0)
    depth, width = (6, 128) if a.big else (C.NET_DEPTH, C.NET_WIDTH)
    print(f"FINAL {regime}: physics factor {f}, budget {a.budget}, seeds {a.seeds}")
    jobs = [(regime, w, a.budget, s, f, "searched", "final", depth, width) for s in a.seeds]
    for tag, seed, sc, sec in parallel_train(jobs, a.workers, a.device):
        print(f"  final seed {seed}: VAL {sc:.3f} ({sec:.0f}s)")


def _val_table(files, regime, title):
    import evaluate as E
    agg = E.evaluate_models(files, C.RE_VAL, verbose=False)
    lines = [title, f"{'Re':>8} {'U':>8} {'P_hat':>8} {'Cf':>8} {'PDE res (x-mom)':>16}"]
    sc = []
    for Re in C.RE_VAL:
        a = agg[Re]
        lines.append(f"{Re:>8} {a['U'][0]:8.2f} {a['P'][0]:8.2f} {a['Cf'][0]:8.2f} {a['res_momx'][0]:16.2f}")
        sc.append((a['U'][0] + a['P'][0] + a['Cf'][0]) / 3)
    head = float(np.mean(sc))
    resm = float(np.mean([agg[Re]['res_momx'][0] for Re in C.RE_VAL]))
    lines.append(f"  headline (mean of [U+P+Cf]/3) = {head:.2f}   mean PDE residual = {resm:.2f}")
    return "\n".join(lines), head, resm, agg


def stage_ablation(a):
    import train as T
    regime = "sparse" if a.sparse else "dense"
    w = T.load_weights(regime)
    variants = [("ablA", "A searched weights", 1.0, "searched"), ("ablB", "B data only", 1.0, "data_only"),
                ("ablC", "C physics x100", 100.0, "searched")]
    jobs = [(regime, w, a.budget, 0, f, v, tag, C.NET_DEPTH, C.NET_WIDTH) for tag, _, f, v in variants]
    parallel_train(jobs, a.workers, a.device)
    txt = []
    for tag, name, f, v in variants:
        files = [os.path.join(T.model_dir(regime), f"{tag}_seed0.pt")]
        t, head, resm, _ = _val_table(files, regime, f"\n[{regime}] ablation {name}  (VAL Re, single seed)")
        print(t); txt.append(t)
    open(os.path.join(C.OUT_DIR, f"ablation_{regime}.txt"), "w").write("\n".join(txt))


def stage_sweep(a):
    import train as T
    regime = "sparse" if a.sparse else "dense"
    w = T.load_weights(regime)
    factors = [1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0]
    jobs = [(regime, w, a.budget, 0, f, "searched", f"sweep_f{int(f)}", C.NET_DEPTH, C.NET_WIDTH) for f in factors]
    parallel_train(jobs, a.workers, a.device)
    rows, txt = [], [f"\n[{regime}] physics-scale sweep (mom, sa, cont multiplied by f), VAL, single seed",
                     f"{'f':>8} {'headline':>9} {'U':>7} {'P':>7} {'Cf':>7} {'PDE res':>9}"]
    import evaluate as E
    for f in factors:
        files = [os.path.join(T.model_dir(regime), f"sweep_f{int(f)}_seed0.pt")]
        _, head, resm, agg = _val_table(files, regime, "")
        U = np.mean([agg[Re]["U"][0] for Re in C.RE_VAL]); Pp = np.mean([agg[Re]["P"][0] for Re in C.RE_VAL])
        Cf = np.mean([agg[Re]["Cf"][0] for Re in C.RE_VAL])
        rows.append(dict(f=f, headline=head, U=U, P=Pp, Cf=Cf, res=resm))
        txt.append(f"{f:>8.0f} {head:9.2f} {U:7.2f} {Pp:7.2f} {Cf:7.2f} {resm:9.2f}")
    best = min(rows, key=lambda r: r["headline"])
    txt.append(f"best factor on VAL: {best['f']:.0f}")
    print("\n".join(txt))
    open(os.path.join(C.OUT_DIR, f"sweep_{regime}.txt"), "w").write("\n".join(txt))
    json.dump(dict(rows=rows, best_factor=best["f"]), open(os.path.join(C.OUT_DIR, f"sweep_{regime}.json"), "w"), indent=1, default=float)


def stage_evaluate(a, roles=("train", "val"), final=False):
    regime = "sparse" if a.sparse else "dense"
    cmd = [PY, "evaluate.py", "--regime", regime, "--tag", "final"]
    cmd += ["--final"] if final else ["--roles", *roles]
    sh(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "selftest", "reference", "verify", "search", "ablation",
                                                         "sweep", "final", "evaluate", "test", "truth"])
    ap.add_argument("--budget", default=None, choices=list(C.BUDGETS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--physics_factor", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke budgets everywhere (minutes)")
    a = ap.parse_args()
    regime = "sparse" if a.sparse else "dense"
    search_budget = "smoke" if a.quick else "cpu_search"
    if a.budget is None:
        a.budget = "smoke" if a.quick else "cpu_final"
    if a.quick:
        a.trials = min(a.trials, 6)
    t0 = time.time()
    st = a.stage
    dense_only = not a.sparse
    if st in ("all", "reference") and dense_only:
        sh([PY, "reference_solver.py", "--workers", str(a.workers)])
    if st in ("all", "selftest") and dense_only:
        sh([PY, "selftest.py"])
    if st in ("all", "verify") and dense_only:
        sh([PY, "verify_reference.py"])
    if st in ("all", "search"):
        sh([PY, "search_weights.py", "--regime", regime, "--trials", str(a.trials), "--workers", str(a.workers),
            "--budget", search_budget])
    if (st == "all" and a.sparse) or st == "ablation":
        stage_ablation(a)
    if (st == "all" and a.sparse) or st == "sweep":
        stage_sweep(a)
    if st in ("all", "final"):
        stage_final(a)
    if st in ("all", "evaluate"):
        stage_evaluate(a)
    if st == "test":
        stage_evaluate(a, final=True)
    if st == "truth":
        sh([PY, "truth_check.py", "--download"])
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
