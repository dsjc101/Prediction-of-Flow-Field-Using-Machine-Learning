"""
run_all.py -- ONE command for the whole pipeline (works on Windows, no shell loops).

    python run_all.py                       # reference -> verify -> search -> finalize -> 3 final seeds -> evaluate
    python run_all.py --stage search        # run just one stage (stages are resumable / idempotent)
    python run_all.py --stage test          # TEST Re + DNS check. Run ONCE, at the very end.

Stages: reference | search | finalize | final | evaluate | test
Everything long-running writes to outputs/logs/*.log; progress is printed here every 60 s.
CPU defaults: 4x64 network, budgets cpu_search / cpu_final. On a GPU the bigger presets are used.
Options: --workers N (parallel search workers, 1 thread each), --trials_per_worker K,
         --search_budget / --final_budget (see train.BUDGETS, e.g. cpu_long for a longer final run).
"""
import argparse
import os
import subprocess
import sys
import time

if "--sparse" in sys.argv:                 # must be set BEFORE config is imported (children inherit it)
    os.environ["PINN_SPARSE"] = "1"

import config as C

PY = sys.executable
LOG = os.path.join(C.OUT_DIR, "logs")
os.makedirs(LOG, exist_ok=True)


def run(cmd, log=None):
    print("$", " ".join(cmd), flush=True)
    if log is None:
        r = subprocess.run(cmd)
    else:
        with open(os.path.join(LOG, log), "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)} (see outputs/logs/{log})" if log else f"FAILED: {' '.join(cmd)}")


def wait_all(procs, label, status=None):
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        msg = status() if status else ""
        print(f"  [{label}] {int((time.time()-t0)/60)} min  {msg}", flush=True)
    bad = [p for p in procs if p.returncode != 0]
    if bad:
        sys.exit(f"{len(bad)} {label} process(es) failed -- see outputs/logs/")


def stage_reference(a):
    run([PY, "build_reference.py"])
    run([PY, "verify_reference.py"], log="verify_reference.log")
    print(open(os.path.join(LOG, "verify_reference.log")).read())


def stage_search(a):
    import shutil
    import optuna
    import train as T
    if C.SPARSE:                             # seed the sparse search with the dense-data winner (re-scored as a trial)
        src, dst = os.path.join(C.ROOT, "outputs", "best_weights.json"), os.path.join(C.OUT_DIR, "best_weights.json")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    budget = a.search_budget or T.default_budget("search")
    first = [PY, "weight_search.py", "--trials", str(a.trials_per_worker), "--budget", budget,
             "--threads", "1", "--search_only"]
    # worker 0 deletes any stale study (unless --resume_search), then the others join the fresh one
    w0 = first + ([] if a.resume_search else ["--fresh"])
    procs = [subprocess.Popen(w0, stdout=open(os.path.join(LOG, "search_w0.log"), "w"), stderr=subprocess.STDOUT)]
    time.sleep(20)                                   # worker 0 creates the study + enqueues the baselines
    for k in range(1, a.workers):
        procs.append(subprocess.Popen(first, stdout=open(os.path.join(LOG, f"search_w{k}.log"), "w"),
                                      stderr=subprocess.STDOUT))
        time.sleep(2)

    def status():
        try:
            s = optuna.load_study(study_name="weights", storage=f"sqlite:///{os.path.join(C.OUT_DIR, 'weight_search.db')}")
            done = [t for t in s.trials if t.state.name == "COMPLETE"]
            pr = [t for t in s.trials if t.state.name == "PRUNED"]
            best = min((t.value for t in done), default=float("nan"))
            return f"trials: {len(done)} complete, {len(pr)} pruned, best VAL={best:.2f}%"
        except Exception:                                                 # noqa
            return ""
    wait_all(procs, "search", status)
    print(status())


def stage_finalize(a):
    import train as T
    budget = a.search_budget or T.default_budget("search")
    run([PY, "weight_search.py", "--finalize", "--budget", budget, "--threads", str(max(1, a.workers))],
        log="finalize.log")
    print(open(os.path.join(LOG, "finalize.log")).read()[-1800:])


def stage_final(a):
    import train as T
    budget = a.final_budget or T.default_budget("final")
    thr = max(1, (os.cpu_count() or 4) // 3)
    wpath = os.path.join(C.OUT_DIR, "best_weights.json")
    if a.physics_factor != 1.0:                      # scale mom, sa, cont together (chosen from the sweep)
        import json
        bw = json.load(open(wpath))
        for k in ("mom", "sa", "cont"):
            bw["weights"][k] *= a.physics_factor
        wpath = os.path.join(C.OUT_DIR, f"best_weights_physx{a.physics_factor:g}.json")
        json.dump(bw, open(wpath, "w"), indent=1)
        print(f"using physics factor x{a.physics_factor:g} -> {wpath}")
    procs = [subprocess.Popen([PY, "train.py", "--weights", wpath,
                               "--budget", budget, "--seed", str(s), "--name", f"final_s{s}", "--threads", str(thr)],
                              stdout=open(os.path.join(LOG, f"final_s{s}.log"), "w"), stderr=subprocess.STDOUT)
             for s in range(3)]
    def last():
        try:
            return open(os.path.join(LOG, "final_s0.log")).read().strip().splitlines()[-1][:150]
        except Exception:                                                 # noqa
            return ""
    wait_all(procs, "final training", last)


def models():
    return [os.path.join(C.OUT_DIR, f"final_s{s}.pt") for s in range(3)]


def stage_evaluate(a):
    run([PY, "evaluate.py", "--models", *models(), "--tag", "val"], log="evaluate.log")
    print(open(os.path.join(LOG, "evaluate.log")).read())


def stage_test(a):
    run([PY, "evaluate.py", "--models", *models(), "--final", "--tag", "final"], log="evaluate_final.log")
    print(open(os.path.join(LOG, "evaluate_final.log")).read())
    run([PY, "dns_check.py", "--models", *models()], log="dns_check.log")
    print(open(os.path.join(LOG, "dns_check.log")).read())


def stage_sweep(a):
    """Physics-scale sweep at the FINAL budget (single seed). The weight search runs at a short budget, which
    biases it against slowly-converging physics terms; this checks the one weight group that matters.
    Reuses abl_A_found (x1) and abl_C_physics_x100 (x100) from the ablation stage; trains the other factors."""
    import json
    import pandas as pd
    import train as T
    budget = a.final_budget or T.default_budget("final")
    bw = json.load(open(os.path.join(C.OUT_DIR, "best_weights.json")))
    W = bw["weights"]
    new = [f for f in a.factors if f not in (1.0, 100.0)]
    thr = max(1, (os.cpu_count() or 4) // max(1, len(new)))
    procs = []
    for f in new:
        jp = os.path.join(C.OUT_DIR, f"abl_P{f:g}.json")
        json.dump(dict(weights=dict(W, mom=W["mom"] * f, sa=W["sa"] * f, cont=W["cont"] * f), lr=bw["lr"]), open(jp, "w"))
        procs.append(subprocess.Popen([PY, "train.py", "--weights", jp, "--budget", budget, "--seed", "0",
                                       "--name", f"abl_P{f:g}", "--threads", str(thr)],
                                      stdout=open(os.path.join(LOG, f"abl_P{f:g}.log"), "w"), stderr=subprocess.STDOUT))
    if procs:
        wait_all(procs, "sweep training")
    variants = {1.0: "abl_A_found", 100.0: "abl_C_physics_x100"}
    variants.update({f: f"abl_P{f:g}" for f in new})
    rows = []
    for f in sorted(variants):
        name = variants[f]
        if not os.path.exists(os.path.join(C.OUT_DIR, f"{name}_metrics.csv")):
            run([PY, "evaluate.py", "--models", os.path.join(C.OUT_DIR, f"{name}.pt"), "--tag", name], log=f"{name}_eval.log")
        m = pd.read_csv(os.path.join(C.OUT_DIR, f"{name}_metrics.csv"))
        v = m[(m.role == "VAL") & (m.region == "all") & (m.field.isin(["u", "p", "cf"]))].pivot_table(index="Re", columns="field", values="mean")
        pr = m[(m.role == "VAL") & (m.region == "fresh-PDE-residual") & (m.field == "mom_x")].set_index("Re")["mean"]
        rows.append(dict(factor=f"x{f:g}", headline=float(v.mean(axis=1).mean()), u=float(v["u"].mean()), p=float(v["p"].mean()),
                         cf=float(v["cf"].mean()), PDE_res=float(pr.mean())))
    tab = pd.DataFrame(rows).set_index("factor")
    txt = ("PHYSICS-SCALE SWEEP (mom, sa, cont multiplied together; VAL Re, mean over the 3 VAL Re; single seed)\n"
           "headline = mean(u,p,cf) relL2 %\n" + tab.round(3).to_string())
    print(txt)
    open(os.path.join(C.OUT_DIR, "sweep_summary.txt"), "w").write(txt + "\n")


def stage_ablation(a):
    """Does the physics do anything?  Three variants, same seed/budget/net, scored on VAL only:
       A = found weights   B = DATA-ONLY (all physics + BC weights 0)   C = physics x100 (mom, sa, cont)
    If B ~ A on the extrapolation VAL case (88k) and in PDE residual, the model is a data-fit surrogate and
    should not be called physics-informed; if A << B there, physics is earning its keep."""
    import json
    import pandas as pd
    import train as T
    budget = a.final_budget or T.default_budget("final")
    bw = json.load(open(os.path.join(C.OUT_DIR, "best_weights.json")))
    W = bw["weights"]
    variants = {
        "A_found": dict(W),
        "B_data_only": dict(W, cont=0.0, mom=0.0, sa=0.0, inlet=0.0, outlet=0.0, sym=0.0),
        "C_physics_x100": dict(W, mom=W["mom"] * 100, sa=W["sa"] * 100, cont=W["cont"] * 100),
    }
    thr = max(1, (os.cpu_count() or 4) // 3)
    procs = []
    for name, w in variants.items():
        jp = os.path.join(C.OUT_DIR, f"abl_{name}.json")
        json.dump(dict(weights=w, lr=bw["lr"]), open(jp, "w"))
        procs.append(subprocess.Popen([PY, "train.py", "--weights", jp, "--budget", budget, "--seed", "0",
                                       "--name", f"abl_{name}", "--threads", str(thr)],
                                      stdout=open(os.path.join(LOG, f"abl_{name}.log"), "w"), stderr=subprocess.STDOUT))
    wait_all(procs, "ablation training")
    rows = []
    for name in variants:
        run([PY, "evaluate.py", "--models", os.path.join(C.OUT_DIR, f"abl_{name}.pt"), "--tag", f"abl_{name}"],
            log=f"abl_{name}_eval.log")
        m = pd.read_csv(os.path.join(C.OUT_DIR, f"abl_{name}_metrics.csv"))
        for _, x in m[(m.role == "VAL") & (m.region == "all") & (m.field.isin(["u", "p", "cf"]))].iterrows():
            rows.append(dict(variant=name, Re=int(x.Re), field=x.field, err=x["mean"]))
        fr = m[(m.role == "VAL") & (m.region == "fresh-PDE-residual") & (m.field == "mom_x")]
        for _, x in fr.iterrows():
            rows.append(dict(variant=name, Re=int(x.Re), field="PDE-res", err=x["mean"]))
    tab = pd.DataFrame(rows).pivot_table(index="variant", columns=["Re", "field"], values="err")
    txt = "ABLATION (VAL Re only; relL2 % for u,p,cf; PDE-res = rms momentum residual / u_tau^2)\n" + tab.round(3).to_string()
    print(txt)
    open(os.path.join(C.OUT_DIR, "ablation_summary.txt"), "w").write(txt + "\n")


STAGES = ["reference", "search", "finalize", "final", "evaluate"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES + ["ablation", "sweep", "test"])
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--trials_per_worker", type=int, default=8)
    ap.add_argument("--sparse", action="store_true", help="sparse-data mode (profiles at a few x-stations); outputs_sparse/")
    ap.add_argument("--resume_search", action="store_true", help="keep the existing Optuna study and add trials")
    ap.add_argument("--factors", type=float, nargs="*", default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    ap.add_argument("--physics_factor", type=float, default=1.0, help="multiply mom, sa, cont weights in the final training")
    ap.add_argument("--search_budget", default=None)
    ap.add_argument("--final_budget", default=None)
    a = ap.parse_args()
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        print(f"\n===== stage: {s} =====", flush=True)
        globals()["stage_" + s](a)
    if a.stage == "all":
        print("\nDone. Send me: outputs/val_summary.txt, outputs/val_profiles_cf.png, outputs/best_weights.json,"
              " outputs/weight_search_trials.csv.\nTEST Re are still untouched: run  python run_all.py --stage test  once we have looked at these.")