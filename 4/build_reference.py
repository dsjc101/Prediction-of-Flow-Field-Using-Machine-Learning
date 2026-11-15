"""
build_reference.py -- run the SA-RANS marching solver for every Re in config.RE_ALL
and cache the result in reference_data/ref_Re<Re>.npz.

    python build_reference.py            # all Re, parallel over CPU cores
    python build_reference.py --force    # recompute even if cached

Takes ~1-3 min per Re on one core (9 Re). Re-running is free (cache).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config as C
import reference_solver as R


def ref_path(Re):
    return os.path.join(C.REF_DIR, f"ref_Re{int(Re)}.npz")


def _build(Re):
    t0 = time.time()
    s = R.solve(float(Re))
    np.savez_compressed(ref_path(Re), **{k: np.asarray(v) for k, v in s.items()})
    return Re, time.time() - t0, float(s["Cf"][-1]), float(s["Retau"][-1]), R.development_length(s)


def load_reference(Re, build_if_missing=True):
    p = ref_path(Re)
    if not os.path.exists(p):
        if not build_if_missing:
            raise FileNotFoundError(p)
        print(f"[reference] building Re={Re} ...")
        _build(Re)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    todo = [Re for Re in C.RE_ALL if a.force or not os.path.exists(ref_path(Re))]
    print(f"Re to build: {todo}  (workers={a.workers})")
    if not todo:
        print("all cached.")
        return
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for Re, dt, cf, rt, ld in ex.map(_build, todo):
            print(f"  Re={Re:>7}  {dt:6.1f}s   Cf_end={cf:.5f}  Re_tau_end={rt:7.1f}  L_dev(1%)={ld:6.1f} h")
    print("done ->", C.REF_DIR)


if __name__ == "__main__":
    main()
