"""
region_report.py -- WHERE are the errors?  Reads the metrics CSVs that evaluate.py already wrote (no training).

    python region_report.py --sparse        # reads outputs_sparse/
    python region_report.py                 # reads outputs/

Prints, for the VAL Reynolds numbers (mean over Re), error by REGION and FIELD for the 3-seed final model
('val') and for every ablation / sweep variant found (abl_*). Regions: upstream of the data (x<3h),
the three held-out streamwise bands (between data stations), inlet zone, fully developed zone.
"""
import argparse
import glob
import os
import sys

if "--sparse" in sys.argv:
    os.environ["PINN_SPARSE"] = "1"

import pandas as pd

import config as C

REGIONS = ["x<3h(upstream of data)", "x-holdout-bands", "inlet(x<10h)", "fully-dev(x>=0.9L)", "all"]
FIELDS = ["u", "p", "cf", "v"]

if __name__ == "__main__":
    argparse.ArgumentParser().parse_known_args()
    files = sorted(glob.glob(os.path.join(C.OUT_DIR, "val_metrics.csv")) + glob.glob(os.path.join(C.OUT_DIR, "abl_*_metrics.csv")))
    if not files:
        sys.exit(f"no *_metrics.csv in {C.OUT_DIR}")
    for role in ("VAL",):
        print(f"\nrelL2 %, role={role}, mean over {role} Re   (cf: relative error of wall shear, x>=1h)")
        print(f"{'variant':22s} {'region':26s} " + " ".join(f"{f:>7s}" for f in FIELDS))
        for fp in files:
            tag = os.path.basename(fp).replace("_metrics.csv", "")
            m = pd.read_csv(fp)
            m = m[(m.role == role) & (m.field.isin(FIELDS))]
            for rg in REGIONS:
                s = m[m.region == rg].groupby("field")["mean"].mean()
                if len(s):
                    print(f"{tag:22s} {rg:26s} " + " ".join(f"{s.get(f, float('nan')):7.2f}" for f in FIELDS))
            print()