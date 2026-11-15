"""
dns_check.py -- independent TRUTH check against channel-flow DNS (not against our own solver).

    python dns_check.py --download                 # try to fetch the DNS mean profiles
    python dns_check.py --models outputs/final_s0.pt   # compare reference solver (and PINN)

Why: the PINN is trained on SA-RANS data, so "PINN vs reference" only measures how well it learnt
the model. This script measures how well the whole chain (SA-RANS reference, and PINN) reproduces
real turbulent channel statistics in the fully developed region:
    Moser, Kim & Mansour (1999) Phys. Fluids 11:943  (Re_tau = 180, 395, 590)
    Lee & Moser (2015) J. Fluid Mech. 774:395        (Re_tau = 550, 1000, 2000, 5200)

DNS file format (both databases): '%' or '#' header lines, then columns
    y/h   y+   U+   dU+/dy ...      (U+ = U/u_tau).
From the file alone we obtain Re_tau (header 'Re_tau' or --retau), U_b+ = int_0^1 U+ dy, hence
    Re_h = Re_tau * U_b+,    Re (this project) = 4 Re_h.
No DNS number is typed in by hand.

If --download fails (the URLs below could not be verified from my sandbox), download the
'*.means' / 'LM_Channel_*_mean_prof.dat' files by hand from
    https://turbulence.oden.utexas.edu/   (MKM 1999 and 'Channel Data (2015)')
and save them in dns_data/ as chan180.means, chan395.means, chan590.means, LM_1000.dat ...
Any file in dns_data/ is picked up; pass --retau if the header has no 'Re_tau'.
"""
import argparse
import glob
import os
import re
import urllib.request

import numpy as np
import torch

import config as C
import data as D
import reference_solver as R
import train as T

URLS = {
    "chan180.means": ["https://turbulence.oden.utexas.edu/data/MKM/chan180/profiles/chan180.means",
                      "https://torroja.dmt.upm.es/turbdata/agard/chapter5/PCH10/chan180/profiles/chan180.means"],
    "chan395.means": ["https://turbulence.oden.utexas.edu/data/MKM/chan395/profiles/chan395.means",
                      "https://torroja.dmt.upm.es/turbdata/agard/chapter5/PCH10/chan395/profiles/chan395.means"],
    "chan590.means": ["https://turbulence.oden.utexas.edu/data/MKM/chan590/profiles/chan590.means",
                      "https://torroja.dmt.upm.es/turbdata/agard/chapter5/PCH10/chan590/profiles/chan590.means"],
}


def download():
    for name, urls in URLS.items():
        dest = os.path.join(C.DNS_DIR, name)
        if os.path.exists(dest):
            print("have", name)
            continue
        for u in urls:
            try:
                urllib.request.urlretrieve(u, dest)
                print("downloaded", name, "from", u)
                break
            except Exception as e:                                  # noqa
                print("  failed", u, "->", type(e).__name__)
        else:
            print(f"!! could not fetch {name}; download manually (see module docstring)")


def read_dns(path, retau=None):
    header = []
    rows = []
    for ln in open(path, errors="ignore"):
        if ln.strip().startswith(("%", "#")):
            header.append(ln)
            continue
        p = ln.split()
        if len(p) >= 3:
            try:
                rows.append([float(v) for v in p[:3]])
            except ValueError:
                pass
    a = np.array(rows)
    y, yp, up = a[:, 0], a[:, 1], a[:, 2]
    if y[0] > y[-1]:
        y, yp, up = y[::-1], yp[::-1], up[::-1]
    if retau is None:
        # Re_tau = y+/y (definition of y+). Robust: do NOT trust header text -- the MKM headers
        # mention several Re_tau values and a regex picked the wrong one.
        k = (y > 0.05) & (y < 0.9)
        retau = float(np.median(yp[k] / y[k]))
    m = y <= 1.0
    y, yp, up = y[m], yp[m], up[m]
    if y[0] > 0:
        y, up = np.r_[0.0, y], np.r_[0.0, up]
    # bulk velocity over [0,1]; if the centreline is missing, extend flat from the last point
    yy = np.r_[y, 1.0] if y[-1] < 1.0 else y
    uu = np.r_[up, up[-1]] if y[-1] < 1.0 else up
    ubp = np.trapezoid(uu, yy)
    return dict(retau=retau, y=y, up=up, ubp=ubp, Reh=retau * ubp, Re=4.0 * retau * ubp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--retau", type=float, default=None)
    a = ap.parse_args()
    if a.download:
        download()
    files = sorted(glob.glob(os.path.join(C.DNS_DIR, "*")))
    if not files:
        print("No DNS files in dns_data/. Run with --download or place them manually (see docstring).")
        return
    device, dtype = T.get_device(), torch.float32
    models = [T.load_model(p, device)[0] for p in a.models]
    print(f"{'file':22s} {'Re_tau':>7} {'Ub+':>6} {'Re(Dh)':>9} | {'ref Retau':>9} {'ref Cf err%':>11} {'ref U relL2%':>12}" +
          ("  PINN U relL2% (vs DNS)" if models else ""))
    for f in files:
        try:
            d = read_dns(f, a.retau)
        except Exception as e:                                     # noqa
            print("skip", os.path.basename(f), e)
            continue
        cf_dns = 2.0 / d["ubp"] ** 2
        sol = R.solve(d["Re"], nx=300, N=300, L=C.L_OVER_H)
        U_ref = np.interp(d["y"], sol["y"], sol["U"][-1])
        U_dns = d["up"] / d["ubp"]                                 # U/U_b
        e_ref = 100 * np.sqrt(np.mean((U_ref - U_dns) ** 2))
        cf_err = 100 * (sol["Cf"][-1] / cf_dns - 1.0)
        line = f"{os.path.basename(f):22s} {d['retau']:7.1f} {d['ubp']:6.2f} {d['Re']:9.0f} | {sol['Retau'][-1]:9.1f} {cf_err:11.2f} {e_ref:12.2f}"
        if models:
            errs = []
            for m in models:
                y = torch.tensor(d["y"][1:], device=device, dtype=dtype)[:, None]
                x = torch.full_like(y, C.L_OVER_H)
                Rt = torch.full_like(y, d["Re"])
                with torch.no_grad():
                    U = m(x, y, Rt)[0].cpu().numpy().ravel()
                errs.append(100 * np.sqrt(np.mean((U - U_dns[1:]) ** 2)))
            line += f"  {np.mean(errs):6.2f}+-{np.std(errs):.2f}"
            inside = C.RE_MIN <= d["Re"] <= C.RE_MAX
            line += "  (inside train Re range)" if inside else "  (OUTSIDE train Re range: extrapolation)"
        print(line)
    print("\nRead: 'ref ... err' is the SA-RANS modelling error (a floor the PINN cannot beat when trained on it);\n"
          "the PINN column is total error vs real turbulence = PINN error + SA model error.")


if __name__ == "__main__":
    main()