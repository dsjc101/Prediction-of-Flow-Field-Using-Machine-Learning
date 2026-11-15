"""
truth_check.py -- errors against REAL pipe data (experiment / DNS), not against the SA reference.

Error chain reported:   PINN --(1)--> SA reference --(2)--> real flow          (1) = evaluate.py, (2) = here
Real data used:
  * Princeton Superpipe mean-velocity profiles (Zagarola & Smits 1998; McKeon et al. 2004), fully developed,
    Re_b = U_b D / nu = 31.6k ... 146k in the range of this project (same Re definition as config.py).
    Each file has a header (Re, Karman number = Re_tau, friction factor lambda, U_avg, U_max, ...) and 4 columns:
    y/R, y+, U+, Ucl+ - U+ .
  * OPTIONAL DNS / any other benchmark: drop a CSV in data/exp/  with header comments
        # Re_b=44000  Re_tau=1136.6
    and two columns   y_over_R (distance from wall / R),  U_plus        (e.g. Pirozzoli et al. 2021 pipe DNS:
    Re_b = 44 000 (Re_tau 1137), 82 500 (1976), 133 000 (3028)).
  * The Superpipe/McKeon (2005) friction law at EXACTLY the 10 project Re (no extra data needed).
The benchmark Re are NOT training Re: the SA solver is run at each benchmark Re, and the trained PINNs are
queried there at x = L (fully developed; development length <= 0.53 L).  Beyond Re = 80k this is extrapolation.

  python truth_check.py --download                 fetch the Superpipe files into data/exp/  (needs internet)
  python truth_check.py                            benchmarks + friction-law table (TRAIN/VAL Re)
  python truth_check.py --final                    also the TEST Re in the friction-law table (do once)
"""
import argparse, glob, json, os, re, urllib.request
import numpy as np
import torch
import config as C
import pinn as P
import reference_solver as R
import train as T

EXP_DIR = os.path.join(C.DATA_DIR, "exp")
os.makedirs(EXP_DIR, exist_ok=True)

ZAG = ["3.1577E+04", "4.1727E+04", "5.6677E+04", "7.4293E+04", "9.8811E+04", "1.4579E+05"]
MCK = {"7.4345E+04": "4af", "1.4458E+05": "6af"}
URLS = ([(f"zag_{k}.txt", [f"http://www.princeton.edu/~gasdyn/Superpipe_data/{k}.txt",
                           f"https://www.princeton.edu/~gasdyn/Superpipe_data/{k}.txt"]) for k in ZAG]
        + [(f"mck_{k}.txt", [f"https://smits.princeton.edu/wp-content/uploads/sites/356/2016/03/{v}.txt"])
           for k, v in MCK.items()])


def download():
    hdr = {"User-Agent": "Mozilla/5.0"}
    for name, urls in URLS:
        path = os.path.join(EXP_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 500:
            print("have", name); continue
        for u in urls:
            try:
                req = urllib.request.Request(u, headers=hdr)
                with urllib.request.urlopen(req, timeout=30) as r:
                    txt = r.read().decode("latin-1")
                if sum(1 for l in txt.splitlines() if len(l.split()) == 4) > 20:
                    open(path, "w").write(txt); print("downloaded", name); break
            except Exception as e:
                print(f"  {u}: {e}")
        else:
            print(f"!! could not fetch {name}; download it manually into {EXP_DIR}")


# ----------------------------------------------------------------------------------
def _floats(line):
    out = []
    for t in line.replace(",", " ").split():
        try: out.append(float(t))
        except ValueError: return None
    return out


NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_file(path):
    """returns dict(Re, Retau, f, y (y/R from the wall), Up (U+), name) or None.
    Header lines are recognised by keywords (reynolds / re_b, karman / re_tau, friction factor / f=)."""
    r4, r2, r5, hdr, zag = [], [], [], {}, False
    for line in open(path, encoding="latin-1").read().splitlines():
        v = _floats(line)
        if v is not None and len(v) == 5:
            # McKeon files: index, y/R, y+, U+, Ucl+-U+   |   Zagarola files: ys/R, rs/R, udef, y+, u+
            r5.append([v[0], v[3], v[4]] if zag else v[1:]); continue
        if "ys/r" in line.lower():
            zag = True; continue
        if v is not None and len(v) == 4:
            r4.append(v); continue
        if v is not None and len(v) == 2:
            r2.append(v); continue
        low = line.lower()
        nums = re.findall(NUM, line)
        if not nums:
            continue
        if "re_tau" in low:
            hdr.setdefault("Retau", float(re.findall(r"re_tau\s*[=:]\s*(" + NUM + ")", low)[0]))
        elif "karman" in low:
            hdr.setdefault("Retau", float(nums[-1]))
        if "re_b" in low:
            hdr.setdefault("Re", float(re.findall(r"re_b\s*[=:]\s*(" + NUM + ")", low)[0]))
        elif "reynolds" in low:
            hdr.setdefault("Re", float(nums[-1]))
        if "friction factor" in low or re.search(r"(^|[#\s])f\s*=", low):
            hdr.setdefault("f", float(nums[-1]))
    best = max([r5, r4, r2], key=len)
    A = np.array(best) if best else np.zeros((0, 4))
    if len(A) < 15 or "Re" not in hdr:
        return None
    if best is r5 and zag:
        y, Up = A[:, 0], A[:, 2]                     # ys/R (from the wall), u+
    elif A.shape[1] == 4:
        y, Up = A[:, 0], A[:, 2]
    else:
        y, Up = A[:, 0], A[:, 1]
    m = (y > 0) & (y <= 1.0)
    y, Up = y[m], Up[m]
    o = np.argsort(y); y, Up = y[o], Up[o]
    Re, Retau, f, note = hdr["Re"], hdr.get("Retau"), hdr.get("f"), ""
    if zag and Retau is not None:
        Retau = Retau / 2.0          # Zagarola header "Karman number" is based on the DIAMETER; y+ column is based on R
    if f is None and Retau is not None:       # lambda = 8 (u_tau/U_b)^2 ,  u_tau/U_b = 2 Re_tau / Re_b
        f = 8.0 * (2.0 * Retau / Re) ** 2
    if f is None:
        f = float(C.mckeon_lambda(Re)); note = " (f from McKeon fit)"
    if Retau is None:
        Retau = np.sqrt(f / 8.0) * Re / 2.0
    return dict(name=os.path.basename(path), Re=Re, Retau=Retau, f=f, y=y, Up=Up, note=note)


def load_benchmarks(re_lo=2.5e4, re_hi=1.6e5):
    out = []
    for p in sorted(glob.glob(os.path.join(EXP_DIR, "*.txt")) + glob.glob(os.path.join(EXP_DIR, "*.csv"))):
        b = parse_file(p)
        if b and re_lo <= b["Re"] <= re_hi:
            out.append(b)
        elif b is None:
            print(f"(could not parse {os.path.basename(p)} -- skipped)")
    return sorted(out, key=lambda b: b["Re"])


# ----------------------------------------------------------------------------------
def ensure_ref(Re):
    path = C.ref_path(Re)
    if not os.path.exists(path):
        print(f"  running SA reference solver at Re = {Re:.0f} ...", flush=True)
        np.savez_compressed(path, **R.Pipe(Re).march())
    return dict(np.load(path))


def profile_from_ref(ref, y_eval):
    y = np.concatenate(([0.0], ref["y"]))
    U = np.concatenate(([0.0], ref["U"][-1]))
    return np.interp(y_eval, y, U), ref["Cf"][-1]


def profile_pinn(model, Re, y_eval):
    r = torch.tensor(1.0 - y_eval, dtype=torch.float32)
    x = torch.full_like(r, C.L); Rr = torch.full_like(r, float(Re))
    U = P.predict(model, x, r, Rr)[:, 0].numpy()
    cf = P.wall_shear_cf(model, torch.tensor([C.L]), torch.tensor([float(Re)])).item()
    return U.astype(float), cf


_trapz = getattr(np, 'trapezoid', None) or np.trapz


def rl2(a, b):
    return 100 * np.sqrt(np.mean((a - b) ** 2))


def load_models(regime):
    fs = sorted(glob.glob(os.path.join(T.model_dir(regime), "final_seed*.pt")))
    return [T.load_model(f)[0] for f in fs]


def benchmark_table(models):
    bms = load_benchmarks()
    if not bms:
        raise SystemExit(f"no benchmark files in {EXP_DIR}: run  python truth_check.py --download  (or add files by hand)")
    rows, curves = [], []
    for b in bms:
        Re = b["Re"]
        Ub_plus = np.sqrt(8.0 / b["f"])
        Ue = b["Up"] / Ub_plus                       # U/U_b of the experiment (uses the experiment's own friction factor)
        Cf_e = b["f"] / 4.0
        # data self-consistency: 2 int U r dr should be 1 (profile extended to the wall with the linear law)
        yy = np.concatenate(([0.0], b["y"])); uu = np.concatenate(([0.0], Ue))
        yg = np.linspace(0, yy.max(), 4000); mass = 2 * _trapz(np.interp(yg, yy, uu) * (1 - yg), yg)
        ref = ensure_ref(Re)
        Us, Cfs = profile_from_ref(ref, b["y"])
        row = dict(name=b["name"], Re=Re, Retau_exp=b["Retau"], Retau_SA=float(np.sqrt(Cfs / 2) * Re / 2),
                   Cf_exp=Cf_e, Cf_SA=float(Cfs), dCf_SA=100 * (Cfs - Cf_e) / Cf_e,
                   Uc_exp=float(Ue.max()), Uc_SA=float(Us.max()), mass_exp=float(mass),
                   U_SA=rl2(Us, Ue), note=b["note"],
                   role="interp" if Re <= 8.0e4 else "extrap")
        cur = dict(Re=Re, y=b["y"], Up_exp=b["Up"], Ub_plus=Ub_plus, Retau=b["Retau"], SA=Us)
        for reg, ms in models.items():
            if not ms:
                continue
            pr = [profile_pinn(m, Re, b["y"]) for m in ms]
            Up_ = np.array([p[0] for p in pr]); cfp = np.array([p[1] for p in pr])
            eU = np.array([rl2(u, Ue) for u in Up_]); eC = 100 * (cfp - Cf_e) / Cf_e
            eUs = np.array([rl2(u, Us) for u in Up_])
            row[f"U_{reg}"] = (float(eU.mean()), float(eU.std()))
            row[f"U_{reg}_vsSA"] = float(eUs.mean())
            row[f"dCf_{reg}"] = (float(eC.mean()), float(eC.std()))
            cur[reg] = Up_.mean(0)
        rows.append(row); curves.append(cur)
    return rows, curves


def print_benchmarks(rows, regs):
    L_ = ["\n=== vs REAL data (fully developed, x = L).  U error = relL2 of U/U_b (percent of bulk velocity) ===",
          "   (U_SA = SA reference vs data = accuracy floor; U_dense/U_sparse = PINN vs data, mean ± std over seeds;",
          "    'vsSA' = PINN vs the SA reference at the same points)"]
    h = f"{'file':<20}{'Re':>8}{'Retau exp/SA':>15}{'Cf exp':>9}{'dCf SA%':>8}{'Uc/Ub exp/SA':>17}{'mass':>7}{'U_SA':>7}"
    for r in regs:
        h += f"{'U_' + r:>14}{'vsSA':>6}{'dCf_' + r + '%':>15}"
    h += "  role"
    L_.append(h)
    for r in rows:
        s = (f"{r['name']:<20}{r['Re']:8.0f}{r['Retau_exp']:8.0f}/{r['Retau_SA']:<6.0f}{r['Cf_exp']:9.5f}{r['dCf_SA']:8.2f}"
             f"{r['Uc_exp']:9.3f}/{r['Uc_SA']:<7.3f}{r['mass_exp']:7.3f}{r['U_SA']:7.2f}")
        for g in regs:
            if f"U_{g}" in r:
                u, c = r[f"U_{g}"], r[f"dCf_{g}"]
                s += f"{u[0]:8.2f}±{u[1]:<4.2f}{r[f'U_{g}_vsSA']:6.2f}{c[0]:9.2f}±{c[1]:<4.2f}"
        L_.append(s + "  " + r["role"] + r["note"])
    txt = "\n".join(L_); print(txt); return txt


def friction_law_table(models, final):
    res = ["\n=== wall friction vs the Superpipe/McKeon (2005) law at the project Re  (Cf error %, fully developed x = L) ==="]
    res.append(f"{'role':<18}{'Re':>8}{'Cf law':>9}{'SA solver':>10}" + "".join(f"{('PINN ' + r):>20}" for r in models))
    rows = []
    for Re in C.RE_ALL_SORTED:
        if C.ROLE[Re] == "TEST" and not final:
            continue
        cfl = float(C.mckeon_lambda(Re)) / 4.0
        cfs = float(R.Pipe(Re).march()["Cf"][-1]) if not os.path.exists(C.ref_path(Re)) else float(np.load(C.ref_path(Re))["Cf"][-1])
        s = f"{C.ROLE_NOTE[Re]:<18}{Re:8d}{cfl:9.5f}{100 * (cfs - cfl) / cfl:+10.2f}"
        row = dict(Re=Re, Cf_law=cfl, SA=100 * (cfs - cfl) / cfl)
        for reg, ms in models.items():
            e = np.array([100 * (P.wall_shear_cf(m, torch.tensor([C.L]), torch.tensor([float(Re)])).item() - cfl) / cfl for m in ms]) if ms else np.array([np.nan])
            s += f"{e.mean():+13.2f} ± {e.std():<4.2f}"
            row[reg] = (float(e.mean()), float(e.std()))
        res.append(s); rows.append(row)
    if not final:
        res.append("(TEST Re omitted; add --final to include them)")
    txt = "\n".join(res); print(txt); return txt, rows


def figure(curves, regs, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(curves); cols = min(3, n); rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(5 * cols, 3.8 * rws), squeeze=False)
    for ax, c in zip(axs.ravel(), curves):
        yp = c["y"] * c["Retau"]
        ax.semilogx(yp, c["Up_exp"], "ko", ms=3, label="real data")
        ax.semilogx(yp, c["SA"] * c["Ub_plus"], "-", color="tab:blue", label="SA reference")
        for reg, col in zip(regs, ["tab:orange", "tab:green"]):
            if reg in c:
                ax.semilogx(yp, c[reg] * c["Ub_plus"], "--", color=col, label=f"PINN {reg}")
        ax.set_title(f"Re_b = {c['Re']:.0f}  (Re_tau {c['Retau']:.0f})"); ax.set_xlabel("y+"); ax.set_ylabel("U+")
    axs[0][0].legend(fontsize=8)
    for ax in axs.ravel()[n:]: ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(C.OUT_DIR, "figure_truth_profiles.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--final", action="store_true", help="include TEST Re in the friction-law table")
    ap.add_argument("--no_pinn", action="store_true", help="SA-vs-data only (no trained models needed)")
    a = ap.parse_args()
    if a.download:
        download()
    models = {} if a.no_pinn else {r: load_models(r) for r in ("dense", "sparse")}
    models = {k: v for k, v in models.items() if v}
    rows, curves = benchmark_table(models)
    t1 = print_benchmarks(rows, list(models))
    t2, frows = friction_law_table(models, a.final)
    figure(curves, list(models), rows)
    open(os.path.join(C.OUT_DIR, "truth_check.txt"), "w").write(t1 + "\n" + t2)
    json.dump(dict(benchmarks=rows, friction_law=frows), open(os.path.join(C.OUT_DIR, "truth_check.json"), "w"), indent=1, default=float)
    print("\nsaved outputs/truth_check.txt/.json and figure_truth_profiles.png")


if __name__ == "__main__":
    main()
