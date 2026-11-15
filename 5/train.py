"""
train.py -- hybrid data + physics training of the pipe RANS+SA PINN.

  python train.py --regime dense  --budget cpu_final --seeds 0 1 2 --tag final
  python train.py --regime sparse --budget cpu_final --seeds 0 1 2 --tag final --physics_factor 100
  python train.py --regime sparse --variant data_only --seeds 0 --tag ablB      (ablation variants)

Adam (lr decayed 100x, grad-clip 10) -> optional L-BFGS polish (kept only if the VAL score improves);
early stopping on the VAL score (VAL Re only); rollback on non-finite loss/gradient.
TEST Re are never loaded here (guarded in data.py users: only evaluate.py --final touches them).
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch
import config as C
import pinn as P
import data as D

LOSS_KEYS = ["data", "cont", "momx", "momr", "sa", "inlet", "outlet", "sym"]


def pick_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def make_val_sets(dev):
    return {Re: D.val_set(Re, dev) for Re in C.RE_VAL}


@torch.no_grad()
def _pred(model, pts):
    return P.predict(model, pts["x"], pts["r"], pts["Re"])


def val_score(model, val_sets, detail=False):
    """mean over VAL Re of [relL2(U) + relL2(P_hat) + relL2(Cf)] / 3   (percent)"""
    model.eval()
    sc, det = [], {}
    for Re, (pts, cf) in val_sets.items():
        o = _pred(model, pts)
        eU = 100 * torch.sqrt(torch.mean((o[:, 0] - pts["U"]) ** 2)).item()
        eP = 100 * torch.sqrt(torch.mean((o[:, 2] - pts["P"]) ** 2)).item()
        cfn = P.wall_shear_cf(model, cf["x"], cf["Re"])
        eC = 100 * torch.sqrt(torch.mean(((cfn - cf["cf_ref"]) / cf["cf_ref"]) ** 2)).item()
        sc.append((eU + eP + eC) / 3)
        det[Re] = (eU, eP, eC)
    model.train()
    m = float(np.mean(sc))
    return (m, det) if detail else m


def compute_losses(model, w, batch_data, dev, n_pde, n_bc, phys_on):
    L = {}
    x, r, U, V, Pp, Re = [batch_data[k] for k in ("x", "r", "U", "V", "P", "Re")]
    Uo, Vo, Po, _ = model(x, r, Re)
    vs = P.V_SCALE_FACTOR * P.est_torch(Re)[1]
    L["data"] = (torch.mean((Uo - U) ** 2) + w["v_sub"] * torch.mean(((Vo - V) / vs) ** 2)
                 + w["p_sub"] * torch.mean((Po - Pp) ** 2))
    if phys_on:
        xp, rp, Rep = D.sample_pde(n_pde, dev)
        res = P.rans_sa_residuals(model, xp, rp, Rep)
        L["cont"] = torch.mean(res["cont"] ** 2)
        L["momx"] = torch.mean(res["momx"] ** 2)
        L["momr"] = torch.mean(res["momr"] ** 2)
        L["sa"] = torch.mean(res["sa"] ** 2)
    else:
        z = torch.zeros((), device=dev)
        L["cont"] = L["momx"] = L["momr"] = L["sa"] = z
    if w["inlet"] > 0:
        ri, Rei = D.sample_inlet(n_bc, dev); L["inlet"] = P.inlet_loss(model, ri, Rei)
    else:
        L["inlet"] = torch.zeros((), device=dev)
    if w["outlet"] > 0:
        ro, Reo = D.sample_outlet(n_bc, dev); L["outlet"] = P.outlet_loss(model, ro, Reo)
    else:
        L["outlet"] = torch.zeros((), device=dev)
    if w["sym"] > 0:
        xa, Rea = D.sample_axis(n_bc, dev); L["sym"] = P.axis_loss(model, xa, Rea)
    else:
        L["sym"] = torch.zeros((), device=dev)
    total = (w["data"] * L["data"] + w["cont"] * L["cont"] + w["mom"] * (L["momx"] + L["momr"])
             + w["sa"] * L["sa"] + w["inlet"] * L["inlet"] + w["outlet"] * L["outlet"] + w["sym"] * L["sym"])
    return total, L


def effective_weights(weights, physics_factor=1.0, variant="searched"):
    w = dict(C.DEFAULT_WEIGHTS)
    w.update({k: float(v) for k, v in (weights or {}).items() if k in w})
    if variant == "data_only":
        for k in ("cont", "mom", "sa", "inlet", "outlet", "sym"):
            w[k] = 0.0
    else:
        for k in ("cont", "mom", "sa"):
            w[k] *= physics_factor
    return w


def train_model(regime, weights, budget="cpu_final", seed=0, physics_factor=1.0, variant="searched",
                depth=C.NET_DEPTH, width=C.NET_WIDTH, device="cpu", pool=None, val_sets=None,
                val_every=500, verbose=True, label="", torch_threads=None):
    if torch_threads:
        torch.set_num_threads(torch_threads)
    dev = pick_device(device)
    bg = C.BUDGETS[budget] if isinstance(budget, str) else budget
    w = effective_weights(weights, physics_factor, variant)
    phys_on = any(w[k] > 0 for k in ("cont", "mom", "sa"))
    torch.manual_seed(seed); np.random.seed(seed)
    model = P.PipePINN(depth, width).to(dev)
    pool = pool or D.training_pool(regime, dev)
    val_sets = val_sets or make_val_sets(dev)
    n_pool = pool["x"].shape[0]
    lr0 = w["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    best = dict(score=float("inf"), state=None, it=0)
    last_good = copy.deepcopy(model.state_dict())
    rollbacks = 0
    hist = []
    t0 = time.time()
    nA = bg["adam"]
    val_every = min(val_every, max(25, nA // 3))
    for it in range(1, nA + 1):
        lr = lr0 * (1e-2) ** ((it - 1) / max(nA, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n_pool, (bg["n_data"],), device=dev)
        batch = {k: pool[k][idx] for k in ("x", "r", "U", "V", "P", "Re")}
        total, L = compute_losses(model, w, batch, dev, bg["n_pde"], bg["n_bc"], phys_on)
        opt.zero_grad(set_to_none=True)
        finite = bool(torch.isfinite(total))
        if finite:
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            finite = bool(torch.isfinite(gn))
        if not finite:
            rollbacks += 1
            if verbose:
                print(f"  [{label}] non-finite at it {it}: rollback #{rollbacks}", flush=True)
            if rollbacks > 10:
                break
            model.load_state_dict(last_good)
            opt = torch.optim.Adam(model.parameters(), lr=lr0)
            lr0 *= 0.5
            continue
        opt.step()
        if it % val_every == 0 or it == nA:
            sc, det = val_score(model, val_sets, detail=True)
            last_good = copy.deepcopy(model.state_dict())
            if sc < best["score"]:
                best = dict(score=sc, state=copy.deepcopy(model.state_dict()), it=it)
            rec = dict(it=it, total=float(total.detach()), val=sc, **{k: float(v.detach()) for k, v in L.items()})
            hist.append(rec)
            if verbose:
                print(f"  [{label}] it {it:6d}/{nA} loss {total.item():9.3e} data {L['data'].item():8.2e} "
                      f"mom {L['momx'].item():8.2e} sa {L['sa'].item():8.2e} | VAL {sc:6.2f} "
                      f"(best {best['score']:.2f}@{best['it']}) lr {lr:.1e} {time.time() - t0:5.0f}s", flush=True)
    if best["state"] is None:                      # e.g. immediate divergence
        best["state"] = copy.deepcopy(model.state_dict()); best["score"] = val_score(model, val_sets)
    model.load_state_dict(best["state"])
    # ---- L-BFGS polish: kept only if VAL improves -------------------------------------------
    if bg["lbfgs"] > 0 and finite:
        pre = copy.deepcopy(model.state_dict())
        nb = 4 * bg["n_data"]
        idx = torch.randint(0, n_pool, (nb,), device=dev)
        batch = {k: pool[k][idx] for k in ("x", "r", "U", "V", "P", "Re")}
        torch.manual_seed(seed + 12345)
        fixed = dict(pde=D.sample_pde(2 * bg["n_pde"], dev), inlet=D.sample_inlet(bg["n_bc"], dev),
                     outlet=D.sample_outlet(bg["n_bc"], dev), axis=D.sample_axis(bg["n_bc"], dev))
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=bg["lbfgs"], history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            lb.zero_grad()
            x, r, U, V, Pp, Re = [batch[k] for k in ("x", "r", "U", "V", "P", "Re")]
            Uo, Vo, Po, _ = model(x, r, Re)
            vs = P.V_SCALE_FACTOR * P.est_torch(Re)[1]
            tot = w["data"] * (torch.mean((Uo - U) ** 2) + w["v_sub"] * torch.mean(((Vo - V) / vs) ** 2)
                               + w["p_sub"] * torch.mean((Po - Pp) ** 2))
            if phys_on:
                res = P.rans_sa_residuals(model, *fixed["pde"])
                tot = tot + w["cont"] * torch.mean(res["cont"] ** 2) + w["mom"] * (
                    torch.mean(res["momx"] ** 2) + torch.mean(res["momr"] ** 2)) + w["sa"] * torch.mean(res["sa"] ** 2)
            if w["inlet"] > 0: tot = tot + w["inlet"] * P.inlet_loss(model, *fixed["inlet"])
            if w["outlet"] > 0: tot = tot + w["outlet"] * P.outlet_loss(model, *fixed["outlet"])
            if w["sym"] > 0: tot = tot + w["sym"] * P.axis_loss(model, *fixed["axis"])
            tot.backward()
            return tot
        try:
            lb.step(closure)
            sc = val_score(model, val_sets)
            if np.isfinite(sc) and sc < best["score"]:
                if verbose:
                    print(f"  [{label}] L-BFGS polish kept: VAL {best['score']:.2f} -> {sc:.2f}", flush=True)
                best["score"] = sc
            else:
                model.load_state_dict(pre)
                if verbose:
                    print(f"  [{label}] L-BFGS polish discarded (VAL {sc:.2f} vs {best['score']:.2f})", flush=True)
        except Exception as e:
            model.load_state_dict(pre)
            if verbose:
                print(f"  [{label}] L-BFGS failed ({e}); discarded", flush=True)
    model.eval()
    info = dict(val_score=best["score"], best_it=best["it"], rollbacks=rollbacks, seconds=time.time() - t0,
                weights=w, regime=regime, seed=seed, budget=budget if isinstance(budget, str) else "custom",
                physics_factor=physics_factor, variant=variant, history=hist)
    return model, info


def save_model(model, info, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state=model.state_dict(), depth=model.depth, width=model.width, info=info), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = P.PipePINN(ck["depth"], ck["width"]).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["info"]


def model_dir(regime):
    return os.path.join(C.OUT_DIR, "models", regime)


def load_weights(regime, path=None):
    path = path or os.path.join(C.OUT_DIR, f"best_weights_{regime}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"(no {path}; using DEFAULT_WEIGHTS)")
    return dict(C.DEFAULT_WEIGHTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["dense", "sparse"], required=True)
    ap.add_argument("--budget", default="cpu_final", choices=list(C.BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="final")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--physics_factor", type=float, default=1.0)
    ap.add_argument("--variant", default="searched", choices=["searched", "data_only"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--big", action="store_true", help="6 x 128 network (GPU)")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    depth, width = (6, 128) if a.big else (C.NET_DEPTH, C.NET_WIDTH)
    weights = load_weights(a.regime, a.weights)
    pool = D.training_pool(a.regime, dev)
    vs = make_val_sets(dev)
    print(f"regime={a.regime} device={dev} data points={pool['x'].shape[0]} net={depth}x{width} budget={a.budget}")
    for s in a.seeds:
        label = f"{a.regime}/{a.tag}/s{s}"
        m, info = train_model(a.regime, weights, a.budget, s, a.physics_factor, a.variant, depth, width, dev,
                              pool, vs, label=label)
        save_model(m, info, os.path.join(model_dir(a.regime), f"{a.tag}_seed{s}.pt"))
        print(f"== {label}: VAL score {info['val_score']:.3f}  ({info['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
