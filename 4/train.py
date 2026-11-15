"""
train.py -- train the hybrid data + RANS/SA PINN.

    python train.py --weights outputs/best_weights.json --budget standard --seed 0
    python train.py --budget quick                       # default weights, short run

Training protocol (sources)
---------------------------
* Adam with exponential LR decay, then L-BFGS polish (Raissi et al. 2019; Rathore et al. 2024,
  ICML, "Challenges in training PINNs: a loss landscape perspective" -> Adam then L-BFGS).
* Fresh collocation/BC points every Adam step (Lu et al. 2021, DeepXDE) -> no memorisation.
* Best checkpoint chosen on the VAL Re only (early stopping, Prechelt 1998).
* Weights: fixed dict (found by weight_search.py) or Wang, Teng & Perdikaris (2021) gradient-norm
  adaptive balancing (--adaptive).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import config as C
import data as D
import physics as PH
from model import ChannelPINN

DEFAULT_WEIGHTS = dict(data=10.0, v_sub=0.05, p_sub=0.1, cont=0.1, mom=0.01, sa=0.05,
                       inlet=5.0, outlet=1.0, sym=1.0)

BUDGETS = {   # adam iterations, lbfgs iterations, n_data, n_pde, n_bc, val every
    "smoke":    dict(adam=30,     lbfgs=5,   n_data=512,  n_pde=512,  n_bc=128, val_every=15),
    "quick":    dict(adam=3000,   lbfgs=0,   n_data=2048, n_pde=4096, n_bc=512, val_every=250),
    "standard": dict(adam=15000,  lbfgs=300, n_data=4096, n_pde=8192, n_bc=1024, val_every=500),
    # CPU-sized presets (no GPU): use with  --hidden 4 --width 64
    "cpu_search": dict(adam=1500, lbfgs=0,   n_data=1024, n_pde=1024, n_bc=256,  val_every=150),
    "cpu_final":  dict(adam=12000, lbfgs=200, n_data=2048, n_pde=2048, n_bc=512, val_every=500),
    "cpu_long":   dict(adam=40000, lbfgs=300, n_data=2048, n_pde=2048, n_bc=512, val_every=1000),
    "full":     dict(adam=40000,  lbfgs=1000, n_data=8192, n_pde=16384, n_bc=2048, val_every=1000),
}


def default_arch():
    """(hidden, width): 6x128 on GPU, 4x64 on CPU (measured ~5-20x cheaper per iteration)."""
    return (6, 128) if torch.cuda.is_available() else (4, 64)


def default_budget(kind):
    gpu = torch.cuda.is_available()
    return {"search": "quick" if gpu else "cpu_search", "final": "standard" if gpu else "cpu_final"}[kind]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_val_sets(device, dtype, stride_x=8, stride_y=4):
    return [D.EvalSet(Re, device, dtype, stride_x=stride_x, stride_y=stride_y) for Re in C.RE_VAL]


def val_score(model, val_sets):
    """Returns (headline, {field: mean relL2}). Headline = mean(u, p, Cf) -- V excluded.
    V is excluded from the selection score: V = -int U_x dy is a tiny difference quantity, so its
    relative error is the AMPLIFIED error of U and cannot be good before U is ~1% accurate
    (measured: V error stuck at ~140% while U went 71%->22%). V is still reported and is tied to
    U through the continuity residual."""
    sd = [D.score_set(model, es)["all"] for es in val_sets]
    vf = {f: float(np.mean([s[f] for s in sd])) for f in ("u", "v", "p", "cf")}
    return (vf["u"] + vf["p"] + vf["cf"]) / 3.0, vf


def compute_losses(model, store, w, nd, npde, nbc, device, dtype):
    b = store.sample(nd)
    U, V, Ph, nt = model(b["x"], b["y"], b["Re"])
    L = {}
    L["data_u"] = ((U - b["u"]) ** 2).mean()
    vscale = C.VSCALE_FRAC * torch.sqrt(0.5 * 0.073 * (b["Re"] / 2.0) ** -0.25)
    L["data_v"] = ((V / vscale - b["v"]) ** 2).mean()
    L["data_p"] = ((Ph - b["p"]) ** 2).mean()
    xc, yc, Rc = D.sample_colloc(npde, device, dtype)
    r = PH.pde_residuals(model, xc, yc, Rc)
    L["cont"] = (r["cont"] ** 2).mean()
    L["mom"] = (r["mom_x"] ** 2).mean() + (r["mom_y"] ** 2).mean()
    L["sa"] = (r["sa"] ** 2).mean()
    bc = PH.boundary_losses(model, D.sample_inlet(nbc, device, dtype),
                            D.sample_outlet(nbc, device, dtype), D.sample_sym(nbc, device, dtype))
    L.update(bc)
    return L


def weighted(L, w):
    return (w["data"] * (L["data_u"] + w["v_sub"] * L["data_v"] + w["p_sub"] * L["data_p"])
            + w["cont"] * L["cont"] + w["mom"] * L["mom"] + w["sa"] * L["sa"]
            + w["inlet"] * L["inlet"] + w["outlet"] * L["outlet"] + w["sym"] * L["sym"])


GROUPS = {   # which raw terms each weight multiplies (for gradient-norm adaptation)
    "data": ["data_u", "data_v"], "cont": ["cont"], "mom": ["mom"], "sa": ["sa"],
    "inlet": ["inlet"], "outlet": ["outlet"], "sym": ["sym"],
}


def adapt_weights_wang(model, L, w, alpha=0.9, clip=(0.1, 100.0)):
    """Wang, Teng & Perdikaris (2021): w_i <- alpha w_i + (1-alpha) max|grad L_pde| / mean|grad L_i|,
    measured on the last-layer weights. Reference term = PDE momentum residual (kept at w=1)."""
    last = [model.layers[-1].weight]
    def gnorm(term):
        g = torch.autograd.grad(term, last, retain_graph=True, allow_unused=True)[0]
        return g.abs().max().item() if g is not None else 0.0
    ref = gnorm(L["mom"]) + 1e-12
    for k, terms in GROUPS.items():
        if k == "mom":
            continue
        gi = sum(gnorm(L[t]) for t in terms) / len(terms) + 1e-12
        hat = float(np.clip(ref / gi, *clip))
        w[k] = float(alpha * w[k] + (1 - alpha) * hat)
    return w


def train(weights=None, budget="quick", seed=C.SEED, adaptive=False, features="wall",
          hidden=C.HIDDEN, width=C.WIDTH, lr=1e-3, verbose=True, report_cb=None, tag="run",
          supervise_nut=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, dtype = get_device(), torch.float32
    if verbose:
        print(f"[{tag}] device = {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else "  <-- CPU: expect ~0.4 s/iter"), flush=True)
    bud = BUDGETS[budget] if isinstance(budget, str) else budget
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    model = ChannelPINN(hidden, width, features=features).to(device).to(dtype)
    store = D.TrainStore(C.RE_TRAIN, device, dtype)
    val_sets = make_val_sets(device, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_it = bud["adam"]
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1.0 / max(n_it, 1)))
    best = dict(score=1e9, state=None, it=-1)
    good = copy.deepcopy(model.state_dict())      # last state with finite parameters
    n_recover = 0
    hist = []
    t0 = time.time()
    for it in range(1, n_it + 1):
        opt.zero_grad()
        L = compute_losses(model, store, w, bud["n_data"], bud["n_pde"], bud["n_bc"], device, dtype)
        if adaptive and it % 250 == 1:
            w = adapt_weights_wang(model, L, w)
        loss = weighted(L, w)
        bad = not torch.isfinite(loss)
        if not bad:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            bad = not torch.isfinite(gn)
        if bad:                                     # roll back, reset Adam moments, halve LR
            n_recover += 1
            model.load_state_dict(good)
            opt.state.clear()
            for g_ in opt.param_groups:
                g_["lr"] *= 0.5
            if verbose:
                print(f"[{tag}] it {it}: non-finite loss/grad -> rolled back to last good state, "
                      f"lr -> {opt.param_groups[0]['lr']:.2e} (recovery {n_recover}/10)", flush=True)
            if n_recover >= 10:
                break
            continue
        opt.step()
        sched.step()
        if it % bud["val_every"] == 0 or it == n_it:
            vs, vf = val_score(model, val_sets)
            if np.isfinite(vs):
                good = copy.deepcopy(model.state_dict())
            hist.append(dict(it=it, loss=loss.item(), val=vs, **{k: v.item() for k, v in L.items()}))
            if vs < best["score"]:
                best = dict(score=vs, state=copy.deepcopy(model.state_dict()), it=it)
            if verbose:
                print(f"[{tag}] it {it:6d} loss {loss.item():9.3e} data(u,v,p)=({L['data_u'].item():.2e},"
                      f"{L['data_v'].item():.2e},{L['data_p'].item():.2e}) pde(c,m,sa)=({L['cont'].item():.2e},"
                      f"{L['mom'].item():.2e},{L['sa'].item():.2e}) VAL={vs:6.2f}% (u {vf['u']:.1f} p {vf['p']:.1f} cf {vf['cf']:.1f} v {vf['v']:.1f}) "
                      f"best={best['score']:.2f}%@{best['it']} {time.time()-t0:6.0f}s", flush=True)
            if report_cb is not None and report_cb(it, vs):
                break                                    # pruned
    model.load_state_dict(best["state"] if best["state"] is not None else model.state_dict())

    if bud["lbfgs"] > 0 and not (report_cb is not None):
        # L-BFGS on a FIXED large batch (L-BFGS needs a deterministic objective)
        torch.manual_seed(seed + 123)
        store_b = store.sample(bud["n_data"] * 2)
        xc, yc, Rc = D.sample_colloc(bud["n_pde"], device, dtype)
        inl, outl, sym = (D.sample_inlet(bud["n_bc"], device, dtype),
                          D.sample_outlet(bud["n_bc"], device, dtype), D.sample_sym(bud["n_bc"], device, dtype))
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=bud["lbfgs"], history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
        pre = copy.deepcopy(model.state_dict())
        pre_score = val_score(model, val_sets)[0]

        def closure():
            lb.zero_grad()
            b = store_b
            U, V, Ph, nt = model(b["x"], b["y"], b["Re"])
            vsc = C.VSCALE_FRAC * torch.sqrt(0.5 * 0.073 * (b["Re"] / 2.0) ** -0.25)
            Ld = {"data_u": ((U - b["u"]) ** 2).mean(), "data_v": ((V / vsc - b["v"]) ** 2).mean(),
                  "data_p": ((Ph - b["p"]) ** 2).mean()}
            r = PH.pde_residuals(model, xc, yc, Rc)
            Ld["cont"] = (r["cont"] ** 2).mean()
            Ld["mom"] = (r["mom_x"] ** 2).mean() + (r["mom_y"] ** 2).mean()
            Ld["sa"] = (r["sa"] ** 2).mean()
            Ld.update(PH.boundary_losses(model, inl, outl, sym))
            l = weighted(Ld, w)
            l.backward()
            return l
        lb.step(closure)
        post_score = val_score(model, val_sets)[0]
        if (not np.isfinite(post_score)) or post_score > pre_score:   # keep L-BFGS only if VAL improves
            model.load_state_dict(pre)
        elif verbose:
            print(f"[{tag}] L-BFGS improved VAL {pre_score:.2f}% -> {post_score:.2f}%")
        best["score"] = min(pre_score, post_score) if np.isfinite(post_score) else pre_score
    return model, dict(best_val=best["score"], best_it=best["it"], weights=w, history=hist,
                       seconds=time.time() - t0, recoveries=n_recover)


def save_model(model, path, meta):
    torch.save(dict(state=model.state_dict(), meta=meta,
                    arch=dict(hidden=len(model.layers) - 1, width=model.layers[0].out_features,
                              features=model.features)), path)


def load_model(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    m = ChannelPINN(a["hidden"], a["width"], features=a["features"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["meta"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="json file with loss weights")
    ap.add_argument("--budget", default=None, choices=list(BUDGETS), help="default: standard (GPU) / cpu_final (CPU)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--features", default="wall", choices=["wall", "plain"])
    ap.add_argument("--lr", type=float, default=None, help="default: lr stored in the weights json, else 1e-3")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.budget = a.budget or default_budget("final")
    dh, dw = default_arch()
    a.hidden, a.width = a.hidden or dh, a.width or dw
    wj = json.load(open(a.weights)) if a.weights else {}
    w = wj.get("weights")
    lr = a.lr if a.lr is not None else wj.get("lr", 1e-3)
    name = a.name or f"model_{a.budget}_s{a.seed}" + ("_adaptive" if a.adaptive else "")
    model, info = train(w, a.budget, a.seed, a.adaptive, a.features, hidden=a.hidden, width=a.width, lr=lr, tag=name)
    path = os.path.join(C.OUT_DIR, name + ".pt")
    save_model(model, path, dict(info=info, seed=a.seed, budget=a.budget, adaptive=a.adaptive))
    json.dump(info["history"], open(os.path.join(C.OUT_DIR, name + "_history.json"), "w"), indent=1)
    print("saved", path, " best VAL = %.2f%%" % info["best_val"])