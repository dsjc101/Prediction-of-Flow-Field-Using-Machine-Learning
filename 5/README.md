# Turbulent PIPE hybrid RANS + Spalart–Allmaras PINN

Axisymmetric (x, r) extension of the turbulent *channel* study to a circular pipe.
Same Reynolds-number roles, same loss structure, same dense / sparse data regimes, same search → ablation →
sweep → final → evaluate workflow. Only the geometry (cylindrical metric terms, axis, Re_tau) changes.

| role  | Re (= U_b·D/ν, D = 2R)                                   |
|-------|----------------------------------------------------------|
| TRAIN | 10 000, 20 000, 40 000, 80 000                           |
| VAL   | 14 000, 56 000 (interp.), 88 000 (1.1× extrap.)          |
| TEST  | 28 000 (interp.), 96 000 (1.2×), 120 000 (1.5×)          |

Note: the channel report defines Re with D_h = 4h, here D = 2R, so the **same Re numbers** mean
Re_τ ≈ 310 … 2800 in the pipe (channel: 160 … 1450). Domain: L = 160 R (verified: development length ≤ 0.53 L).

## Install
```
pip install numpy scipy matplotlib torch optuna
```
(CPU is fine. GPU: add `--device cuda --big` for a 6×128 network.)

## Run order (what to send back)
```
python selftest.py                              # 10 s, MUST print ALL OK
python run_all.py --stage reference             # ~1 min, 10 reference solutions -> data/
python verify_reference.py                      # tables: Cf vs Prandtl/McKeon, log slope, L_dev, grid convergence
python run_all.py --quick --workers 2           # optional: whole dense pipeline with tiny budgets (~3 min) to check your install
python run_all.py                               # DENSE: search -> final (3 seeds) -> evaluate (TRAIN+VAL)
python run_all.py --sparse                      # SPARSE: search -> ablation -> sweep -> final -> evaluate
python run_all.py --stage test                  # TEST Re, dense model    (ONCE)
python run_all.py --sparse --stage test         # TEST Re, sparse model   (ONCE)
```
Everything printed + `outputs/*.txt`, `outputs/*.json`, `outputs/*.png` is what I need.

## Files
| file | purpose |
|---|---|
| `config.py` | Re roles, geometry, grids, budgets, Re-only scalings (Prandtl smooth-pipe law) |
| `reference_solver.py` | independent **parabolic** SA marching solver, axisymmetric (finite volume, damped-Newton SA) |
| `verify_reference.py` | verification: mass, Cf vs Prandtl & McKeon (Superpipe), log-law slope, U_c/U_b, L_dev, grid/step convergence |
| `pinn.py` | network, hard constraints, **full elliptic axisymmetric RANS+SA residuals** (divergence form), BC losses, overflow-safe SA terms |
| `selftest.py` | SA terms vs textbook numpy, Poiseuille residual = 0, 60 random-net NaN test, reference inserted in PINN momentum eq. |
| `data.py` | dense / sparse regimes, hold-out bands, VAL / evaluation sets, samplers, fresh-residual points (seed 777) |
| `train.py` | Adam + clip + exp-LR decay, L-BFGS polish (kept only if VAL improves), early stopping, rollback |
| `search_weights.py` | Optuna TPE weight search on VAL only, 3-seed confirmation of top 3 |
| `evaluate.py` | relL2 tables (U, V̂, P̂, Cf), regions, fresh PDE residual, ratios, figures. TEST needs `--final` |
| `run_all.py` | orchestration (stages, ablation A/B/C, physics-scale sweep ×1…×10⁴ incl. 300) |

## Geometry-specific choices (differences from the channel code)
* Equations: continuity `U_x + V_r + V/r = 0`; x-mom `U U_x + V U_r + P_x = ∂ₓ(2ν_e U_x) + (1/r)∂ᵣ[r ν_e (U_r+V_x)]`;
  r-mom `… = ∂ₓ[ν_e(U_r+V_x)] + (1/r)∂ᵣ(2 r ν_e V_r) − 2ν_e V/r²`; SA with `(1/r)∂ᵣ[r(ν+ν̃)ν̃_r]`. V is the **radial** velocity.
* Wall distance `d = 1 − r`; no-slip exact via `g = tanh(Re_τ,est (1−r)/10)`; **axis V = 0 exact** via a factor `r` in V
  (U_r = P_r = ν̃_r = 0 on the axis is a soft loss `sym`). PDE points kept at r ≥ 0.02 (1/r terms).
* Re-only scalings use the Prandtl smooth-pipe law (never a solved field): `1/√λ = 2 log10(Re√λ) − 0.8`,
  `u_τ/U_b = √(λ/8)`, `Re_τ = (u_τ/U_b)·Re/2`.
* Hold-out bands (x/R ∈ [4,7], [25,32], [70,85]) and sparse stations (x/R = 3, 15, 50, 130) are the channel's, in R instead of h.
* Pressure scale `P̂ = p/(½ρU_b²)`, V̂ = V/(0.05 u_τ,est), all error scales are global Re-only (same as channel).

## Errors against REAL data (new): `truth_check.py`
Error chain: PINN -> SA reference (evaluate.py)  and  SA reference -> real pipe flow (truth_check.py).
```
python truth_check.py --download      # fetches Princeton Superpipe profiles (31.6k-146k) into data/exp/ ; needs internet
python truth_check.py                 # SA solver + both PINNs vs real profiles/friction, plus friction law at the 10 project Re
python truth_check.py --final         # additionally includes the TEST Re in the friction-law table (do once)
```
The solver is run at each benchmark Re (not a training Re) and the PINNs are queried at x = L; above 80k that is extrapolation.
Extra DNS: put a CSV in data/exp/ with header `# Re_b=44000 Re_tau=1136.6` and columns `y_over_R, U_plus`
(e.g. Pirozzoli et al. 2021 pipe DNS: Re_b 44k / 82.5k / 133k).
