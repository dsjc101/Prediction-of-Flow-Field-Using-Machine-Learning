# Turbulent developing channel flow -- hybrid data + RANS/SA PINN

Same geometry as the laminar report (parallel plates, uniform inlet, outlet pressure gauge), now turbulent:
Re = U_b D_h / nu in **10 000 ... 120 000** (Re_tau ~ 160 ... 1450), channel length L = 160 h.
Only the lower half-channel is computed (mirror symmetry). Non-dimensional: lengths / h (half-height), velocity / U_b,
pressure / (0.5 U_b^2).

The PINN learns (x, y, Re) -> (U, V, P, nu~) from reference data **plus** the full elliptic steady RANS equations with the
Spalart-Allmaras closure. Two data regimes are supported:

* **dense** (default): every reference grid point outside three held-out streamwise bands is training data;
* **sparse** (`--sparse`): data only on 4 wall-normal profiles per Re at x/h = 3, 15, 50, 130 (well under 1 % of the
  stations of the dense run). Outputs go to `outputs_sparse/`.

## Install and run order (Windows cmd / PowerShell / Linux; no shell loops needed)

```
pip install -r requirements.txt

# --- dense-data pipeline ---------------------------------------------------------------
python run_all.py                       # reference -> verify -> weight search -> finalize -> 3 final seeds -> evaluate (VAL only)

# --- sparse-data pipeline (the physics-informed regime) --------------------------------
python run_all.py --sparse                          # same stages, outputs in outputs_sparse/
python run_all.py --sparse --stage ablation         # A = found weights, B = data only, C = physics x100   (VAL only)
python run_all.py --sparse --stage sweep            # physics-scale sweep x1, x10, x100, x1000, x10000     (VAL only)
python run_all.py --sparse --stage final --physics_factor 100    # 3 seeds with mom, sa, cont weights x100
python run_all.py --sparse --stage evaluate
python region_report.py --sparse                    # errors by region (upstream of data, between stations, ...)

# --- diagnostics -----------------------------------------------------------------------
python dns_check.py --download          # once (needs internet); then:  python dns_check.py [--models outputs/final_s*.pt]
python check_residuals.py --models outputs_sparse/abl_A_found.pt ...   # reference residual under the PINN equations vs PINN residual

# --- TEST Reynolds numbers: ONCE per model, at the very end -----------------------------
python run_all.py --sparse --stage test
python run_all.py --stage test          # dense baseline
```

Stages: `reference | search | finalize | final | evaluate | ablation | sweep | test` (each can be run alone; all are resumable /
idempotent). Options: `--workers N` and `--trials_per_worker K` (parallel Optuna workers), `--search_budget`, `--final_budget`
(`cpu_long` = 40k iterations), `--factors ...` (sweep), `--physics_factor F` (final), `--resume_search`.
Budgets (`train.py`): `smoke` (wiring test, seconds), `quick`, `standard`, `full` (GPU-sized), `cpu_search`, `cpu_final`, `cpu_long`.
Defaults: 4x64 network and CPU budgets without a GPU; 6x128 and GPU budgets with one. `PINN_FAST=1` switches to a coarse
reference and separate folders (debugging only, never for reported results). Long jobs log to `<outputs>/logs/`.

Measured wall-clock on a 16-thread CPU (no GPU): reference for all Re ~ 1 min; weight search (48 trials, 6 workers) ~ 17-21 min;
3 final seeds in parallel ~ 30 min; ablation (3 variants) ~ 30 min; sweep (3 new variants) ~ 30 min.

## Files

| file | role |
|---|---|
| `config.py` | all constants, Re roles (TRAIN/VAL/TEST), SA constants, Re-only scales, sparse-mode settings |
| `reference_solver.py` | parabolic marching SA-RANS solver (turbulent counterpart of the Bodoia-Osterle solver) |
| `build_reference.py`, `verify_reference.py` | cache + verification (Dean, law of the wall, grid convergence, mass, development length) |
| `data.py` | samplers, hold-outs, sparse stations, Cf metric, **test-set guard** |
| `model.py`, `physics.py` | network with hard no-slip; RANS + SA residuals by autograd (float32-safe SA terms) |
| `train.py` | Adam -> L-BFGS, resampling, NaN rollback, early stopping on VAL |
| `weight_search.py` | Optuna TPE + Hyperband search of the loss weights; top-3 re-trained with new seeds |
| `run_all.py` | one-command driver for all stages |
| `evaluate.py`, `region_report.py` | metrics, overfitting diagnostics, error by region |
| `dns_check.py`, `check_residuals.py` | DNS truth check; equation-consistency check of the reference data |

## Design decisions and their sources

* **Closure: Spalart-Allmaras** (Spalart & Allmaras 1992/1994; limiter of Allmaras, Johnson & Spalart 2012; inlet nu~ = 3 nu from
  Spalart & Rumsey 2007). RANS + SA inside a PINN: Patel, Mons, Marquet & Rigas (2024, Phys. Rev. Fluids 9:034605); RANS-PINNs in
  general: Eivazi, Tahani, Schlatter & Vinuesa (2022, Phys. Fluids 34:075117). Pioch et al. (2023, Fluids 8:43) report that
  transport-equation closures are hard to train in PINNs. This was borne out here: the searched momentum/SA weights ended up tiny
  relative to the data weight and the physics scale had to be set by a separate sweep (see Results).
* **Reference solver**: parabolised RANS marching (Patankar & Spalding 1970/1972; Bodoia & Osterle 1961), wall-resolved. The SA
  station equation is solved by damped (pseudo-transient) Newton because explicit-production Picard does not converge near the wall
  (Kelley & Keyes 1998). It is a different method from the PINN's elliptic equations, so agreement is an independent check;
  `check_residuals.py` shows the reference satisfies the PINN equations to <= 0.17 u_tau^2 (y+ >= 5).
* **Independent truth**: channel DNS of Moser, Kim & Mansour (1999) at Re_tau = 178, 392, 587 (`dns_check.py`; files in
  Lee & Moser (2015) format dropped into `dns_data/` are also read); Dean (1978) correlation in `verify_reference.py`.
* **Hard no-slip** via output factor tanh(y+/10): Lu et al. (2021, SISC 43:B1105), Sukumar & Srivastava (2022, CMAME 389:114333).
  No wall-BC weight exists to tune.
* **Wall-log input coordinate** log(1+y+)/log(1+Re_tau,est) and sqrt(x) inlet stretching: law-of-the-wall structure (Pope 2000);
  coordinate stretching (Tannehill, Anderson & Pletcher 1997). Both are Re-only functions (no solved field enters).
* **Resampling every step** (Lu et al. 2021, DeepXDE) with wall/inlet-clustered points; Adam then L-BFGS (Raissi et al. 2019;
  Rathore et al. 2024), L-BFGS kept only if VAL improves.
* **Weights by search, not assumption**: TPE (Bergstra et al. 2011) via Optuna (Akiba et al. 2019), Hyperband pruner (Li et al. 2018),
  random-over-grid rationale (Bergstra & Bengio 2012). Objective = mean relL2 of u, p and Cf on the VAL Reynolds numbers. Optional
  gradient-norm adaptive weights (Wang, Teng & Perdikaris 2021, `train.py --adaptive`) exist but were **not** used for any reported result.
* **Selection score excludes V**: V = -int U_x dy is a small difference quantity, so its relative error is U's error amplified; V is
  reported but never scored.

## Anti-overfitting (enforced in code)

1. **Re split**: TRAIN 10k/20k/40k/80k; VAL 14k, 56k (interpolation) and 88k (1.1x extrapolation) for early stopping, weight search
   and ablations only; TEST 28k (interpolation), 96k (1.2x), 120k (1.5x). `data.py` raises `PermissionError` if a TEST Re is loaded
   without `--final`. (Cawley & Talbot 2010: tuning and reporting on the same data gives optimistic bias.) 88k is in VAL because
   interpolation alone cannot reward the physics terms.
2. **x-band hold-out** (dense mode): three streamwise bands (x/h in 4-7, 25-32, 70-85) are removed from the data loss at every TRAIN Re
   and scored separately. In sparse mode the stations avoid these bands, so they are again held out.
3. **Fresh points every step**; **independent fresh-point PDE residual** at evaluation.
4. **Early stopping** on VAL (Prechelt 1998).
5. **Top-3 weight configurations re-trained with two fresh seeds**; reported numbers are mean +- std over 3 seeds.
6. TEST Re (and any Re above the TRAIN maximum) never see any loss term.

## Results (all numbers: relL2 %, mean +- std over 3 seeds; full tables in `Turbulent_PINN_results.md`)

SA-RANS reference vs DNS, fully developed region (Re_tau 178 / 392 / 587): U error 0.56 / 0.92 / 0.98 %, Cf error -1.9 / -0.8 / +0.9 %.

| model | TEST 28k (interp) u / p / Cf | TEST 96k (1.2x) | TEST 120k (1.5x) | PDE residual at 28k / 96k / 120k |
|---|---|---|---|---|
| sparse, physics x100 | 2.39 / 0.71 / 2.69 | 1.72 / 0.51 / 2.75 | 1.62 / 0.94 / 3.33 | 1.5 / 6.0 / 11.8 |
| dense | 0.42 / 0.38 / 6.11 | 0.48 / 0.44 / 5.23 | 0.58 / 0.58 / 5.22 | 22.5 / 49 / 76 |

Sparse physics ablation (VAL, single seed): data only 3.23 % headline (pressure 3.9 %, PDE residual up to 1460) vs 1.60 % with physics x100
(pressure 0.57 %, residual <= 4.7); x1000 degrades to 5.3 % and x10000 to 18 %.

## Caveats

* The PINN learns **SA-RANS**, not real turbulence. Floor on total error = SA model error (`dns_check.py`): up to 1.0 % (U) and 1.9 % (Cf)
  in the fully developed region; DNS covers only three Re and only that region.
* **Inlet region.** The parabolic reference is least reliable within ~1-2 h of the inlet (same limitation as Bodoia-Osterle at low Re;
  Wendel 1961); `X_DATA_MIN = 0.05`. In sparse mode u error upstream of the first station (x < 3 h) is ~6 % and V is unusable overall
  (~97 % VAL) though ~7-8 % between stations.
* The sparse PDE residual grows with extrapolation (3.9 at 80k -> 5.9 at 96k -> 11.8 at 120k) while the error against the reference stays flat.
* The ablation and the physics-scale sweep are **single-seed** and validation-only; x100 lies within a factor 10 of an accuracy cliff.
  The weight search runs at a short budget (1500 iterations), is noisy (top trials tied within seed noise), and biases against slowly
  converging physics terms; only the physics scale was re-checked at the final budget.
* A 4x64 network was used (CPU only); the residual stops improving near 2.6 for physics factors >= 100, suggesting a capacity or
  near-wall limit (not tested).
* Sparse station locations were chosen once and not varied.
* Pressure in the reference is P(x) only (boundary-layer approximation); the PINN's y-momentum may differ slightly (required P_y <= 0.4 u_tau^2).
* Reference solver vs Dean: Cf -5.4 % at Re = 1e4 improving to 0 % at 1.2e5; log-layer slope 2.5-2.7 vs 1/kappa = 2.44 (expected for SA).
* L/h = 160 makes every Re fully developed (measured L_dev <= 110 h); it differs from the laminar report's L, which is Re-dependent.
* Claims apply to Re = 10k-120k (Re_tau 160-1450) and the half-channel with symmetry.
