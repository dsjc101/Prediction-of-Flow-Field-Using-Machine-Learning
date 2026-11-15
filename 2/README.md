# Axisymmetric Circular-Pipe Entrance-Flow PINN — 3-D Extension

This is the 3-D (circular-pipe) follow-up to the 2-D plane-channel hybrid
data+physics PINN, built to the same specification: **same architecture
family, same hybrid loss, same Re train/test split (50/100/200/500 train,
300 interpolation + 600 extrapolation test)**, applied to a circular pipe
instead of a parallel-plate channel, and validated against the standard
analytical solution (Hagen–Poiseuille) exactly as the channel project was
validated against plane-Poiseuille flow.

**Why axisymmetric, not full 3-D Cartesian:** a circular pipe with a
uniform inlet and no swirl has a solution that is exactly independent of
the azimuthal angle θ — this is not a simplifying approximation, it's the
standard formulation for this class of problem (Hornbeck 1964 solves it
the same way). The network is queried over (x, r, Re) and the full 3-D
Cartesian field is reconstructed at any point via `model.to_cartesian`.
See `config.py`'s and `model.py`'s docstrings for the full argument and
citations.

Every formula, discretization choice, and design decision in this project
is backed by a cited source in the file where it's used — see each file's
docstring. Headline citations:

| What | Source |
|---|---|
| Reference-data marching solver | Hornbeck, R.W. (1964), *Laminar Flow in the Entrance Region of a Pipe*, Appl. Sci. Res. A13, 224–232. https://doi.org/10.1007/BF00382049 |
| Entrance-length functional form | Langhaar, H.L. (1942), *Steady Flow in the Transition Length of a Straight Tube*, J. Appl. Mech. 9(2), A55–A58. https://doi.org/10.1115/1.4009183 |
| Fully-developed exact solution | Hagen–Poiseuille flow; White, F.M. (2006), *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, Sec. 3-4 |
| PINN hybrid-loss framework | Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019), *Physics-informed neural networks*, J. Comput. Phys. 378, 686–707 |
| Axisymmetric NS in cylindrical coords | White (2006), App. B; Schlichting & Gersten (2017), *Boundary-Layer Theory*, 9th ed., Ch. 3 |
| Axisymmetric PINN precedent | Axisymmetric RANS-PINN for submerged jet grouting, Acta Geotechnica (2026), https://doi.org/10.1007/s11440-026-02979-7 |
| Adaptive loss balancing (optional) | Wang, S., Teng, Y. & Perdikaris, P. (2021), SIAM J. Sci. Comput. 43(5), A3055–A3081 |

## Files

| File | What it is |
|---|---|
| `config.py` | Geometry (pipe radius/diameter), fluid properties, Re sweep, exact Hagen–Poiseuille profile, entrance-length correlation |
| `entrance_solver.py` | Hornbeck (1964) axisymmetric finite-difference marching reference-data generator |
| `generate_data.py` | Caches reference data to `outputs/reference_data.npz` |
| `model.py` | `PipePINN`: (x, r, Re) → (u_x, u_r, p), plus `to_cartesian` for full 3-D field queries |
| `physics.py` | Full elliptic axisymmetric Navier–Stokes residuals via autodiff |
| `adaptive_weights.py` | Optional Wang-et-al.-style adaptive loss balancing (unchanged from the 2-D project — geometry-agnostic) |
| `train_hybrid.py` | Training loop — hybrid loss (data + PDE + inlet + wall + **axis** + outlet), fixed weights by default, `--adaptive` flag. Bare `python train_hybrid.py` now runs the full Adam→L-BFGS pipeline in one command (see "Running it" below) |
| `evaluate.py` | Scores all three outputs — **u, v, p** — developing-region check (vs. Hornbeck), fully-developed check (vs. Hagen–Poiseuille for u, exact v=0, and the exact constant-dp/dx pressure for p), train-vs-unseen-Re generalization proof for each field |
| `evaluate_global.py` | Entire-(x,r)-domain error heatmaps and error-vs-Re scatter, for **u, v, p** |
| `deploy_and_visualize.py` | Full flowfield predictions, linear-axis error distributions, TorchScript (.pt) / ONNX model export, parameter extraction, and physics learning verification |
| `compare_weighting.py` | Trains fixed/adaptive/outlet-boosted/data-boosted loss-weight configurations to separate checkpoints, scores each on u, v and p, and copies the config with the best combined (unweighted mean of global u/v/p error) score over `outputs/pipe_pinn_hybrid.pt` |
| `Pipe_PINN_Report.docx` | Written report — full derivation and citation trail |
| `outputs/` | Checkpoint, figures, metrics CSVs, evaluation summary (from an actual training run — see report) |

## Running it

```bash
python generate_data.py            # Hornbeck (1964) reference solutions -> outputs/reference_data.npz

python train_hybrid.py             # DEFAULT: runs the full pipeline -- Adam(2500) then L-BFGS(300) polish, one command
python train_hybrid.py --fresh     # same, but starts from a fresh model instead of resuming outputs/pipe_pinn_hybrid.pt
python train_hybrid.py --adaptive  # same, but with Wang-et-al. (2021) adaptive loss weighting instead of fixed weights

python evaluate.py                 # u/v/p profile comparisons + generalization proof -> outputs/
python evaluate_global.py          # entire-domain u/v/p heatmaps -> outputs/

python compare_weighting.py        # OPTIONAL: sweep loss-weighting configs (fixed/adaptive/
                                    # outlet_boost/data_boost), score each on u/v/p, and copy the
                                    # winner over outputs/pipe_pinn_hybrid.pt (see its own
                                    # docstring for exactly how "best" is decided)
```

`train_hybrid.py` used to require two separate commands (`adam` then
`lbfgs`) and it was easy to forget the second one — running Adam alone
looks like it has mostly converged (the loss plateaus), but the near-inlet
region is still under-resolved without the L-BFGS polish (developing-
region error ~15–23% instead of ~3–6%). A bare `python train_hybrid.py`
now runs both stages automatically, so this is no longer something you
can accidentally skip. The old two-stage interface is still available if
you want to inspect the model between stages or resume just one of them:

```bash
python train_hybrid.py adam 2500 --fresh   # Adam stage only
python train_hybrid.py lbfgs 300           # L-BFGS stage only (resumes the last checkpoint)
```

## What changed vs. the 2-D channel project, and why

- **Geometry**: (x, y) plane-channel → (x, r) axisymmetric pipe cross-section.
- **Reference solver**: Bodoia & Osterle (1961) → Hornbeck (1964), the
  circular-pipe analogue of the same marching method (same superposition
  trick, same mass-conservation constraint, different — cylindrical —
  finite-difference molecule and a genuine r=0 axis singularity to handle,
  see `entrance_solver.py`).
- **PDE**: 2-D Cartesian Navier–Stokes → axisymmetric (no-swirl)
  Navier–Stokes, which adds curvature (1/r, 1/r²) terms that vanish in the
  flat limit (`physics.py`).
- **Boundary conditions**: two symmetric no-slip walls → one no-slip wall
  (r=R) **plus** a symmetry-axis condition (r=0, v=0) that has no
  analogue in the channel geometry.
- **Analytical benchmark**: plane-Poiseuille (peak/mean ratio 1.5) →
  Hagen–Poiseuille (peak/mean ratio 2.0) — the different geometry changes
  this ratio, and the evaluation checks against the correct one.
- **Re/train/test split, architecture depth/width, normalization
  philosophy, hybrid-loss structure, optional adaptive weighting**: all
  kept identical, per the assignment ("same Re train and same Re test").
