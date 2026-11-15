# Rectangular-Duct PINN — 3rd Extension (2-D Channel → 3-D Pipe → 3-D Duct)

This is the requested next step after the circular-pipe project: same
PINN framework, same Re train/test split, same evaluation logic, now
applied to a rectangular duct. Every governing equation, discretization
choice, correlation, and design decision below is backed by a cited
source, same discipline as the earlier two reports.

## 1. Objective and scope

Same hybrid data+physics PINN family (Raissi, Perdikaris & Karniadakis
2019), same Re train (50,100,200,500) / test (300 interp, 600 extrap),
same two-part validation: (a) against an independent finite-difference
marching reference in the developing region, (b) against the exact
analytical fully-developed solution far downstream — now a rectangular
duct instead of a circular pipe, with a rectangular-duct-specific
analytical benchmark (Fourier series, not Hagen–Poiseuille).

CHANGED: the PINN predicts and is scored against all four fields —
streamwise velocity u, cross-stream velocities v and w, and kinematic
pressure p — not just u. This required extending `duct_solver.py` to
also produce reference v, w, p (it originally only marched u), via the
pressure-splitting method of **Patankar & Spalding (1972)**, already in
this project's own reference list (Sec 4 below) — see that file's module
docstring for the full method and for an explicit note on where the
implementation deliberately departs from full coupled SIMPLE (which was
tried and found numerically unstable in this single-pass-per-station
setting; the simplification used instead is a stable, station-local
elliptic recovery, still literature-grounded).

## 2. Why this is genuinely 3-D (no axisymmetric-style reduction)

The pipe project reduced (x,y,z)→(x,r) because a swirl-free circular pipe
is provably rotation-invariant about its axis. A rectangle has no such
continuous symmetry — rotating it by an arbitrary angle does not map it to
itself — so the duct's velocity field genuinely depends on both
cross-sectional coordinates (y,z); see the Fourier series in
`analytical_duct.py`. The network therefore takes the full (x,y,z,Re)
input (`pinn_model.py`), and the reference solver marches a full 2-D
cross-plane solve per station (`duct_solver.py`), not a 1-D
cross-stream solve. This is the single biggest structural change from
the pipe project and is why the marching-solver method (Section 4 below)
is different in kind, not just detail.

The duct DOES have two discrete reflection symmetries (about y=H/2 and
z=W/2), used only as a *sampling* convenience (`mirror_quadrant_points` in
`pinn_model.py`) — not a dimensionality reduction.

## 3. Geometry and Re sweep (`config.py`)

- Height H = 0.10 m (matches the pipe's D = 0.10 m and the channel's full
  height, so all three projects share one linear scale).
- Width W = 0.20 m → aspect ratio AR = 2.0, a standard benchmark ratio
  tabulated in **Shah, R.K. & London, A.L. (1978), *Laminar Flow Forced
  Convection in Ducts*, Academic Press, Ch. 7** — chosen specifically so
  the computed `u_max/u_mean` and `f·Re_Dh` (both computed in
  `analytical_duct.py`) are checkable against that published table when
  writing the report.
- Re defined on hydraulic diameter Dh = 4A/P (White, F.M. (2006), *Viscous
  Fluid Flow*, 3rd ed., McGraw-Hill, Ch. 6; Shah & London 1978, Ch. 1) —
  the standard convention for non-circular ducts, not H or W individually.
- ν = 1×10⁻⁴ m²/s, identical fluid to both earlier projects.

## 4. Reference (ground-truth) solver: `duct_solver.py`

Rectangular-duct counterpart of Bodoia & Osterle (1961) [channel] and
Hornbeck (1964) [pipe], generalized from a 1-D cross-stream marching
solve to a full **2-D cross-plane** marching solve, because the duct has
no 1-D reduction. This generalization follows:

- **Patankar, S.V. & Spalding, D.B. (1972)**, "A calculation procedure for
  heat, mass and momentum transfer in three-dimensional parabolic flows,"
  *Int. J. Heat Mass Transfer*, 15, 1787–1806,
  https://doi.org/10.1016/0017-9310(72)90054-3 — the paper that first
  generalized Bodoia-Osterle/Hornbeck-style parabolic marching to genuinely
  3-D duct flows.
- **Curr, R.M., Sharma, D. & Tatchell, D.G. (1972)**, "Numerical prediction
  of some three-dimensional boundary layers in ducts," applying that class
  of method specifically to duct entrance flow, and benchmarking against
  the classical rectangular-duct entrance analyses of **Han, L.S. (1960)**,
  "Hydrodynamic Entrance Lengths for Incompressible Laminar Flow in
  Rectangular Ducts," *J. Appl. Mech., Trans. ASME*, 27 (Series E),
  403–409, https://doi.org/10.1115/1.3644015 — the direct rectangular-duct
  analogue of Langhaar (1942), already cited in the pipe report.

Implementation note stated explicitly (not hidden): Patankar & Spalding
solved the per-station 2-D system with a line-by-line (ADI-style) TDMA
sweep, the standard 1970s iterative technology (see also Patankar, S.V.
(1980), *Numerical Heat Transfer and Fluid Flow*, Hemisphere, Ch.5,
already cited in the pipe report). This code instead assembles the
per-station system as one sparse linear system and solves it directly
(`scipy.sparse.linalg.spsolve`) — same discretization and marching logic,
a modern direct solve replacing a 1970s iterative sweep, noted as a
substitution rather than claimed as a literal reproduction.

Two solves per station (particular G=0, homogeneous G=1), superposed via
the mass-conservation constraint — the same trick Hornbeck (1964) used,
generalized from 1-D to 2-D.

**Self-consistency checks are run BEFORE any neural network is
involved** (`generate_data.py` STEP 1, mirroring pipe report Sec 4.3):
mass conservation at every station, outlet field vs. the *exact* Fourier
series (not a memorized table constant), and measured entrance-length
coefficient K. **Run this first and send me the printed numbers** — if
the checks don't pass config asserts and stops, which is intentional.

## 5. Analytical benchmark: `analytical_duct.py`

Exact Fourier-series fully-developed solution (2-D Poisson equation, all
four walls no-slip), standard material reproduced in **Shah & London
(1978), Ch. 7** and **White (2006), Sec. 3-4** — same source family the
pipe/channel projects drew their exact benchmarks from. Implementation
note: rather than transcribe the textbook closed-form prefactor by hand
(error-prone), the code solves for the free constant via the same
mass-conservation constraint used everywhere else in the project — same
PDE, same BCs, same solution, just fixed numerically instead of by
substitution. `u_max/u_mean` and `f·Re_Dh` are both **computed from the
series**, not quoted from memory, precisely so they can be checked
against Shah & London's table rather than risk mis-quoting it.

## 6. Entrance length

Functional form `Le = K·Dh·Re` after **Han (1960)** — same pattern as the
pipe report's use of Langhaar (1942) for the functional form only. K is
**measured from this project's own solver** (not asserted from a
literature table), for the same reason the pipe report gave: the
literature K depends on convergence criterion and (here) aspect ratio.

## 7. Physics-loss equations: full elliptic 3-D Navier–Stokes

Standard Cartesian incompressible Navier–Stokes, **no curvature terms**
(contrast with the pipe's v/r, u_r/r, v_r/r−v/r² cylindrical terms) — a
rectangular duct in Cartesian coordinates has no metric curvature. This
is itself a clean cross-check: as W→H with H→∞ (very wide duct), the
duct physics should recover the plane-channel physics with zero
curvature-term artifacts, unlike the pipe-vs-channel comparison. Framework
per Raissi, Perdikaris & Karniadakis (2019), same as both earlier
projects.

## 8. Network and loss terms

Architecture UNCHANGED (4×80, tanh, Xavier) from both earlier projects,
so accuracy differences are attributable to geometry/physics, not
capacity — identical rationale to pipe report Sec 8. Loss terms:
data (TRAIN Re only), pde (all Re), inlet, wall (four faces, not one wall
+ axis), outlet. No axis term — rectangular ducts have no coordinate
singularity. Fixed weights 10:1:5:5:1, carried over unchanged from the
pipe project's data:pde:inlet:wall:outlet weights (its axis:1 term simply
drops). Optional Wang, Teng & Perdikaris (2021) adaptive weighting via
`--adaptive`, carried over unchanged (`adaptive_weights.py`).

Training: Adam (2500 it., lr=8e-4, StepLR) + L-BFGS polish (300 it.,
strong-Wolfe) — identical two-stage schedule to both earlier projects.

## 9. What to run, in order

```bash
pip install torch numpy scipy matplotlib

python config.py            # sanity-print geometry/Re sweep
python duct_solver.py        # standalone self-consistency check (now also
                              # reports outlet |v,w|/Umean and the p'/p_main
                              # ratio, see duct_solver.py)
python generate_data.py      # STEP 1 self-consistency battery, STEP 2 full runs
                              # -> writes data/duct_Re*.npz (U,V,W,P), data/meta.txt
python weight_search.py      # -> checkpoints/weight_search_results.npy;
                              # prints the best loss-weight set found (random
                              # search over data_u/data_v/data_w/data_p/pde/
                              # inlet/wall/outlet, scored on held-out-Re error
                              # -- see weight_search.py for the method and its
                              # citation). Copy the printed best weights into
                              # config.LOSS_WEIGHTS before the next step.
python train_hybrid.py       # -> writes checkpoints/duct_pinn.pt, loss_history.npy
                              # (add --adaptive to use gradient-norm loss balancing
                              # on top of the searched starting weights)
python evaluate.py           # -> figures/profile_Re*.png (u,v,w,p panels),
                              # developing_region_summary.txt (per-field relL2)
python evaluate_global.py    # -> figures/errormap_Re*.png (u,v,w,p error maps),
                              # global_summary.txt (per-field entire-domain and
                              # fully-developed-vs-exact relL2)
```

If a full run is too slow on your machine, reduce `Ny, Nz` in
`generate_data.py`'s `run_all()` call and/or `config.ADAM_ITERS` (and
`weight_search.py`'s `--n-candidates`/`--short-iters`) — flag in your
reply that you did this so the report can note the resolution actually
used (same "actual run, not projected" discipline as the pipe report).

## 10. What to send back for the final report

1. Printed output of `generate_data.py` STEP 1 (mass_err, outlet_relL2,
   Le, K, outlet|v,w|/Umean, p'/p_main_ratio per Re) — this becomes the
   Sec-4.3-equivalent self-consistency table.
2. `data/meta.txt` (measured K, L_DOMAIN used, Le per Re).
3. `weight_search.py`'s printed best-weight set and
   `checkpoints/weight_search_results.npy` — becomes the loss-weighting
   paragraph/table.
4. Printed loss trajectory from `train_hybrid.py` (first/last Adam
   losses per term, final L-BFGS loss) — becomes the training-trajectory
   paragraph.
5. Printed table from `evaluate.py` (per-field, per-Re developing-region
   relL2 + TRAIN vs UNSEEN generalization numbers) and the
   `figures/profile_Re*.png` images.
6. Printed table from `evaluate_global.py` (per-field entire-domain relL2
   and fully-developed-vs-exact relL2 per Re) and the
   `figures/errormap_Re*.png` images.
7. The `analytical_duct.py` standalone printout (`u_max/u_mean` and
   `f·Re_Dh` for AR=2) so the report can state whether it matches Shah &
   London's tabulated value for AR=2 (a real, checkable cross-reference
   — I'll flag the comparison honestly either way once I see the number).

Once I have those, I'll write the final Word report in the same format as
the channel/pipe reports, including a Section-11-style three-way
comparison table (channel vs. pipe vs. duct) and full reference list.

## 11. References

- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). J. Comput. Phys.,
  378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Bodoia, J.R. & Osterle, J.F. (1961). Applied Scientific Research A, 10,
  265–276. https://doi.org/10.1007/BF00411919
- Hornbeck, R.W. (1964). Applied Scientific Research A, 13, 224–232.
  https://doi.org/10.1007/BF00382049
- Han, L.S. (1960). J. Appl. Mech., Trans. ASME, 27 (Series E), 403–409.
  https://doi.org/10.1115/1.3644015
- Patankar, S.V. & Spalding, D.B. (1972). Int. J. Heat Mass Transfer, 15,
  1787–1806. https://doi.org/10.1016/0017-9310(72)90054-3
- Curr, R.M., Sharma, D. & Tatchell, D.G. (1972). Numerical prediction of
  some three-dimensional boundary layers in ducts. Computer Methods in
  Applied Mechanics and Engineering.
- Shah, R.K. & London, A.L. (1978). Laminar Flow Forced Convection in
  Ducts. Academic Press.
- White, F.M. (2006). Viscous Fluid Flow, 3rd ed. McGraw-Hill.
- Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow.
  Hemisphere Publishing.
- Wang, S., Teng, Y. & Perdikaris, P. (2021). SIAM J. Sci. Comput., 43(5),
  A3055–A3081. https://doi.org/10.1137/20M1318043
- Chorin, A.J. (1968). Math. Comp., 22, 745–762 (cited for the
  Poisson/potential decomposition used to recover v, w from continuity
  in duct_solver.py).
- Gresho, P.M. & Sani, R.L. (1987). Int. J. Numer. Methods Fluids, 7,
  1111–1145 (cited for the pressure-Poisson-equation approach used to
  recover p' in duct_solver.py).
- Bergstra, J. & Bengio, Y. (2012). J. Machine Learning Research, 13,
  281–305. https://jmlr.org/papers/v13/bergstra12a.html (cited for the
  random-search method used in weight_search.py).
- Langhaar, H.L. (1942). J. Appl. Mech., 9(2), A55–A58.
  https://doi.org/10.1115/1.4009183 (cited for the Le=K·D·Re functional
  form only, carried over context from the pipe report)
