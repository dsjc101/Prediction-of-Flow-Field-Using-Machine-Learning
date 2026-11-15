# Prediction of Flow Field Using Machine Learning: A Multi-Stage Physics-Informed Neural Network Investigation

Physics-informed neural networks (PINNs) are trained to predict developing internal flows in five configurations of increasing complexity: laminar flow between parallel plates (Stage 1), in a circular pipe (Stage 2) and in a rectangular duct of aspect ratio 2 (Stage 3), and turbulent flow in a channel (Stage 4) and in a pipe (Stage 5) at $\mathrm{Re} = 10^4$ to $1.2\times10^5$, modeled with the Reynolds-averaged Navier--Stokes equations and the Spalart--Allmaras turbulence model. Each network is trained on reference solutions from a parabolic marching solver together with the governing equations, and is tested at Reynolds numbers outside its training set.

The full technical paper and evaluation report is available at [`report/report.pdf`](report/report.pdf).

---

## Repository Structure

The repository maintains the exact 5-stage architectural layout:

```text
├── 1/                  # Stage 1: 2D Laminar Channel Flow (L = 4.0 m, h = 0.05 m)
│   ├── config.py
│   ├── entrance_solver.py
│   ├── model.py
│   ├── train_hybrid.py
│   └── outputs/        # Checkpoint: channel_pinn_hybrid.pt, metrics_full_domain.csv
├── 2/                  # Stage 2: Axisymmetric Laminar Pipe Flow (L = 4.0 m, R = 0.05 m)
│   ├── config.py
│   ├── entrance_solver.py
│   ├── model.py
│   ├── train_hybrid.py
│   ├── checkpoints/    # Checkpoint: pipe_pinn.pt
│   └── outputs/        # Reference data & evaluation summaries
├── 3/                  # Stage 3: 3D Rectangular Duct Flow (AR = 2.0, L = 5.0 m)
│   ├── config.py
│   ├── duct_solver.py
│   ├── generate_data.py
│   ├── pinn_model.py
│   ├── train_hybrid.py
│   ├── checkpoints/    # Checkpoint: duct_pinn.pt
│   └── figures/        # global_summary.txt
├── 4/                  # Stage 4: Developing Turbulent Channel Flow (RANS + Spalart-Allmaras)
│   ├── config.py
│   ├── model.py
│   ├── reference_solver.py
│   ├── train.py
│   ├── weight_search.py
│   ├── models/         # Checkpoint: best.pt
│   └── outputs/        # final_metrics.csv, ablation summaries
├── 5/                  # Stage 5: Axisymmetric Turbulent Pipe Flow (RANS + Spalart-Allmaras)
│   ├── config.py
│   ├── data.py
│   ├── pinn.py
│   ├── reference_solver.py
│   ├── search_weights.py
│   ├── train.py
│   ├── evaluate.py
│   └── outputs/        # Checkpoint: models/dense/final_seed0.pt, eval_dense_final_test.json
├── report/             # Complete LaTeX source, compiled PDF, tables, and figures
│   ├── report.pdf
│   ├── report.tex
│   ├── tables/
│   └── report_figs/
├── tools/              # Verification & prediction visualization tools
├── references/         # Academic reference index & Superpipe data access notes
└── archive/            # Earlier exploratory scripts
```

> **Stage 6 Status:** Stage 6 (turbulent 3D rectangular duct) was started but not completed; no checkpoints or quantitative results are claimed.

---

## Installation

Ensure you have Python 3.10.4 installed:

```bash
git clone https://github.com/dsjc101/Prediction-of-Flow-Field-Using-Machine-Learning.git
cd Prediction-of-Flow-Field-Using-Machine-Learning
pip install -r requirements.txt
```

---

## Evaluating Trained Checkpoints

To evaluate the saved production checkpoints and reproduce the quantitative error tables in the report:

- **Stage 1 (Table 3 in report):**
  ```bash
  python 1/train_hybrid.py --eval_only
  # Evaluates 1/outputs/channel_pinn_hybrid.pt -> outputs/metrics_full_domain.csv
  ```

- **Stage 2 (Table 5 in report):**
  ```bash
  python 2/train_hybrid.py --eval_only
  # Evaluates 2/checkpoints/pipe_pinn.pt -> outputs/metrics_global.csv
  ```

- **Stage 3 (Table 7 in report):**
  ```bash
  python 3/evaluate_global.py
  # Evaluates 3/checkpoints/duct_pinn.pt -> figures/global_summary.txt
  ```

- **Stage 4 (Table 9 in report):**
  ```bash
  python 4/train.py --eval_only
  # Evaluates 4/models/best.pt -> outputs/final_metrics.csv
  ```

- **Stage 5 (Table 11 in report & Superpipe validation):**
  ```bash
  python 5/evaluate.py
  # Evaluates 5/outputs/models/dense/final_seed0.pt -> outputs/eval_dense_final_test.json
  ```

---

## Training from Scratch

To retrain any stage model (fixed seed 0, identical hyperparameters as reported):

```bash
# Stage 1: Laminar Channel (Adam 3000 iters + L-BFGS 300 iters)
python 1/train_hybrid.py

# Stage 2: Laminar Pipe (Adam 3000 iters + L-BFGS 300 iters)
python 2/train_hybrid.py

# Stage 3: Laminar Duct (Adam 2500 iters + L-BFGS 300 iters)
python 3/train_hybrid.py

# Stage 4: Turbulent Channel (Adam 12,000 iters + L-BFGS 200 iters)
python 4/train.py

# Stage 5: Turbulent Pipe (Adam 12,000 iters + L-BFGS 200 iters)
python 5/train.py
```

---

## Known Modeling Discrepancies and Limitations

As documented in Section 10.1 of the report:
1. **Single Random Seed:** Production models for Stages 1--4 use random seed 0; Stage 5 assesses multi-seed stability (seeds 0, 1, and 2).
2. **Parabolic vs. Elliptic Reference:** Reference training data are generated with Hornbeck-type implicit parabolic marching solvers ($\partial^2 u/\partial x^2 pprox 0, \partial p/\partial y = 0$), creating localized entrance discrepancies with the elliptic Navier--Stokes PINN ($x \lesssim 0.5\,D_h$).
3. **Stage 3 Duct Pressure Gradient:** In the rectangular duct, the PINN downstream pressure gradient slope is $0.22$ to $0.38$ times the reference solver gradient at $\mathrm{Re}=50$--$100$, compounded by the reference solver's inlet transient and outlet gauge offset.

---

## Author & Contact

- **Author:** Divyansh Singh
- **Institution:** Indian Institute of Technology Kanpur (IIT Kanpur)

---

