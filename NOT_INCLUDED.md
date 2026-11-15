# Files Excluded From the Repository

The following assets from the development workspace are excluded from this repository to ensure compliance with copyright best practices, stay well within GitHub repository size boundaries (< 1 GB), and maintain clean project hygiene:

1. **Third-Party Academic Literature PDFs (`sources/`, ~93 MB):**
   - In accordance with academic copyright guidelines, full-text PDFs by original authors (NASA SP-297, Raissi et al., Glorot & Bengio, etc.) are not redistributed in the repository.
   - Complete official persistent identifiers (DOIs, arXiv links, URLs) and exact cited page numbers are cataloged in [`references/README.md`](references/README.md).

2. **Large 3D Duct Discretization Grids (`3/data/duct_Re*.npz`, ~1.4 GB total):**
   - The 6 Stage 3 spatial grids ($12,035,304$ nodes each) are ~240 MB per file, exceeding GitHub's hard 100 MB per-file upload limit.
   - The exact finite-difference reference solver script [`3/generate_data.py`](3/generate_data.py) is included to regenerate these files locally.
   - Pre-computed arrays are also available as GitHub Release assets.

3. **Raw OpenFOAM Grid and Time-Step Fields (`openfoam_pinn_pipeline/cases/`, ~12.2 GB):**
   - Raw CFD case folders containing 19,700+ individual time-step field files from initial exploratory comparisons are excluded.
   - The validated Hornbeck-type implicit marching solvers in `1/`, `2/`, `3/`, `4/`, and `5/` generate the production reference solutions in seconds to minutes.

4. **Build and Local Artifacts:**
   - Intermediate LaTeX compilation cache files (`*.aux`, `*.toc`, `*.out`, `*.synctex.gz`).
   - Versioned backup snapshots (`backup/`).
