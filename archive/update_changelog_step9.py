import os

changelog_path = r'.\CHANGELOG.md'
with open(changelog_path, 'r', encoding='utf-8') as f:
    content = f.read()

categorized_section = """

---

## 4. Final Categorized Audit Status

### DONE
- **Complete 9-Step Verification & Overhaul:**
  - **Step 1 (`truth.csv`):** Generated source-of-truth database linking 98 parameters and metrics across Stages 1–5 directly to code, config, logs, and checkpoint tensors.
  - **Step 2 (Training Consistency):** Purged obsolete references to Cosine Annealing, GradNorm, and unverified epoch counts (50,000, 30,000). Rebuilt Table 16 (`tables/table_c4_weights.tex`) directly from `truth.csv`.
  - **Step 3 (Mathematical & Envelopes Consistency):** Updated network formulations, $[ -1, 1 ]$ input feature scaling, exact $\log_{10}\mathrm{Re}$ parameters (mid 4.4515, half 0.4515), exact hard boundary envelopes ($U = g(1+a_0)$, $V = v_s g^2 a_1$ and $V = (V_{\mathrm{scale}} u_\tau) g^2 r a_1$, $\tilde{\nu} = \nu \mathrm{Re}_\tau g \operatorname{softplus}(a_3 + b)$), and unified pressure scaling formulas across text, loss, and forward pass tables. Executed deterministic forward pass verification via `verify_forward_pass.py` for all 5 stages.
  - **Step 4 (Number Reconciliation & Scientific Tone):** Rebuilt Table 11 test pressure error ranges ($0.38\%-0.58\%$ for Stage 4, $0.65\%-2.57\%$ for Stage 5). Replaced qualitative pressure claims with measured ratios ($3.3\times-29.0\times$ underprediction). Added Stage 1 local pointwise error quantiles (median $2.42\%-3.48\%$, P95 $6.82\%-11.88\%$, P99 $9.93\%-14.83\%$). Softened assertions from "proves" to "demonstrating that the physics-informed loss acts as an effective regularizer".
  - **Step 5 (Clickable References & Citations):** Fully audited all 19 bibliographic entries in `refs.bib` and `link_audit.csv` with verified working DOIs and URLs. Added clickable titles `\href{...}{...}` with visible provenance tags (`[\textsc{pdf} p.~N]`, `[\textsc{doi}: ...]`, `[\textsc{nasa} ...]`). Corrected citations to Atkinson et al. / Shah & London for parallel-plate correlation and Chen / Hornbeck for tube entry.
  - **Step 6 (Figure Overhaul & Normalization):** Normalized duct analytical comparison curves by $U_{\mathrm{mean}}$ in Fig 19, demonstrating proper convergence toward the Shah & London 1.992 asymptote. Expanded Fig 11 into an 8-panel ($2 \times 4$) matrix validating against all 8 Princeton Superpipe Pitot traverse datasets ($2.81\%-4.18\%$). Capped error map colorbars at local panel 99th percentile (P99) to eliminate yellow washout.
  - **Step 7 (Layout & Margin Polish):** Wrapped wide tables (Tables 11 & 12) in `\resizebox{\textwidth}{!}`. Wrapped layer activation tables (Tables 1, 5, 7, 9) in `\resizebox{\linewidth}{!}`. Formatted long file paths and split composite weight equations. Formatted Title Page: Author set to "Farzi", Academic Supervisor set to "Prof. Subhadip Das". Eliminated all Overfull `\hbox` warnings $>1\,\text{pt}$ (0 warnings remain).
  - **Step 8 (Missing Content & Plain-English Abstract):** Added explicit thin-shear-layer parabolic equations, dropped terms, computational grid ($800 \times 400$), solver file paths, and structural modeling distinctions (parabolic reference vs. elliptic PINN) to Stage 4 and Stage 5 Section C. Defined all Spalart--Allmaras symbols ($\tilde{S}, f_{v1}, f_w, d, c_{b1}, c_{b2}, c_{w1}, \sigma, \kappa, c_{v1}, c_{w2}, c_{w3}$) in Stage 5. Rewrote abstract to exactly 165 words in plain English, honestly reporting both critical failures, Superpipe validation, and Stage 6 status.
  - **Step 9 (Final Compilation & Verification):** Successfully compiled twice with `pdflatex` (44 pages, 11,949,276 bytes). Confirmed 0 undefined references, 0 undefined citations, and 0 overfull boxes $>1\,\text{pt}$.

### CHANGED
- **Author Identity:** Updated title page metadata from "Project Researcher" / "Project Advisor" to Author "Farzi" and Academic Supervisor "Prof. Subhadip Das".
- **Duct Pressure Narrative:** Corrected inaccurate claim that "pressure drop closely tracks reference data" and "2x to 5x underprediction" to measured underprediction ratios ($3.31\times$ at $\mathrm{Re}=600$ up to $28.97\times$ at $\mathrm{Re}=50$; $15.92\times$ at $\mathrm{Re}=100$).
- **Turbulence Working Variable Loss Mechanism:** Replaced speculative "cylindrical envelope core regularization" explanation with the quantitatively verified loss weight ratio $w_{\mathrm{pde}}/w_{\mathrm{data}}$ ($5.08\times 10^{-9}$ in Stage 4 vs. $1.12\times 10^{-3}$ in Stage 5, a $200{,}000\times$ disparity).
- **Master Summary Pressure Error Bounds:** Updated Table 11 test pressure error ranges from $0.21\%-0.35\%$ to $0.38\%-0.58\%$ (Stage 4) and from $0.18\%-0.28\%$ to $0.65\%-2.57\%$ (Stage 5) to match underlying test set evaluations.
- **Superpipe Error Range:** Reconciled Superpipe relative $L_2$ error bounds across the abstract, text, and Table 15 to the verified range $2.81\%-4.18\%$.

### NOT DONE
- **Stage 6 (3D Turbulent Rectangular Duct):** Attempted but not completed because of time constraints; no checkpoints, evaluation logs, or result archives exist in the codebase, so none are reported in the document.

### UNVERIFIED
- **Cylindrical Envelope Core Regularization:** The hypothesis that cylindrical geometry regularizes core eddy viscosity independently of loss weighting was unverified by controlled ablation and has been removed from the report.
"""

if "## 4. Final Categorized Audit Status" not in content:
    with open(changelog_path, 'a', encoding='utf-8') as f:
        f.write(categorized_section)
    print("SUCCESS: Appended categorized audit status to CHANGELOG.md")
else:
    print("ALREADY PRESENT: Categorized section already exists in CHANGELOG.md")

