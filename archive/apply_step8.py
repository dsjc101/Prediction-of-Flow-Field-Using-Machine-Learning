path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Polish Abstract
old_abs = '''\\begin{abstract}
% src: 1/outputs/metrics_full_domain.csv, 2/outputs/metrics_summary.json, 3/figures/global_summary.txt, 4/outputs/final_metrics.csv, 5/outputs/eval_dense_final_test.json
This report presents a five-stage numerical investigation of Physics-Informed Neural Networks (PINNs) applied to developing internal fluid flows. The pipeline progresses from laminar flow between parallel flat plates (Stage~1) and in circular pipes (Stage~2) to three-dimensional rectangular ducts (Stage~3, aspect ratio 2.0). It then examines high-speed turbulent flows governed by the Reynolds-Averaged Navier--Stokes (RANS) equations closed with the Spalart--Allmaras (SA) eddy-viscosity model in channels (Stage~4) and pipes (Stage~5, Reynolds number $10^4$ to $1.2\\times 10^5$). Stage~6 (turbulent rectangular duct) was attempted but not completed because of time; no checkpoints or results exist, so none are reported. 

In laminar flow, streamwise velocity errors range from $2.20\\%$ to $5.69\\%$ across channels and pipes, and $8.81\\%$ to $12.69\\%$ in the 3D duct. In turbulent flow, hard boundary envelopes enforce exact wall conditions, yielding velocity errors of $0.36\\%$ to $0.58\\%$ and matching Princeton Superpipe experimental profiles within $2.81\\%$ to $4.18\\%$. However, audits reveal two critical failures: Stage~3 underpredicts the total duct pressure drop by $3.3\\times$ to $29.0\\times$ ($15.9\\times$ at $\\mathrm{Re}=100$, relative error $56.8\\%$ to $106.4\\%$), and Stage~4 suffers complete eddy viscosity extinction (working variable $\\tilde{\\nu} \\approx 0$, error $>99\\%$) because its relative physics loss weight ($w_{\\mathrm{pde}}/w_{\\mathrm{data}} \\approx 5.0\\times 10^{-9}$) is $200{,}000\\times$ smaller than in Stage~5 ($1.0\\times 10^{-3}$).
\\end{abstract}'''

new_abs = '''\\begin{abstract}
% src: 1/outputs/metrics_full_domain.csv, 2/outputs/metrics_summary.json, 3/figures/global_summary.txt, 4/outputs/final_metrics.csv, 5/outputs/eval_dense_final_test.json
This report presents a five-stage numerical investigation of Physics-Informed Neural Networks (PINNs) applied to developing internal fluid flows across canonical conduit geometries. The benchmark evaluates laminar developing flows in parallel plates (Stage~1), circular pipes (Stage~2), and 3D rectangular ducts (Stage~3, aspect ratio 2.0). It extends to turbulent regimes using Reynolds-Averaged Navier--Stokes (RANS) with Spalart--Allmaras (SA) closure in channels (Stage~4) and pipes (Stage~5, $\\mathrm{Re} = 10^4$ to $1.2\\times 10^5$). Stage~6 (turbulent duct) was attempted but not completed because of time; no checkpoints or results exist, so none are reported.

Streamwise velocity errors span $2.20\\%$--$5.69\\%$ in laminar conduit flows and $0.36\\%$--$0.58\\%$ in turbulent configurations, where predictions match Princeton Superpipe Pitot traverses within $2.81\\%$--$4.18\\%$. However, audits reveal two critical failures: Stage~3 severely underpredicts total duct pressure drop by $3.3\\times$ to $29.0\\times$ ($15.9\\times$ at $\\mathrm{Re}=100$, relative error $56.8\\%$--$106.4\\%$), and Stage~4 suffers complete eddy viscosity extinction (working variable $\\tilde{\\nu} \\approx 0$, error $>99\\%$) because its relative physics loss weight ($w_{\\mathrm{pde}}/w_{\\mathrm{data}} \\approx 5.08\\times 10^{-9}$) is $200{,}000\\times$ smaller than in Stage~5 ($1.12\\times 10^{-3}$).
\\end{abstract}'''

assert old_abs in text, 'old_abs not found'
text = text.replace(old_abs, new_abs)

# 2. Stage 4 Subsection C
old_s4_c = '''\\subsection{C. Reference Truth Marching Solver and Discretization}
Reference truth data was computed with an independent parabolized RANS marching solver (in file \\texttt{4/reference\\_solver.py}) using the SA-noft2 closure with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}. The solver drops streamwise diffusion ($\\partial^2/\\partial x^2 = 0$) and transverse pressure variation ($P = P(x)$ only). The cross-stream grid comprises $n_y = 400$ wall-stretched finite-volume cells with first cell spacing $y^+ < 1.0$, marched across $n_x = 800$ axial stations. Global mass flux is maintained by pressure-gradient superposition.'''

new_s4_c = '''\\subsection{C. Reference Truth Marching Solver and Discretization}
Reference truth data was computed with an independent parabolized RANS marching solver (\\texttt{4/reference\\_}\\allowbreak\\texttt{solver.py}) using the SA-noft2 closure with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}. The solver solves the thin-shear-layer (parabolic) boundary-layer equations:
\\begin{align}
\\frac{\\partial U}{\\partial x} + \\frac{\\partial V}{\\partial y} &= 0, \\\\
U \\frac{\\partial U}{\\partial x} + V \\frac{\\partial U}{\\partial y} &= -\\frac{dP}{dx} + \\frac{\\partial}{\\partial y}\\left(\\nu_{\\mathrm{eff}}\\frac{\\partial U}{\\partial y}\\right), \\\\
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial y} &= c_{b1}\\tilde{S}\\tilde{\\nu} - c_{w1}f_w\\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 + \\frac{1}{\\sigma}\\frac{\\partial}{\\partial y}\\left((\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial y}\\right) + \\frac{c_{b2}}{\\sigma}\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial y}\\right)^2.
\\end{align}
% src: 4/reference_solver.py:10-35; 4/config.py:33-35
\\noindent\\textbf{Terms Dropped in Parabolic Formulation:} The reference marching scheme drops all streamwise diffusion ($\\partial^2 U/\\partial x^2 = 0, \\partial^2 \\tilde{\\nu}/\\partial x^2 = 0$) and neglects transverse pressure variation ($\\partial P/\\partial y = 0 \\implies P = P(x)$). Constant mass flux $\\int_0^1 U\\,dy = 1$ is enforced by decomposing velocity into particular and homogeneous fields ($U = U_a - \\frac{dP}{dx} U_b$).

\\noindent\\textbf{Spatial Discretization and Grid:} Numerical integration employs an implicit backward-Euler marching scheme in $x$ with Picard iteration for nonlinear coefficients. The spatial grid consists of $N_x = 800$ axial stations (power-law clustered toward the inlet) and $N_y = 400$ wall-normal finite-volume cells (tanh-stretched with first cell $y^+ < 1.0$).

\\noindent\\textbf{Structural Modeling Distinction:} Crucially, while the training and evaluation data originate from this parabolic boundary-layer solver, the PINN is optimized against the full elliptic RANS system (Eqs.~44--46). In the entrance region ($x/h < 10$), where non-negligible transverse pressure gradients and streamwise diffusion exist, this physical mismatch constitutes an intrinsic modeling discrepancy between reference data and PINN physics loss.'''

assert old_s4_c in text, 'old_s4_c not found'
text = text.replace(old_s4_c, new_s4_c)

# 3. Stage 5 Subsection B & C
old_s5_b_c = '''The SA working variable $\\tilde{\\nu}$ satisfies:
\\begin{equation}
\\begin{split}
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial r} = c_{b1} \\tilde{S} \\tilde{\\nu} - c_{w1} f_w \\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 &+ \\frac{1}{\\sigma}\\left[ \\frac{\\partial}{\\partial x}\\left((\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right) + \\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r (\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right) \\right. \\\\
&\\quad\\left. +\\, c_{b2}\\left(\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right)^2 + \\left(\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right)^2\\right) \\right],
\\end{split}
\\end{equation}
where $d = 1 - r$ is distance to the pipe wall, and Pirozzoli et al.~\\cite{pirozzoli2021} provides pipe DNS statistics.

\\subsection{C. Reference Marching Solver and Discretization}
Reference truth data was computed with an independent axisymmetric marching code (\\texttt{5/reference\\_}\\allowbreak\\texttt{solver.py}) on an $800 \\times 400$ wall-stretched grid ($N_X = 800, N_R = 400$). The reference solver drops streamwise diffusion ($\\partial^2/\\partial x^2 = 0$) and assumes $P = P(x)$ only.'''

new_s5_b_c = '''The SA working variable $\\tilde{\\nu}$ satisfies:
\\begin{equation}
\\begin{split}
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial r} = c_{b1} \\tilde{S} \\tilde{\\nu} - c_{w1} f_w \\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 &+ \\frac{1}{\\sigma}\\left[ \\frac{\\partial}{\\partial x}\\left((\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right) + \\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r (\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right) \\right. \\\\
&\\quad\\left. +\\, c_{b2}\\left(\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right)^2 + \\left(\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right)^2\\right) \\right].
\\end{split}
\\end{equation}
The eddy viscosity is $\\nu_t = \\tilde{\\nu} f_{v1}$ and $\\nu_{\\mathrm{eff}} = \\nu + \\nu_t$. Standard NASA Turbulence Modeling Resource closure definitions in cylindrical coordinates are:
\\begin{align}
\\chi &= \\frac{\\tilde{\\nu}}{\\nu}, \\quad f_{v1} = \\frac{\\chi^3}{\\chi^3 + c_{v1}^3}, \\quad f_{v2} = 1 - \\frac{\\chi}{1 + \\chi f_{v1}}, \\\\
\\tilde{S} &= S + \\frac{\\tilde{\\nu}}{\\kappa^2 d^2} f_{v2}, \\quad S = \\left|\\frac{\\partial U}{\\partial r} - \\frac{\\partial V}{\\partial x}\\right|, \\quad r_d = \\min\\left(\\frac{\\tilde{\\nu}}{\\tilde{S}\\kappa^2 d^2}, 10\\right), \\\\
g_w &= r_d + c_{w2}(r_d^6 - r_d), \\quad f_w = g_w \\left(\\frac{1 + c_{w3}^6}{g_w^6 + c_{w3}^6}\\right)^{1/6},
\\end{align}
where $d = 1 - r$ denotes normal distance to the pipe wall, and empirical model constants are $c_{b1} = 0.1355$, $c_{b2} = 0.622$, $\\sigma = 2/3$, $\\kappa = 0.41$, $c_{w1} = c_{b1}/\\kappa^2 + (1+c_{b2})/\\sigma$, $c_{w2} = 0.3$, $c_{w3} = 2.0$, and $c_{v1} = 7.1$. Pipe DNS benchmarks by Pirozzoli et al.~\\cite{pirozzoli2021} confirm these parameters.

\\subsection{C. Reference Marching Solver and Discretization}
Reference truth data was computed with an independent axisymmetric marching code (\\texttt{5/reference\\_}\\allowbreak\\texttt{solver.py}) on an $800 \\times 400$ wall-stretched grid ($N_X = 800, N_R = 400$). The reference solver solves the parabolized cylindrical boundary-layer equations:
\\begin{align}
\\frac{\\partial U}{\\partial x} + \\frac{1}{r}\\frac{\\partial(r V)}{\\partial r} &= 0, \\\\
U \\frac{\\partial U}{\\partial x} + V \\frac{\\partial U}{\\partial r} &= -\\frac{dP}{dx} + \\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r \\nu_{\\mathrm{eff}}\\frac{\\partial U}{\\partial r}\\right), \\\\
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial r} &= c_{b1}\\tilde{S}\\tilde{\\nu} - c_{w1}f_w\\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 + \\frac{1}{\\sigma}\\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r (\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right) + \\frac{c_{b2}}{\\sigma}\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right)^2.
\\end{align}
% src: 5/reference_solver.py:10-40; 5/config.py:30-32
\\noindent\\textbf{Terms Dropped in Parabolic Formulation:} The marching solver drops all axial diffusion ($\\partial^2 U/\\partial x^2 = 0, \\partial^2 \\tilde{\\nu}/\\partial x^2 = 0$) and drops radial momentum balance ($\\partial P/\\partial r = 0 \\implies P = P(x)$). Constant volumetric mass flux $2\\int_0^1 U r\\,dr = 1$ is maintained via pressure-gradient superposition ($U = U_a - \\frac{dP}{dx} U_b$).

\\noindent\\textbf{Discretization and Computational Grid:} Spatial discretization uses implicit axial marching with a damped Levenberg--Marquardt Newton iteration at each streamwise station. The computational mesh comprises $N_X = 800$ axial stations (power-law clustered near the inlet) and $N_R = 400$ wall-stretched radial cells (tanh-stretched with $y^+ < 1.0$ at $r=1.0$).

\\noindent\\textbf{Structural Modeling Distinction:} Similar to Stage 4, the reference solver is strictly parabolic, whereas the PINN enforces the full elliptic cylindrical RANS system (Eqs.~58--60). This model difference is a recognized source of residual error in the immediate development zone near $x=0$.'''

assert old_s5_b_c in text, 'old_s5_b_c not found'
text = text.replace(old_s5_b_c, new_s5_b_c)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Applied all Step 8 updates to report.tex')

