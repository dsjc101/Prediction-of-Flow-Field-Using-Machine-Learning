import os, re, shutil

ROOT = r"."
report_path = os.path.join(ROOT, "report.tex")
backup_dir = os.path.join(ROOT, "backup")

# Backup before applying fixes
shutil.copyfile(report_path, os.path.join(backup_dir, "fix_5fixes_before.tex"))

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------------------------------------------------------
# FIX 1: Restore truncated text and move mid-line `% src:` to their own lines
# -----------------------------------------------------------------------------

# Abstract
old_abs_block = r"""Streamwise velocity errors span $2.20\%$--$5.69\%$ in laminar conduit flows and $0.36\%$--$0.58\%$ in turbulent configurations, where predictions match Princeton Superpipe Pitot traverses within $2.81\%$--$4.18\%$. However, audits reveal two critical failures: Stage~3 severely underpredicts total duct pressure drop by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, relative error $56.8\%$--$106.4\%$, measured as centerline gauge drop $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ from $x=0.05\,\text{m}$ to $5.0\,\text{m}$; % src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt), and Stage~4 suffers complete eddy viscosity extinction (working variable $\tilde{\nu} \approx 0$, error $>99\%$) because its relative physics loss weight ($w_{\mathrm{pde}}/w_{\mathrm{data}} \approx 5.08\times 10^{-9}$) is $200{,}000\times$ smaller than in Stage~5 ($1.12\times 10^{-3}$)."""

new_abs_block = r"""Streamwise velocity errors span $2.20\%$--$5.69\%$ in laminar conduit flows and $0.36\%$--$0.58\%$ in turbulent configurations, where predictions match Princeton Superpipe Pitot traverses within $2.81\%$--$4.18\%$. However, audits reveal two critical failures: Stage~3 severely underpredicts total duct pressure drop by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, relative error $56.8\%$--$106.4\%$, measured as centerline gauge drop $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ from $x=0.05\,\text{m}$ to $5.0\,\text{m}$), and Stage~4 suffers complete eddy viscosity extinction (working variable $\tilde{\nu} \approx 0$, error $>99\%$) because its relative physics loss weight ($w_{\mathrm{pde}}/w_{\mathrm{data}} \approx 5.08\times 10^{-9}$) is about $200{,}000\times$ smaller than in Stage~5 ($1.12\times 10^{-3}$).
% src: 3/data/duct_Re*.npz; 3/checkpoints/duct_pinn.pt; 4/outputs/final_metrics.csv; 4/outputs/best_weights.json; 5/outputs/best_weights_dense.json"""

assert old_abs_block in text, "old_abs_block not found"
text = text.replace(old_abs_block, new_abs_block)

# Stage 1 Section 3.1
old_s1_p = r"At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.05\,\text{m/s}$, and dynamic pressure scale is $p_{\mathrm{scale}} = 0.5 U_0^2 = 0.00125\,\text{m}^2/\text{s}^2$ (% src: 1/model.py:54). The training set covers $\mathrm{Re} \in \{50, 100, 200, 500\}$; $\mathrm{Re}=300$ tests interpolation, and $\mathrm{Re}=600$ tests extrapolation."
new_s1_p = r"""At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.05\,\text{m/s}$, and dynamic pressure scale is $p_{\mathrm{scale}} = 0.5 U_0^2 = 0.00125\,\text{m}^2/\text{s}^2$. The training set covers $\mathrm{Re} \in \{50, 100, 200, 500\}$; $\mathrm{Re}=300$ tests interpolation, and $\mathrm{Re}=600$ tests extrapolation.
% src: 1/model.py:54"""
assert old_s1_p in text, "old_s1_p not found"
text = text.replace(old_s1_p, new_s1_p)

# Stage 2 Section 4.1
old_s2_p = r"Downstream, the profile converges to the Hagen--Poiseuille solution with peak centerline velocity $u_{\max} = 2 U_0$. At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.10\,\text{m/s}$, and the pressure scale is chosen as the total Hagen--Poiseuille viscous pressure drop over the pipe length $p_{\mathrm{scale}} = \frac{32 \nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$ (% src: 2/model.py:51-53)."
new_s2_p = r"""Downstream, the profile converges to the Hagen--Poiseuille solution with peak centerline velocity $u_{\max} = 2 U_0$. At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.10\,\text{m/s}$, and the pressure scale is chosen as the total Hagen--Poiseuille viscous pressure drop over the pipe length $p_{\mathrm{scale}} = \frac{32 \nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$.
% src: 2/model.py:51-53"""
assert old_s2_p in text, "old_s2_p not found"
text = text.replace(old_s2_p, new_s2_p)

# Stage 2 Section 4.5
old_s2_loss = r"The supervised data loss normalizes velocities by $U_0$ and pressure by $p_{\mathrm{scale}} = \frac{32\nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$ (% src: 2/train_hybrid.py:180-183; 2/model.py:51):"
new_s2_loss = r"""The supervised data loss normalizes velocities by $U_0$ and pressure by $p_{\mathrm{scale}} = \frac{32\nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$:
% src: 2/train_hybrid.py:180-183; 2/model.py:51"""
assert old_s2_loss in text, "old_s2_loss not found"
text = text.replace(old_s2_loss, new_s2_loss)

# Stage 3 Section 5.4
old_s3_arch = r"Training uses Adam for 2500 iterations ($8.0\times 10^{-4}$ decayed via StepLR every 500 steps) followed by 300 iterations of L-BFGS polish. % src: 3/train_hybrid.py:147-148"
new_s3_arch = r"""Training uses Adam for 2500 iterations ($8.0\times 10^{-4}$ decayed via StepLR every 500 steps) followed by 300 iterations of L-BFGS polish.
% src: 3/train_hybrid.py:147-148"""
assert old_s3_arch in text, "old_s3_arch not found"
text = text.replace(old_s3_arch, new_s3_arch)

# Stage 3 Section 5.8
old_s3_fail = r"""    \item \textbf{Total Pressure Drop Failure:} As documented in Table~\ref{tab:stage3_metrics}, kinematic pressure exhibits large relative $L_2$ errors ranging from $\mathbf{56.8\%}$ to $\mathbf{106.4\%}$ . The PINN severely underpredicts the cumulative pressure drop by a factor of $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$) over the long duct domain ($L/D_h = 37.5$), measured as the ratio of reference to PINN centerline gauge pressure drops $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ between $x=0.05\,\text{m}$ and $x=5.0\,\text{m}$ along $(y=H/2, z=W/2)$ (% src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt). The optimization failed to enforce the global axial pressure gradient necessary to overcome cumulative wall friction."""
new_s3_fail = r"""    \item \textbf{Total Pressure Drop Failure:} As documented in Table~\ref{tab:stage3_metrics}, kinematic pressure exhibits large relative $L_2$ errors ranging from $\mathbf{56.8\%}$ to $\mathbf{106.4\%}$ . The PINN severely underpredicts the cumulative pressure drop by a factor of $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$) over the long duct domain ($L/D_h = 37.5$), measured as the ratio of reference to PINN centerline gauge pressure drops $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ between $x=0.05\,\text{m}$ and $x=5.0\,\text{m}$ along $(y=H/2, z=W/2)$. The optimization failed to enforce the global axial pressure gradient necessary to overcome cumulative wall friction.
% src: 3/data/duct_Re*.npz; 3/checkpoints/duct_pinn.pt"""
assert old_s3_fail in text, "old_s3_fail not found"
text = text.replace(old_s3_fail, new_s3_fail)

# Stage 4 Section 6.4
old_s4_env = r"where $v_s = 0.05 \sqrt{0.5 \times 0.073 (\mathrm{Re}/2)^{-0.25}}$, $b_{\mathrm{nt}} = -3.0$, and $\nu = 4/\mathrm{Re}$. This guarantees $U=V=\tilde{\nu}=0$ identically at $y=0$ for any network parameter values. % src: 4/train.py:39,139; 4/weight_search.py:98"
new_s4_env = r"""where $v_s = 0.05 \sqrt{0.5 \times 0.073 (\mathrm{Re}/2)^{-0.25}}$, $b_{\mathrm{nt}} = -3.0$, and $\nu = 4/\mathrm{Re}$. This guarantees $U=V=\tilde{\nu}=0$ identically at $y=0$ for any network parameter values.
% src: 4/train.py:39,139; 4/weight_search.py:98"""
assert old_s4_env in text, "old_s4_env not found"
text = text.replace(old_s4_env, new_s4_env)

# Conclusions Section 9
old_conc_s3 = r"""    \item \textbf{Duct Pressure Drop Failure:} Stage 3 exhibited a complete failure in predicting duct pressure drop ($56.8\%$--$106.4\%$ relative error), underpredicting total pumping head by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, evaluated across $x \in [0.05, 5.0]\,\text{m}$ along duct centerline; % src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt). Centerline velocity errors peaked at the inlet core ($26\%$--$68\%$). PINNs struggle to capture cumulative axial pressure drops in long ducts ($L/D_h = 37.5$) without specialized loss weighting."""
new_conc_s3 = r"""    \item \textbf{Duct Pressure Drop Failure:} Stage 3 exhibited a complete failure in predicting duct pressure drop ($56.8\%$--$106.4\%$ relative error), underpredicting total pumping head by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, evaluated across $x \in [0.05, 5.0]\,\text{m}$ along duct centerline). Centerline velocity errors peaked at the inlet core ($26\%$--$68\%$). PINNs struggle to capture cumulative axial pressure drops in long ducts ($L/D_h = 37.5$) without specialized loss weighting.
% src: 3/data/duct_Re*.npz; 3/checkpoints/duct_pinn.pt"""
assert old_conc_s3 in text, "old_conc_s3 not found"
text = text.replace(old_conc_s3, new_conc_s3)


# -----------------------------------------------------------------------------
# FIX 2: Stage 3 Re_n normalization formula and text consistency
# -----------------------------------------------------------------------------
old_s3_ren = r"""The Stage 3 network (\texttt{3/pinn\_model.py}, class \texttt{DuctPINN}) takes four normalized inputs:
\begin{equation}
\hat{x} = \frac{x}{L}, \quad \hat{y} = \frac{y}{H}, \quad \hat{z} = \frac{z}{W}, \quad \mathrm{Re}_n = 2\frac{\log_{10}(\mathrm{Re}) - \log_{10}(50)}{\log_{10}(600) - \log_{10}(50)} - 1.
\end{equation}"""

new_s3_ren = r"""The Stage 3 network (\texttt{3/pinn\_model.py}, class \texttt{DuctPINN}) takes four normalized inputs. The Reynolds number embedding $\mathrm{Re}_n$ is normalized over the full evaluation sweep $\mathrm{Re} \in [50, 600]$ ($\mathrm{Re}_{\min}=50, \mathrm{Re}_{\max}=600$, yielding $\mathrm{Re}_n = -0.4421$ at $\mathrm{Re}=100$; code \texttt{3/pinn\_model.py:27}):
\begin{equation}
\label{eq:stage3_renorm}
\hat{x} = \frac{x}{L}, \quad \hat{y} = \frac{y}{H}, \quad \hat{z} = \frac{z}{W}, \quad \mathrm{Re}_n = 2\frac{\log_{10}(\mathrm{Re}) - \log_{10}(50)}{\log_{10}(600) - \log_{10}(50)} - 1 \in [-1, 1].
\end{equation}"""
assert old_s3_ren in text, "old_s3_ren not found"
text = text.replace(old_s3_ren, new_s3_ren)


# -----------------------------------------------------------------------------
# FIX 3: Geometry TikZ diagrams overhaul (Figs 1, 2, 5, 8, 10, 12)
# -----------------------------------------------------------------------------

# Fig 1: Pipeline Flowchart
old_fig1 = r"""\begin{tikzpicture}[
    stage/.style={rectangle, draw=blue!70, fill=blue!5, thick, rounded corners, minimum width=2.8cm, minimum height=1.2cm, align=center, font=\footnotesize},
    arrow/.style={thick, ->, >=stealth}]
    \node (s1) at (0, 0) [stage] {\textbf{Stage 1: 2D Channel}\\Laminar NS\\$L=4\,\text{m}, 2h=0.1\,\text{m}$};
    \node (s2) at (4.8, 0) [stage] {\textbf{Stage 2: Circular Pipe}\\Axisymmetric NS\\$L=4\,\text{m}, D=0.1\,\text{m}$};
    \node (s3) at (9.6, 0) [stage] {\textbf{Stage 3: 3D Duct}\\Parabolic Marching\\$L=5\,\text{m}, \mathrm{AR}=2.0$};
    \node (s4) at (0, -2.6) [stage] {\textbf{Stage 4: 2D Channel}\\RANS + SA Closure\\$L/h=160, \mathrm{Re}\le 1.2\times 10^5$};
    \node (s5) at (4.8, -2.6) [stage] {\textbf{Stage 5: Circular Pipe}\\RANS + SA Closure\\Superpipe Validated};
    \draw [arrow] (s1) -- node[midway, above=2pt, font=\scriptsize\bfseries] {Cross-Section} (s2);
    \draw [arrow] (s2) -- node[midway, above=2pt, font=\scriptsize\bfseries] {3D Walls} (s3);
    \draw [arrow] (s1) -- node[midway, left=2pt, font=\scriptsize\bfseries] {Turbulence} (s4);
    \draw [arrow] (s4) -- node[midway, above=2pt, font=\scriptsize\bfseries] {Cross-Section} (s5);
    \draw [arrow] (s2) -- node[midway, right=2pt, font=\scriptsize\bfseries] {Turbulence} (s5);
\end{tikzpicture}"""

new_fig1 = r"""\begin{tikzpicture}[
    stage/.style={rectangle, draw=blue!70, fill=blue!5, thick, rounded corners, minimum width=3.0cm, minimum height=1.25cm, align=center, font=\footnotesize},
    arrow/.style={thick, ->, >=stealth}]
    \node (s1) at (0, 0) [stage] {\textbf{Stage 1: 2D Channel}\\Laminar NS\\$L=4\,\text{m}, 2h=0.1\,\text{m}$};
    \node (s2) at (5.2, 0) [stage] {\textbf{Stage 2: Circular Pipe}\\Axisymmetric NS\\$L=4\,\text{m}, D=0.1\,\text{m}$};
    \node (s3) at (10.4, 0) [stage] {\textbf{Stage 3: 3D Duct}\\Parabolic Marching\\$L=5\,\text{m}, \mathrm{AR}=2.0$};
    \node (s4) at (0, -3.2) [stage] {\textbf{Stage 4: 2D Channel}\\RANS + SA Closure\\$L/h=160, \mathrm{Re}\le 1.2\times 10^5$};
    \node (s5) at (5.2, -3.2) [stage] {\textbf{Stage 5: Circular Pipe}\\RANS + SA Closure\\Superpipe Validated};
    \draw [arrow] (s1) -- node[midway, fill=white, inner sep=2pt, font=\scriptsize\bfseries] {Cross-Section} (s2);
    \draw [arrow] (s2) -- node[midway, fill=white, inner sep=2pt, font=\scriptsize\bfseries] {3D Walls} (s3);
    \draw [arrow] (s1) -- node[midway, fill=white, inner sep=2pt, font=\scriptsize\bfseries] {Turbulence} (s4);
    \draw [arrow] (s4) -- node[midway, fill=white, inner sep=2pt, font=\scriptsize\bfseries] {Cross-Section} (s5);
    \draw [arrow] (s2) -- node[midway, fill=white, inner sep=2pt, font=\scriptsize\bfseries] {Turbulence} (s5);
\end{tikzpicture}"""
assert old_fig1 in text, "old_fig1 not found"
text = text.replace(old_fig1, new_fig1)

# Fig 2: Stage 1 Geometry
old_fig2 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.45) {Upper wall: $y = +h = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[thick] (0, 1.2) -- (7.0, 1.2);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.2) rectangle (7.0, 1.4);
    \draw[thick] (0, -1.2) -- (7.0, -1.2);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -1.4) rectangle (7.0, -1.2);
    \node[below, font=\footnotesize] at (3.5, -1.45) {Lower wall: $y = -h = -0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[dashed, gray] (0, 0) -- (7.0, 0);
    \draw[thick] (0, -1.2) -- (0, 1.2);
    \node[left, font=\footnotesize, align=center] at (-0.15, 0.4) {Inlet\\($x = 0$)};
    \draw[blue, thick, ->] (-0.8, 0.8) -- (-0.1, 0.8);
    \draw[blue, thick, ->] (-0.8, 0.0) -- (-0.1, 0.0);
    \draw[blue, thick, ->] (-0.8, -0.8) -- (-0.1, -0.8);
    \node[blue, font=\scriptsize, left] at (-0.8, 0.0) {Uniform $U_0$};
    \draw[thick] (7.0, -1.2) -- (7.0, 1.2);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.6) {Outlet\\($x = L = 4.0\,\text{m}, p = 0$)};
    \node[right, font=\footnotesize] at (7.1, -0.6) {Centerline: $y = 0$};
    \draw[domain=-1.2:1.2, smooth, variable=\y, red, thick] plot ({7.0 - 1.1*(1 - (\y/1.2)*(\y/1.2))}, {\y});
    \node[red, font=\scriptsize, left] at (5.8, 0) {$u(y) = 1.5 U_0 [1 - (y/h)^2]$};
    \draw[<->] (0, -1.9) -- (7.0, -1.9) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.8, -1.2) -- (-1.8, 1.2) node[midway, left, font=\footnotesize] {$2h = 0.10\,\text{m}$ ($D_h = 0.20\,\text{m}$)};
\end{tikzpicture}"""

new_fig2 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.6) {Upper wall: $y = +h = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[thick] (0, 1.2) -- (7.0, 1.2);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.2) rectangle (7.0, 1.4);
    \draw[thick] (0, -1.2) -- (7.0, -1.2);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -1.4) rectangle (7.0, -1.2);
    \node[below, font=\footnotesize] at (3.5, -1.6) {Lower wall: $y = -h = -0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[dashed, gray] (0, 0) -- (7.0, 0);
    \draw[thick] (0, -1.2) -- (0, 1.2);
    \draw[blue, thick, ->] (-0.9, 0.7) -- (-0.1, 0.7);
    \draw[blue, thick, ->] (-0.9, 0.0) -- (-0.1, 0.0);
    \draw[blue, thick, ->] (-0.9, -0.7) -- (-0.1, -0.7);
    \node[blue, font=\scriptsize, above] at (-0.5, 0.7) {Uniform $U_0$};
    \node[font=\footnotesize, below] at (-0.5, -0.85) {Inlet ($x=0$)};
    \draw[thick] (7.0, -1.2) -- (7.0, 1.2);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.6) {Outlet\\($x = L = 4.0\,\text{m}, p = 0$)};
    \node[right, font=\footnotesize] at (7.1, -0.6) {Centerline: $y = 0$};
    \draw[domain=-1.2:1.2, smooth, variable=\y, red, thick] plot ({7.0 - 1.1*(1 - (\y/1.2)*(\y/1.2))}, {\y});
    \node[red, font=\scriptsize, fill=white, inner sep=2pt, left] at (5.7, 0.5) {$u(y) = 1.5 U_0 [1 - (y/h)^2]$};
    \draw[<->] (0, -2.1) -- (7.0, -2.1) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.9, -1.2) -- (-1.9, 1.2) node[midway, left=2pt, font=\footnotesize] {$2h = 0.10\,\text{m}$ ($D_h = 0.20\,\text{m}$)};
\end{tikzpicture}"""
assert old_fig2 in text, "old_fig2 not found"
text = text.replace(old_fig2, new_fig2)

# Fig 5: Stage 2 Geometry (Pipe)
old_fig4 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.75) {Pipe wall: $r = R = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.15) {Pipe axis: $r = 0$ (symmetry: $v=0, \partial u/\partial r = 0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \node[left, font=\footnotesize, align=center] at (-0.15, 0.75) {Inlet\\($x = 0$)};
    \draw[blue, thick, ->] (-0.8, 1.1) -- (-0.1, 1.1);
    \draw[blue, thick, ->] (-0.8, 0.4) -- (-0.1, 0.4);
    \node[blue, font=\scriptsize, left] at (-0.8, 0.75) {Uniform $U_0$};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x = L = 4.0\,\text{m}, p = 0$)};
    \draw[domain=0:1.5, smooth, variable=\y, red, thick] plot ({7.0 - 1.2*(1 - (\y/1.5)*(\y/1.5))}, {\y});
    \node[red, font=\scriptsize, left] at (5.7, 0.5) {$u(r) = 2 U_0 [1 - (r/R)^2]$};
    \draw[<->] (0, -0.65) -- (7.0, -0.65) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.7, 0) -- (-1.7, 1.5) node[midway, left, font=\footnotesize] {$R = 0.05\,\text{m}$ ($D = 0.10\,\text{m}$)};
\end{tikzpicture}"""

new_fig4 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.9) {Pipe wall: $r = R = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.25) {Pipe axis: $r = 0$ (symmetry: $v=0, \partial u/\partial r = 0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \draw[blue, thick, ->] (-0.9, 1.0) -- (-0.1, 1.0);
    \draw[blue, thick, ->] (-0.9, 0.5) -- (-0.1, 0.5);
    \node[blue, font=\scriptsize, above] at (-0.5, 1.0) {Uniform $U_0$};
    \node[font=\footnotesize, below] at (-0.5, 0.25) {Inlet ($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x = L = 4.0\,\text{m}, p = 0$)};
    \draw[domain=0:1.5, smooth, variable=\y, red, thick] plot ({7.0 - 1.2*(1 - (\y/1.5)*(\y/1.5))}, {\y});
    \node[red, font=\scriptsize, fill=white, inner sep=2pt, left] at (5.6, 0.75) {$u(r) = 2 U_0 [1 - (r/R)^2]$};
    \draw[<->] (0, -0.75) -- (7.0, -0.75) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.8, 0) -- (-1.8, 1.5) node[midway, left=2pt, font=\footnotesize] {$R = 0.05\,\text{m}$ ($D = 0.10\,\text{m}$)};
\end{tikzpicture}"""
assert old_fig4 in text, "old_fig4 not found"
text = text.replace(old_fig4, new_fig4)

# Fig 8: Stage 3 Geometry (Duct)
old_fig6 = r"""\begin{tikzpicture}[scale=0.85]
    \draw[thick] (0,0) rectangle (2.5, 1.2);
    \node[left, font=\footnotesize, align=center] at (-0.1, 0.6) {Inlet\\($x = 0$)};
    \draw[thick] (0,1.2) -- (4.5, 2.8) -- (7.0, 2.8) -- (2.5, 1.2);
    \node[above, font=\footnotesize] at (3.5, 2.2) {Top wall: $y = H = 0.10\,\text{m}$ (no-slip)};
    \draw[thick] (2.5,0) -- (7.0, 1.6) -- (7.0, 2.8);
    \draw[dashed] (0,0) -- (4.5, 1.6) -- (7.0, 1.6);
    \draw[dashed] (4.5,1.6) -- (4.5, 2.8);
    \node[right, font=\footnotesize, align=left] at (7.1, 2.2) {Outlet\\($x = 5.0\,\text{m}$)};
    \draw[<->] (0, -0.4) -- (2.5, -0.4) node[midway, below, font=\footnotesize] {$W = 0.20\,\text{m}$};
    \draw[<->] (-0.5, 0) -- (-0.5, 1.2) node[midway, left, font=\footnotesize] {$H = 0.10\,\text{m}$};
    \draw[<->] (2.8, -0.2) -- (7.3, 1.4) node[midway, below right, font=\footnotesize] {$L = 5.0\,\text{m}$};
\end{tikzpicture}"""

new_fig6 = r"""\begin{tikzpicture}[scale=0.85]
    \draw[thick] (0,0) rectangle (2.5, 1.2);
    \node at (1.25, 0.6) [font=\footnotesize, fill=white, inner sep=2pt] {Inlet ($x=0$)};
    \draw[thick] (0,1.2) -- (4.5, 2.8) -- (7.0, 2.8) -- (2.5, 1.2);
    \node[above, font=\footnotesize, fill=white, inner sep=2pt] at (3.5, 2.3) {Top wall: $y = H = 0.10\,\text{m}$ (no-slip)};
    \draw[thick] (2.5,0) -- (7.0, 1.6) -- (7.0, 2.8);
    \draw[dashed] (0,0) -- (4.5, 1.6) -- (7.0, 1.6);
    \draw[dashed] (4.5,1.6) -- (4.5, 2.8);
    \node[right, font=\footnotesize, align=left] at (7.1, 2.2) {Outlet\\($x = 5.0\,\text{m}$)};
    \draw[<->] (0, -0.5) -- (2.5, -0.5) node[midway, below, font=\footnotesize] {$W = 0.20\,\text{m}$};
    \draw[<->] (-0.7, 0) -- (-0.7, 1.2) node[midway, left=2pt, font=\footnotesize] {$H = 0.10\,\text{m}$};
    \draw[<->] (2.8, -0.3) -- (7.3, 1.3) node[midway, below right, font=\footnotesize] {$L = 5.0\,\text{m}$};
\end{tikzpicture}"""
assert old_fig6 in text, "old_fig6 not found"
text = text.replace(old_fig6, new_fig6)

# Fig 10: Stage 4 Geometry (Turbulent Channel)
old_fig8 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.65) {Centerline / Symmetry plane: $y/h = 1.0$ ($\partial U/\partial y = 0, V = 0, \partial \tilde{\nu}/\partial y = 0$)};
    \draw[dashdotted, thick, gray] (0, 1.5) -- (7.0, 1.5);
    \draw[thick] (0, 0) -- (7.0, 0);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -0.2) rectangle (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.25) {Solid wall: $y/h = 0$ (no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \node[left, font=\footnotesize, align=center] at (-0.15, 0.75) {Inlet\\($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x=160h$)};
    \node[blue, font=\scriptsize, align=center] at (3.5, 0.75) {Hard Boundary Envelope: $g(y) = \tanh(\mathrm{Re}_\tau y / 10)$\\Guarantees exact wall no-slip and zero eddy viscosity};
    \draw[<->] (0, -0.7) -- (7.0, -0.7) node[midway, below, font=\footnotesize] {$L/h = 160$};
    \draw[<->] (-1.3, 0) -- (-1.3, 1.5) node[midway, left, font=\footnotesize] {$h = 1.0$};
\end{tikzpicture}"""

new_fig8 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.7) {Centerline / Symmetry plane: $y/h = 1.0$ ($\partial U/\partial y = 0, V = 0, \partial \tilde{\nu}/\partial y = 0$)};
    \draw[dashdotted, thick, gray] (0, 1.5) -- (7.0, 1.5);
    \draw[thick] (0, 0) -- (7.0, 0);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -0.2) rectangle (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.3) {Solid wall: $y/h = 0$ (no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \draw[blue, thick, ->] (-0.9, 1.0) -- (-0.1, 1.0);
    \draw[blue, thick, ->] (-0.9, 0.5) -- (-0.1, 0.5);
    \node[blue, font=\scriptsize, above] at (-0.5, 1.0) {Inlet flow};
    \node[font=\footnotesize, below] at (-0.5, 0.3) {Inlet ($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x=160h$)};
    \node[blue, font=\scriptsize, fill=white, inner sep=3pt, align=center] at (3.5, 0.75) {Hard Boundary Envelope: $g(y) = \tanh(\mathrm{Re}_\tau y / 10)$\\Guarantees exact wall no-slip and zero eddy viscosity};
    \draw[<->] (0, -0.8) -- (7.0, -0.8) node[midway, below, font=\footnotesize] {$L/h = 160$};
    \draw[<->] (-1.8, 0) -- (-1.8, 1.5) node[midway, left=2pt, font=\footnotesize] {$h = 1.0$};
\end{tikzpicture}"""
assert old_fig8 in text, "old_fig8 not found"
text = text.replace(old_fig8, new_fig8)

# Fig 12: Stage 5 Geometry (Turbulent Pipe)
old_fig10 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.85) {Pipe wall: $r/R = 1.0$ ($\eta = 0$, no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.2) {Centerline / Pipe axis: $r/R = 0$ ($V=0, \partial U/\partial r = 0, \partial \tilde{\nu}/\partial r = 0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \node[left, font=\footnotesize, align=center] at (-0.15, 0.75) {Inlet\\($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x=160R$)};
    \draw[<->] (0, -0.7) -- (7.0, -0.7) node[midway, below, font=\footnotesize] {$L/R = 160$};
    \draw[<->] (-1.3, 0) -- (-1.3, 1.5) node[midway, left, font=\footnotesize] {$R = 1.0$ ($D = 2.0$)};
\end{tikzpicture}"""

new_fig10 = r"""\begin{tikzpicture}[scale=0.95]
    \node[above, font=\footnotesize] at (3.5, 1.9) {Pipe wall: $r/R = 1.0$ ($\eta = 0$, no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.25) {Centerline / Pipe axis: $r/R = 0$ ($V=0, \partial U/\partial r = 0, \partial \tilde{\nu}/\partial r = 0$)};
    \draw[thick] (0, 0) -- (0, 1.5);
    \draw[blue, thick, ->] (-0.9, 1.0) -- (-0.1, 1.0);
    \draw[blue, thick, ->] (-0.9, 0.5) -- (-0.1, 0.5);
    \node[blue, font=\scriptsize, above] at (-0.5, 1.0) {Inlet flow};
    \node[font=\footnotesize, below] at (-0.5, 0.3) {Inlet ($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5);
    \node[right, font=\footnotesize, align=left] at (7.1, 0.75) {Outlet\\($x=160R$)};
    \draw[<->] (0, -0.8) -- (7.0, -0.8) node[midway, below, font=\footnotesize] {$L/R = 160$};
    \draw[<->] (-1.8, 0) -- (-1.8, 1.5) node[midway, left=2pt, font=\footnotesize] {$R = 1.0$ ($D = 2.0$)};
\end{tikzpicture}"""
assert old_fig10 in text, "old_fig10 not found"
text = text.replace(old_fig10, new_fig10)


# -----------------------------------------------------------------------------
# FIX 4: Small text fixes
# -----------------------------------------------------------------------------
# (a) Eq. 23 caption
old_eq23 = r"The circular pipe hydrodynamic entrance length correlation (correlation, source not verified; attributed to Chen~\cite{chen1973}, benchmarked against Hornbeck~\cite{hornbeck1973}) is:"
new_eq23 = r"The circular pipe hydrodynamic entrance length correlation is computed via the standard engineering correlation (not verified against~\cite{chen1973}):"
assert old_eq23 in text, "old_eq23 not found"
text = text.replace(old_eq23, new_eq23)

# (b) Listings cleanup in Appendix A
old_listing_c1 = "used in both source reports, which keeps tanh activations away from"
new_listing_c1 = "which keeps tanh activations away from"
assert old_listing_c1 in text, "old_listing_c1 not found"
text = text.replace(old_listing_c1, new_listing_c1)

old_listing_c2 = r"""        # Xavier-normal init, standard for tanh PINNs (Raissi et al. 2019
        # practice; explicitly called out in the hybrid report's
        # Architecture table)."""
new_listing_c2 = r"""        # Xavier-normal init, standard for tanh PINNs (Raissi et al. 2019 practice)."""
assert old_listing_c2 in text, "old_listing_c2 not found"
text = text.replace(old_listing_c2, new_listing_c2)

# (c) Plain-English explanation of Stage 3 p_scale
old_s3_pscale_intro = r"Normalized input vector evaluates to $[0.5000, 0.5000, 0.5000, -0.4421]$. With $U_{\mathrm{ref}} = 0.075\,\text{m/s}$ and $p_{\mathrm{scale}} = 0.5 U_{\mathrm{ref}}^2 (L / D_h) = 0.10547\,\text{m}^2/\text{s}^2$, physical predictions are:"
new_s3_pscale_intro = r"""Unlike Stage~1 (dynamic pressure $q = 0.5 U_0^2$) and Stage~2 (total viscous Hagen--Poiseuille drop $32\nu U_0 L / D^2$), the Stage~3 pressure normalization scale $p_{\mathrm{scale}} = 0.5 U_{\mathrm{ref}}^2 (L / D_h) = 0.10547\,\text{m}^2/\text{s}^2$ scales dynamic pressure by the duct slenderness ratio $L / D_h = 37.5$ to characterize cumulative frictional pressure drop across the extended duct passage. Normalized input vector evaluates to $[0.5000, 0.5000, 0.5000, -0.4421]$. With $U_{\mathrm{ref}} = 0.075\,\text{m/s}$ and $p_{\mathrm{scale}} = 0.10547\,\text{m}^2/\text{s}^2$, physical predictions are:"""
assert old_s3_pscale_intro in text, "old_s3_pscale_intro not found"
text = text.replace(old_s3_pscale_intro, new_s3_pscale_intro)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Applied 5 fixes to report.tex successfully.")

