import os, re, shutil

ROOT = r"."
report_path = os.path.join(ROOT, "report.tex")
backup_dir = os.path.join(ROOT, "backup")

shutil.copyfile(report_path, os.path.join(backup_dir, "fix_all_before.tex"))

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# FIX 1 & 2: Stage 3 Re_n provenance and Pressure Scale Consistency
old_s1_p = r"At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.05\,\text{m/s}$, and dynamic pressure scale is $p_{\mathrm{scale}} = U_0^2 = 0.0025\,\text{m}^2/\text{s}^2$."
new_s1_p = r"At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.05\,\text{m/s}$, and dynamic pressure scale is $p_{\mathrm{scale}} = 0.5 U_0^2 = 0.00125\,\text{m}^2/\text{s}^2$ (% src: 1/model.py:54)."
assert old_s1_p in text, "old_s1_p not found"
text = text.replace(old_s1_p, new_s1_p)

old_s2_intro = r"Downstream, the profile converges to the Hagen--Poiseuille solution with peak centerline velocity $u_{\max} = 2 U_0$."
new_s2_intro = r"Downstream, the profile converges to the Hagen--Poiseuille solution with peak centerline velocity $u_{\max} = 2 U_0$. At $\mathrm{Re}=100$, bulk velocity is $U_0 = 0.10\,\text{m/s}$, and the pressure scale is chosen as the total Hagen--Poiseuille viscous pressure drop over the pipe length $p_{\mathrm{scale}} = \frac{32 \nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$ (% src: 2/model.py:51-53)."
assert old_s2_intro in text, "old_s2_intro not found"
text = text.replace(old_s2_intro, new_s2_intro)

old_s2_loss = r"""The composite loss function incorporates six terms:
\begin{equation}
\mathcal{L} = w_{\mathrm{data}}\mathcal{L}_{\mathrm{data}} + w_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}} + w_{\mathrm{inlet}}\mathcal{L}_{\mathrm{inlet}} + w_{\mathrm{wall}}\mathcal{L}_{\mathrm{wall}} + w_{\mathrm{axis}}\mathcal{L}_{\mathrm{axis}} + w_{\mathrm{outlet}}\mathcal{L}_{\mathrm{outlet}}.
\end{equation}"""

new_s2_loss = r"""The composite loss function incorporates six terms:
\begin{equation}
\mathcal{L} = w_{\mathrm{data}}\mathcal{L}_{\mathrm{data}} + w_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}} + w_{\mathrm{inlet}}\mathcal{L}_{\mathrm{inlet}} + w_{\mathrm{wall}}\mathcal{L}_{\mathrm{wall}} + w_{\mathrm{axis}}\mathcal{L}_{\mathrm{axis}} + w_{\mathrm{outlet}}\mathcal{L}_{\mathrm{outlet}}.
\end{equation}
The supervised data loss normalizes velocities by $U_0$ and pressure by $p_{\mathrm{scale}} = \frac{32\nu U_0 L}{D^2} = 0.128\,\text{m}^2/\text{s}^2$ (% src: 2/train_hybrid.py:180-183; 2/model.py:51):
\begin{equation}
\mathcal{L}_{\mathrm{data}} = \frac{1}{N_d}\sum_{i=1}^{N_d} \left[ \left(\frac{u_i - u_i^{\mathrm{ref}}}{U_0}\right)^2 + \left(\frac{v_i - v_i^{\mathrm{ref}}}{U_0}\right)^2 + 0.1\left(\frac{p_i - p_i^{\mathrm{ref}}}{p_{\mathrm{scale}}}\right)^2 \right].
\end{equation}"""
assert old_s2_loss in text, "old_s2_loss not found"
text = text.replace(old_s2_loss, new_s2_loss)

# FIX 3: Entrance length formulas - label correlation, source not verified
old_eq5_text = r"The hydrodynamic entrance length $L_e$ is computed using the Atkinson et al.~\cite{atkinson1969} correlation for parallel plates (as tabulated in Shah and London~\cite{shah1978}):"
new_eq5_text = r"The hydrodynamic entrance length $L_e$ for laminar flow between parallel plates is estimated using the composite correlation (correlation, source not verified; cf.~Atkinson et al.~\cite{atkinson1969} and Shah and London~\cite{shah1978}):"
assert old_eq5_text in text, "old_eq5_text not found"
text = text.replace(old_eq5_text, new_eq5_text)

old_eq23_text = r"The circular pipe hydrodynamic entrance length correlation from Chen~\cite{chen1973} (benchmarked against Hornbeck~\cite{hornbeck1973}) is:"
new_eq23_text = r"The circular pipe hydrodynamic entrance length correlation (correlation, source not verified; attributed to Chen~\cite{chen1973}, benchmarked against Hornbeck~\cite{hornbeck1973}) is:"
assert old_eq23_text in text, "old_eq23_text not found"
text = text.replace(old_eq23_text, new_eq23_text)

# FIX 4: Duct pressure-drop underprediction ratio provenance
old_abs = r"Stage~3 severely underpredicts total duct pressure drop by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, relative error $56.8\%$--$106.4\%$)"
new_abs = r"Stage~3 severely underpredicts total duct pressure drop by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, relative error $56.8\%$--$106.4\%$, measured as centerline gauge drop $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ from $x=0.05\,\text{m}$ to $5.0\,\text{m}$; % src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt)"
assert old_abs in text, "old_abs not found"
text = text.replace(old_abs, new_abs)

old_s58 = r"The PINN severely underpredicts the cumulative pressure drop by a factor of $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$) over the long duct domain ($L/D_h = 37.5$)."
new_s58 = r"The PINN severely underpredicts the cumulative pressure drop by a factor of $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$) over the long duct domain ($L/D_h = 37.5$), measured as the ratio of reference to PINN centerline gauge pressure drops $|\Delta p_{\mathrm{ref}}|/|\Delta p_{\mathrm{PINN}}|$ between $x=0.05\,\text{m}$ and $x=5.0\,\text{m}$ along $(y=H/2, z=W/2)$ (% src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt)."
assert old_s58 in text, "old_s58 not found"
text = text.replace(old_s58, new_s58)

old_conc = r"underpredicting total pumping head by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$)."
new_conc = r"underpredicting total pumping head by $3.3\times$ to $29.0\times$ ($15.9\times$ at $\mathrm{Re}=100$, evaluated across $x \in [0.05, 5.0]\,\text{m}$ along duct centerline; % src: 3/data/duct_Re*.npz, 3/checkpoints/duct_pinn.pt)."
assert old_conc in text, "old_conc not found"
text = text.replace(old_conc, new_conc)

# FIX 5: Geometry drawings overlap fix (TikZ Figs 1, 2, 4, 6, 8, 10)
old_fig1 = r"""\begin{tikzpicture}[node distance=1.6cm and 2.4cm, auto,
    stage/.style={rectangle, draw=blue!70, fill=blue!5, thick, rounded corners, minimum width=2.6cm, minimum height=1.1cm, align=center, font=\footnotesize},
    arrow/.style={thick, ->, >=stealth}]
    \node (s1) at (0, 0) [stage] {\textbf{Stage 1: 2D Channel}\\Laminar NS\\$L=4\,\text{m}, 2h=0.1\,\text{m}$};
    \node (s2) at (4.6, 0) [stage] {\textbf{Stage 2: Circular Pipe}\\Axisymmetric NS\\$L=4\,\text{m}, D=0.1\,\text{m}$};
    \node (s3) at (9.2, 0) [stage] {\textbf{Stage 3: 3D Duct}\\Parabolic Marching\\$L=5\,\text{m}, \mathrm{AR}=2.0$};
    \node (s4) at (0, -2.4) [stage] {\textbf{Stage 4: 2D Channel}\\RANS + SA Closure\\$L/h=160, \mathrm{Re}\le 1.2\times 10^5$};
    \node (s5) at (4.6, -2.4) [stage] {\textbf{Stage 5: Circular Pipe}\\RANS + SA Closure\\Superpipe Validated};
    \draw [arrow] (s1) -- node[above, font=\scriptsize] {Geometry} (s2);
    \draw [arrow] (s2) -- node[above, font=\scriptsize] {3D Effects} (s3);
    \draw [arrow] (s1) -- node[left, font=\scriptsize] {Turbulence} (s4);
    \draw [arrow] (s4) -- node[above, font=\scriptsize] {Geometry} (s5);
    \draw [arrow] (s2) -- node[right, font=\scriptsize] {Turbulence} (s5);
\end{tikzpicture}"""

new_fig1 = r"""\begin{tikzpicture}[
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
assert old_fig1 in text, "old_fig1 not found"
text = text.replace(old_fig1, new_fig1)

old_fig2 = r"""\begin{tikzpicture}[scale=0.95]
    \draw[thick] (0, 1.2) -- (7.0, 1.2);
    \node[above, font=\footnotesize] at (3.5, 1.2) {Upper wall: $y = +h = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[thick] (0, -1.2) -- (7.0, -1.2);
    \node[below, font=\footnotesize] at (3.5, -1.2) {Lower wall: $y = -h = -0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[dashed, gray] (0, 0) -- (7.0, 0);
    \node[right, font=\footnotesize] at (7.0, 0) {Centerline: $y = 0$};
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.2) rectangle (7.0, 1.4);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -1.4) rectangle (7.0, -1.2);
    \draw[blue, thick, ->] (-0.8, 0.8) -- (-0.1, 0.8);
    \draw[blue, thick, ->] (-0.8, 0.0) -- (-0.1, 0.0) node[midway, above, font=\footnotesize] {Uniform $U_0$};
    \draw[blue, thick, ->] (-0.8, -0.8) -- (-0.1, -0.8);
    \draw[thick] (0, -1.2) -- (0, 1.2) node[above, font=\footnotesize] {Inlet ($x = 0$)};
    \draw[thick] (7.0, -1.2) -- (7.0, 1.2) node[above, font=\footnotesize] {Outlet ($x = L = 4.0\,\text{m}, p = 0$)};
    \draw[domain=-1.2:1.2, smooth, variable=\y, red, thick] plot ({7.0 - 1.1*(1 - (\y/1.2)*(\y/1.2))}, {\y});
    \node[red, font=\scriptsize, left] at (5.8, 0) {$u(y) = 1.5 U_0 [1 - (y/h)^2]$};
    \draw[<->] (0, -1.7) -- (7.0, -1.7) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.1, -1.2) -- (-1.1, 1.2) node[midway, left, font=\footnotesize] {$2h = 0.10\,\text{m}$ ($D_h = 0.20\,\text{m}$)};
\end{tikzpicture}"""

new_fig2 = r"""\begin{tikzpicture}[scale=0.95]
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
assert old_fig2 in text, "old_fig2 not found"
text = text.replace(old_fig2, new_fig2)

old_fig4 = r"""\begin{tikzpicture}[scale=0.95]
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \node[above, font=\footnotesize] at (3.5, 1.5) {Pipe wall: $r = R = 0.05\,\text{m}$ (no-slip, $u=v=0$)};
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, 0) {Pipe axis: $r = 0$ (symmetry: $v=0, \partial u/\partial r = 0$)};
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \draw[blue, thick, ->] (-0.8, 1.0) -- (-0.1, 1.0);
    \draw[blue, thick, ->] (-0.8, 0.4) -- (-0.1, 0.4) node[midway, above, font=\footnotesize] {Uniform $U_0$};
    \draw[thick] (0, 0) -- (0, 1.5) node[above, font=\footnotesize] {Inlet ($x = 0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5) node[above, font=\footnotesize] {Outlet ($x = L = 4.0\,\text{m}, p = 0$)};
    \draw[domain=0:1.5, smooth, variable=\y, red, thick] plot ({7.0 - 1.2*(1 - (\y/1.5)*(\y/1.5))}, {\y});
    \node[red, font=\scriptsize, left] at (5.7, 0.4) {$u(r) = 2 U_0 [1 - (r/R)^2]$};
    \draw[<->] (0, -0.6) -- (7.0, -0.6) node[midway, below, font=\footnotesize] {$L = 4.0\,\text{m}$};
    \draw[<->] (-1.0, 0) -- (-1.0, 1.5) node[midway, left, font=\footnotesize] {$R = 0.05\,\text{m}$ ($D = 0.10\,\text{m}$)};
\end{tikzpicture}"""

new_fig4 = r"""\begin{tikzpicture}[scale=0.95]
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
assert old_fig4 in text, "old_fig4 not found"
text = text.replace(old_fig4, new_fig4)

old_fig6 = r"""\begin{tikzpicture}[scale=0.85]
    \draw[thick] (0,0) rectangle (2.5, 1.2);
    \node at (1.25, 0.6) [font=\footnotesize] {Inlet ($x=0$)};
    \draw[thick] (0,1.2) -- (4.5, 2.8) -- (7.0, 2.8) -- (2.5, 1.2);
    \draw[thick] (2.5,0) -- (7.0, 1.6) -- (7.0, 2.8);
    \draw[dashed] (0,0) -- (4.5, 1.6) -- (7.0, 1.6);
    \draw[dashed] (4.5,1.6) -- (4.5, 2.8);
    \node[above, font=\footnotesize] at (5.75, 2.8) {Outlet ($x = 5.0\,\text{m}$)};
    \draw[<->] (0, -0.4) -- (2.5, -0.4) node[midway, below, font=\footnotesize] {$W = 0.20\,\text{m}$};
    \draw[<->] (-0.4, 0) -- (-0.4, 1.2) node[midway, left, font=\footnotesize] {$H = 0.10\,\text{m}$};
    \draw[<->] (2.8, -0.2) -- (7.3, 1.4) node[midway, below right, font=\footnotesize] {$L = 5.0\,\text{m}$};
\end{tikzpicture}"""

new_fig6 = r"""\begin{tikzpicture}[scale=0.85]
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
assert old_fig6 in text, "old_fig6 not found"
text = text.replace(old_fig6, new_fig6)

old_fig8 = r"""\begin{tikzpicture}[scale=0.95]
    \draw[dashdotted, thick, gray] (0, 1.5) -- (7.0, 1.5);
    \node[above, font=\footnotesize] at (3.5, 1.5) {Centerline / Symmetry plane: $y/h = 1.0$ ($\partial U/\partial y = 0, V = 0, \partial \tilde{\nu}/\partial y = 0$)};
    \draw[thick] (0, 0) -- (7.0, 0);
    \fill[pattern=north east lines, pattern color=gray!60] (0, -0.2) rectangle (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, -0.2) {Solid wall: $y/h = 0$ (no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[thick] (0, 0) -- (0, 1.5) node[above, font=\footnotesize] {Inlet ($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5) node[above, font=\footnotesize] {Outlet ($x=160h$)};
    \node[blue, font=\scriptsize, align=center] at (3.5, 0.75) {Hard Boundary Envelope: $g(y) = \tanh(\mathrm{Re}_\tau y / 10)$\\Guarantees exact wall no-slip and zero eddy viscosity};
    \draw[<->] (0, -0.6) -- (7.0, -0.6) node[midway, below, font=\footnotesize] {$L/h = 160$};
    \draw[<->] (-0.8, 0) -- (-0.8, 1.5) node[midway, left, font=\footnotesize] {$h = 1.0$};
\end{tikzpicture}"""

new_fig8 = r"""\begin{tikzpicture}[scale=0.95]
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
assert old_fig8 in text, "old_fig8 not found"
text = text.replace(old_fig8, new_fig8)

old_fig10 = r"""\begin{tikzpicture}[scale=0.95]
    \draw[thick] (0, 1.5) -- (7.0, 1.5);
    \fill[pattern=north east lines, pattern color=gray!60] (0, 1.5) rectangle (7.0, 1.7);
    \node[above, font=\footnotesize] at (3.5, 1.7) {Pipe wall: $r/R = 1.0$ ($\eta = 0$, no-slip: $U=V=0, \tilde{\nu}=0$)};
    \draw[dashdotted, thick, gray] (0, 0) -- (7.0, 0);
    \node[below, font=\footnotesize] at (3.5, 0) {Centerline / Pipe axis: $r/R = 0$ ($V=0, \partial U/\partial r = 0, \partial \tilde{\nu}/\partial r = 0$)};
    \draw[thick] (0, 0) -- (0, 1.5) node[above, font=\footnotesize] {Inlet ($x=0$)};
    \draw[thick] (7.0, 0) -- (7.0, 1.5) node[above, font=\footnotesize] {Outlet ($x=160R$)};
    \draw[<->] (0, -0.6) -- (7.0, -0.6) node[midway, below, font=\footnotesize] {$L/R = 160$};
    \draw[<->] (-0.8, 0) -- (-0.8, 1.5) node[midway, left, font=\footnotesize] {$R = 1.0$ ($D = 2.0$)};
\end{tikzpicture}"""

new_fig10 = r"""\begin{tikzpicture}[scale=0.95]
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
assert old_fig10 in text, "old_fig10 not found"
text = text.replace(old_fig10, new_fig10)

# FIX 6: Figs 3 and 5 - Make each subfigure full width (stacked vertically)
old_fig3_block = r"""\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{stage1_error_u.png}
    \caption{Axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%].}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{stage1_error_p.png}
    \caption{Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%].}
\end{subfigure}
\caption{Stage 1: Linear-axis error distributions for developing laminar channel flow.}
\label{fig:stage1_errors}
\end{figure}"""

new_fig3_block = r"""\begin{figure}[H]
\centering
\begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.92\textwidth]{stage1_error_u.png}
    \caption{Axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%].}
\end{subfigure}
\par\vspace{0.3cm}
\begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.92\textwidth]{stage1_error_p.png}
    \caption{Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%].}
\end{subfigure}
\caption{Stage 1: Linear-axis error distributions for developing laminar channel flow across the Reynolds sweep.}
\label{fig:stage1_errors}
\end{figure}"""
assert old_fig3_block in text, "old_fig3_block not found"
text = text.replace(old_fig3_block, new_fig3_block)

old_fig5_block = r"""\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{stage2_error_u.png}
    \caption{Axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%].}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{stage2_error_p.png}
    \caption{Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%].}
\end{subfigure}
\caption{Stage 2: Axisymmetric error maps for circular pipe entrance flow.}
\label{fig:stage2_errors}
\end{figure}"""

new_fig5_block = r"""\begin{figure}[H]
\centering
\begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.92\textwidth]{stage2_error_u.png}
    \caption{Axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%].}
\end{subfigure}
\par\vspace{0.3cm}
\begin{subfigure}{\textwidth}
    \centering
    \includegraphics[width=0.92\textwidth]{stage2_error_p.png}
    \caption{Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%].}
\end{subfigure}
\caption{Stage 2: Error heatmaps for developing laminar axisymmetric pipe flow across the Reynolds sweep.}
\label{fig:stage2_errors}
\end{figure}"""
assert old_fig5_block in text, "old_fig5_block not found"
text = text.replace(old_fig5_block, new_fig5_block)

# FIX 8: Listings cleanup (delete dangling comments)
target1 = "used in both source reports (Sec. 4.2 hybrid report / Sec. 5.4,7\n    physics-only report), which keeps tanh activations away from"
repl1 = "used in both source reports, which keeps tanh activations away from"
assert target1 in text, "target1 not found"
text = text.replace(target1, repl1)

target2 = "(avoids the float32 round-off problem discussed in the physics-only\n        report Sec. 7.2, since these ratios are all O(1) regardless of Re)."
repl2 = "(avoids float32 round-off issues since these ratios are all O(1) regardless of Re)."
assert target2 in text, "target2 not found"
text = text.replace(target2, repl2)

old_c3 = r"        p_scale = 0.5 * U_ref ** 2 * (config.L_DOMAIN / config.DH)  # bug fix, see module docstring"
new_c3 = r"        p_scale = 0.5 * U_ref ** 2 * (config.L_DOMAIN / config.DH)"
assert old_c3 in text, "old_c3 not found"
text = text.replace(old_c3, new_c3)

old_c4 = "# ----------------------------------------------------------------------------------\n# SA terms (overflow-safe: documented in model specifications)\n# ----------------------------------------------------------------------------------"
assert old_c4 in text, "old_c4 not found"
text = text.replace(old_c4, "")

# FIX 9: Preamble placeins
if r"\usepackage{placeins}" not in text:
    text = text.replace(r"\usepackage{float}", "\\usepackage{float}\n\\usepackage{placeins}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Applied all fixes to report.tex successfully.")
