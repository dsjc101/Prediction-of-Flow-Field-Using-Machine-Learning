import os

path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Title page
old_author = '''\\author{\\textbf{Project Researcher}\\\\[0.3em]
\\normalsize Computational Fluid Mechanics and Scientific Machine Learning Laboratory\\\\[0.2em]
\\small Academic Supervisor: Project Advisor}'''

new_author = '''\\author{\\textbf{Farzi}\\\\[0.3em]
\\normalsize Computational Fluid Mechanics and Scientific Machine Learning Laboratory\\\\[0.2em]
\\small Academic Supervisor: Prof. Subhadip Das}'''

assert old_author in text, 'old_author not found'
text = text.replace(old_author, new_author)

# 2. Line 77
old_77 = 'Traditional computational approaches rely on spatial discretization: finite-difference marching schemes for parabolic entrance equations~\\cite{bodoia1961,hornbeck1973} or elliptic finite-volume solvers (such as OpenFOAM~\\cite{openfoam2019}). While effective, mesh-based techniques require separate computational grids for each geometric configuration and re-meshing across parametric operating sweeps.'
new_77 = 'Traditional computational approaches rely on spatial discretization: finite-difference marching schemes for parabolic entrance equations~\\cite{bodoia1961,hornbeck1973} or elliptic finite-volume packages such as OpenFOAM~\\cite{openfoam2019}. While effective, mesh-based techniques require separate computational grids for each geometric configuration and re-meshing across parametric operating sweeps.'
assert old_77 in text, 'old_77 not found'
text = text.replace(old_77, new_77)

# 3. Line 105
old_105 = '\\item \\textbf{Stage 3 (Laminar 3D Rectangular Duct):} Rectangular passage ($L=5.0\\,\\text{m}, H=0.10\\,\\text{m}, W=0.20\\,\\text{m}$, aspect ratio $\\mathrm{AR}=2.0$, $D_h=0.1333\\,\\text{m}$), incorporating two-dimensional wall shear interactions.'
new_105 = '\\item \\textbf{Stage 3 (Laminar 3D Rectangular Duct):} Rectangular passage ($L=5.0\\,\\text{m}$, $H=0.10\\,\\text{m}$, $W=0.20\\,\\text{m}$, aspect ratio $\\mathrm{AR}=2.0$, $D_h=0.1333\\,\\text{m}$), incorporating two-dimensional wall shear interactions.'
assert old_105 in text, 'old_105 not found'
text = text.replace(old_105, new_105)

# 4. Line 228
old_228 = 'The architecture comprises four hidden layers of 80 neurons with $\\tanh$ activation functions ($[3, 80, 80, 80, 80, 3]$). Trainable weights and biases total:'
new_228 = 'The architecture comprises four hidden layers of 80 neurons with $\\tanh$ activations (layer dimensions $[3, 80, 80, 80, 80, 3]$). Trainable weights and biases total:'
assert old_228 in text, 'old_228 not found'
text = text.replace(old_228, new_228)

# 5. Table 1 (Stage 1 activations)
old_tab1 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.0579, 0.1571, -0.0009, -0.1015]$ & $[-0.0579, 0.1558, -0.0009, -0.1011]$ & 3.43 & 0.96 \\\\
Hidden 2 & $[-0.7194, 0.7283, 0.0744, -0.5888]$ & $[-0.6166, 0.6220, 0.0742, -0.5290]$ & 11.30 & 0.70 \\\\
Hidden 3 & $[-1.4141, -2.8177, 0.9572, 2.0372]$ & $[-0.8884, -0.9929, 0.7430, 0.9666]$ & 12.00 & 0.51 \\\\
Hidden 4 & $[1.2799, -1.5198, 3.7344, 0.2901]$ & $[0.8565, -0.9087, 0.9989, 0.2822]$ & 12.12 & 0.40 \\\\
Output & $[1.2933, 0.0168, 16.6964]$ & N/A & 5.44 & 0.29 \\\\
\\bottomrule
\\end{tabular}'''

new_tab1 = '''\\begin{table}[H]
\\centering
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.0579, 0.1571, -0.0009, -0.1015]$ & $[-0.0579, 0.1558, -0.0009, -0.1011]$ & 3.43 & 0.96 \\\\
Hidden 2 & $[-0.7194, 0.7283, 0.0744, -0.5888]$ & $[-0.6166, 0.6220, 0.0742, -0.5290]$ & 11.30 & 0.70 \\\\
Hidden 3 & $[-1.4141, -2.8177, 0.9572, 2.0372]$ & $[-0.8884, -0.9929, 0.7430, 0.9666]$ & 12.00 & 0.51 \\\\
Hidden 4 & $[1.2799, -1.5198, 3.7344, 0.2901]$ & $[0.8565, -0.9087, 0.9989, 0.2822]$ & 12.12 & 0.40 \\\\
Output & $[1.2933, 0.0168, 16.6964]$ & N/A & 5.44 & 0.29 \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab1 in text, 'old_tab1 not found'
text = text.replace(old_tab1, new_tab1)

# 6. Line 274
old_274 = 'Normalized coordinates evaluate to $\\hat{x} = 0.5/4.0 = 0.125$, $\\hat{y} = 0.02/0.05 = 0.400$, and $\\mathrm{Re}_n = 2\\log_{10}(100/50)/1.0 - 1 = -0.3979$. The network outputs normalized vector $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$. Scaling by $U_0 = 0.05\\,\\text{m/s}$ and dynamic pressure $p_{\\mathrm{scale}} = 0.5 U_0^2 = 0.00125\\,\\text{m}^2/\\text{s}^2$:'
new_274 = 'Normalized coordinates evaluate to $\\hat{x} = 0.5/4.0 = 0.125$, $\\hat{y} = 0.02/0.05 = 0.400$, and $\\mathrm{Re}_n = 2\\log_{10}(100/50)/1.0 - 1 = -0.3979$. The network outputs normalized vector $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$.\nRescaling by $U_0 = 0.05\\,\\text{m/s}$ and dynamic pressure scale $p_{\\mathrm{scale}} = 0.5 U_0^2 = 0.00125\\,\\text{m}^2/\\text{s}^2$ yields:'
assert old_274 in text, 'old_274 not found'
text = text.replace(old_274, new_274)

# 7. Line 377
old_377 = 'The axis symmetry penalty is $\\mathcal{L}_{\\mathrm{axis}} = \\frac{1}{N_{\\mathrm{ax}}}\\sum (v_k/U_0)^2$. The final loss weights read from \\texttt{2/train\\_hybrid.py} are:'
new_377 = 'The axis symmetry penalty is $\\mathcal{L}_{\\mathrm{axis}} = \\frac{1}{N_{\\mathrm{ax}}}\\sum (v_k/U_0)^2$.\nThe final loss weights read from \\texttt{2/train\\_hybrid.py} are:'
assert old_377 in text, 'old_377 not found'
text = text.replace(old_377, new_377)

# 8. Line 412
old_412 = 'Table~\\ref{tab:stage2_metrics} reports quantitative errors ordered monotonically by Reynolds number ($\\mathrm{Re} = 50, 100, 200, 300, 500, 600$).'
new_412 = 'Table~\\ref{tab:stage2_metrics} reports quantitative errors ordered monotonically across all Reynolds numbers ($\\mathrm{Re} \\in \\{50, 100, 200, 300, 500, 600\\}$).'
assert old_412 in text, 'old_412 not found'
text = text.replace(old_412, new_412)

# 9. Line 496 (Stage 3 weights)
old_496 = 'w_{u} = 8.733, \\; w_{v} = 0.110, \\; w_{w} = 11.109, \\; w_{p} = 2.421, \\; w_{\\mathrm{pde}} = 32.183, \\; w_{\\mathrm{in}} = 0.683, \\; w_{\\mathrm{wall}} = 11.138, \\; w_{\\mathrm{out}} = 0.639.'
new_496 = '''\\begin{aligned}
w_{u} &= 8.733, \\quad & w_{v} &= 0.110, \\quad & w_{w} &= 11.109, \\quad & w_{p} &= 2.421, \\\\
w_{\\mathrm{pde}} &= 32.183, \\quad & w_{\\mathrm{in}} &= 0.683, \\quad & w_{\\mathrm{wall}} &= 11.138, \\quad & w_{\\mathrm{out}} &= 0.639.
\\end{aligned}'''
assert old_496 in text, 'old_496 not found'
text = text.replace(old_496, new_496)

# 10. Table 5 (Stage 3 activations)
old_tab5 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2847, -0.1165, 0.4132, 0.0802]$ & $[-0.2773, -0.1160, 0.3912, 0.0801]$ & 2.93 & 0.65 \\\\
Hidden 2 & $[-0.1891, 0.2480, -0.0711, 0.0232]$ & $[-0.1869, 0.2430, -0.0710, 0.0232]$ & 9.47 & 0.44 \\\\
Hidden 3 & $[0.6916, -0.3750, -0.1283, 0.2216]$ & $[0.5990, -0.3584, -0.1276, 0.2181]$ & 9.36 & 0.26 \\\\
Hidden 4 & $[0.1684, -0.4821, -0.1735, -0.1191]$ & $[0.1668, -0.4479, -0.1717, -0.1185]$ & 9.12 & 0.16 \\\\
Output & $[2.1008, -0.0022, -0.0020, 0.3058]$ & N/A & 2.96 & 0.04 \\\\
\\bottomrule
\\end{tabular}'''

new_tab5 = '''\\begin{table}[H]
\\centering
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2847, -0.1165, 0.4132, 0.0802]$ & $[-0.2773, -0.1160, 0.3912, 0.0801]$ & 2.93 & 0.65 \\\\
Hidden 2 & $[-0.1891, 0.2480, -0.0711, 0.0232]$ & $[-0.1869, 0.2430, -0.0710, 0.0232]$ & 9.47 & 0.44 \\\\
Hidden 3 & $[0.6916, -0.3750, -0.1283, 0.2216]$ & $[0.5990, -0.3584, -0.1276, 0.2181]$ & 9.36 & 0.26 \\\\
Hidden 4 & $[0.1684, -0.4821, -0.1735, -0.1191]$ & $[0.1668, -0.4479, -0.1717, -0.1185]$ & 9.12 & 0.16 \\\\
Output & $[2.1008, -0.0022, -0.0020, 0.3058]$ & N/A & 2.96 & 0.04 \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab5 in text, 'old_tab5 not found'
text = text.replace(old_tab5, new_tab5)

# 11. Line 604
old_604 = 'Reference truth data was computed with an independent parabolized RANS marching solver (\\texttt{4/reference\\_solver.py}) using the SA-noft2 model with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
new_604 = 'Reference truth data was computed with an independent parabolized RANS marching solver (\\texttt{4/reference\\_}\\allowbreak\\texttt{solver.py}) using the SA-noft2 model with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
assert old_604 in text, 'old_604 not found'
text = text.replace(old_604, new_604)

# 12. Table 7 (Stage 4 activations)
old_tab7 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2403, 0.0214, 0.1577, -0.3019]$ & $[-0.2358, 0.0214, 0.1564, -0.2931]$ & 3.50 & 1.41 \\\\
Hidden 2 & $[-0.1327, -0.3392, -1.7185, -1.1004]$ & $[-0.1319, -0.3267, -0.9377, -0.8007]$ & 11.07 & 1.22 \\\\
Hidden 3 & $[13.9724, -1.0994, 3.3107, -2.2269]$ & $[1.0000, -0.8003, 0.9973, -0.9770]$ & 11.47 & 0.81 \\\\
Hidden 4 & $[2.8991, 3.1455, -6.1401, 2.1948]$ & $[0.9940, 0.9963, -1.0000, 0.9755]$ & 10.35 & 0.97 \\\\
Output & $[0.0830, 0.0649, 0.7981, -5.8677]$ & N/A & 5.09 & 0.52 \\\\
\\bottomrule
\\end{tabular}'''

new_tab7 = '''\\begin{table}[H]
\\centering
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2403, 0.0214, 0.1577, -0.3019]$ & $[-0.2358, 0.0214, 0.1564, -0.2931]$ & 3.50 & 1.41 \\\\
Hidden 2 & $[-0.1327, -0.3392, -1.7185, -1.1004]$ & $[-0.1319, -0.3267, -0.9377, -0.8007]$ & 11.07 & 1.22 \\\\
Hidden 3 & $[13.9724, -1.0994, 3.3107, -2.2269]$ & $[1.0000, -0.8003, 0.9973, -0.9770]$ & 11.47 & 0.81 \\\\
Hidden 4 & $[2.8991, 3.1455, -6.1401, 2.1948]$ & $[0.9940, 0.9963, -1.0000, 0.9755]$ & 10.35 & 0.97 \\\\
Output & $[0.0830, 0.0649, 0.7981, -5.8677]$ & N/A & 5.09 & 0.52 \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab7 in text, 'old_tab7 not found'
text = text.replace(old_tab7, new_tab7)

# 13. Line 726 (Stage 5 SA transport eq)
old_726 = '''\\begin{equation}
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial r} = c_{b1} \\tilde{S} \\tilde{\\nu} - c_{w1} f_w \\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 + \\frac{1}{\\sigma}\\left[ \\frac{\\partial}{\\partial x}\\left((\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right) + \\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r (\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right) + c_{b2}\\left(\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right)^2 + \\left(\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right)^2\\right) \\right],
\\end{equation}'''

new_726 = '''\\begin{equation}
\\begin{split}
U \\frac{\\partial \\tilde{\\nu}}{\\partial x} + V \\frac{\\partial \\tilde{\\nu}}{\\partial r} = c_{b1} \\tilde{S} \\tilde{\\nu} - c_{w1} f_w \\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 &+ \\frac{1}{\\sigma}\\left[ \\frac{\\partial}{\\partial x}\\left((\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right) + \\frac{1}{r}\\frac{\\partial}{\\partial r}\\left(r (\\nu+\\tilde{\\nu})\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right) \\right. \\\\
&\\quad\\left. +\\, c_{b2}\\left(\\left(\\frac{\\partial \\tilde{\\nu}}{\\partial x}\\right)^2 + \\left(\\frac{\\partial \\tilde{\\nu}}{\\partial r}\\right)^2\\right) \\right],
\\end{split}
\\end{equation}'''
assert old_726 in text, 'old_726 not found'
text = text.replace(old_726, new_726)

# 14. Line 731
old_731 = 'Reference truth data was computed with an independent axisymmetric marching code (\\texttt{5/reference\\_solver.py}) on an $800 \\times 400$ wall-stretched grid ($N_X = 800, N_R = 400$).'
new_731 = 'Reference truth data was computed with an independent axisymmetric marching code (\\texttt{5/reference\\_}\\allowbreak\\texttt{solver.py}) on an $800 \\times 400$ wall-stretched grid ($N_X = 800, N_R = 400$).'
assert old_731 in text, 'old_731 not found'
text = text.replace(old_731, new_731)

# 15. Line 763
old_763 = 'Table~\\ref{tab:stage5_act} reports the deterministic forward pass evaluated from checkpoint \\texttt{5/outputs/models/dense/final\\_seed0.pt} at $(x=40R, r=0.5R, \\mathrm{Re}=40{,}000)$.'
new_763 = 'Table~\\ref{tab:stage5_act} reports the deterministic forward pass evaluated from checkpoint \\texttt{5/outputs/models/}\\allowbreak\\texttt{dense/final\\_seed0.pt} at $(x=40R, r=0.5R, \\mathrm{Re}=40{,}000)$.'
assert old_763 in text, 'old_763 not found'
text = text.replace(old_763, new_763)

# 16. Table 9 (Stage 5 activations)
old_tab9 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2267, 0.3997, 0.2123, 0.3520]$ & $[-0.2229, 0.3797, 0.2091, 0.3382]$ & 4.24 & 2.41 \\\\
Hidden 2 & $[1.7690, -1.0227, 1.5966, -4.9825]$ & $[0.9435, -0.7710, 0.9212, -0.9999]$ & 13.35 & 1.71 \\\\
Hidden 3 & $[3.0137, -0.7519, 0.4437, -0.0928]$ & $[0.9952, -0.6363, 0.4167, -0.0925]$ & 12.81 & 1.23 \\\\
Hidden 4 & $[2.0068, -0.4525, -2.8995, -0.6122]$ & $[0.9645, -0.4240, -0.9940, -0.5457]$ & 10.87 & 1.51 \\\\
Output & $[0.1523, -0.1620, 1.3229, -1.4583]$ & N/A & 5.37 & 0.57 \\\\
\\bottomrule
\\end{tabular}'''

new_tab9 = '''\\begin{table}[H]
\\centering
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Network Layer} & \\textbf{Pre-activations} $z_{[1:4]}$ & \\textbf{Post-activations} $a_{[1:4]} = \\tanh(z)$ & $\\|W\\|_F$ & $\\|b\\|_2$ \\\\
\\midrule
Hidden 1 & $[-0.2267, 0.3997, 0.2123, 0.3520]$ & $[-0.2229, 0.3797, 0.2091, 0.3382]$ & 4.24 & 2.41 \\\\
Hidden 2 & $[1.7690, -1.0227, 1.5966, -4.9825]$ & $[0.9435, -0.7710, 0.9212, -0.9999]$ & 13.35 & 1.71 \\\\
Hidden 3 & $[3.0137, -0.7519, 0.4437, -0.0928]$ & $[0.9952, -0.6363, 0.4167, -0.0925]$ & 12.81 & 1.23 \\\\
Hidden 4 & $[2.0068, -0.4525, -2.8995, -0.6122]$ & $[0.9645, -0.4240, -0.9940, -0.5457]$ & 10.87 & 1.51 \\\\
Output & $[0.1523, -0.1620, 1.3229, -1.4583]$ & N/A & 5.37 & 0.57 \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab9 in text, 'old_tab9 not found'
text = text.replace(old_tab9, new_tab9)

# 17. Table 11 (Master summary)
old_tab11 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Stage \\& Configuration} & \\textbf{Regime} & \\textbf{Re Range} & \\textbf{Streamwise $u$ Err} & \\textbf{Pressure $p$ Err} & \\textbf{Viscosity $\\tilde{\\nu}$ Err} & \\textbf{Params} \\\\
\\midrule
1. 2D Laminar Channel & Laminar & $50 - 600$ & $3.88\\% - 5.69\\%$ & $4.95\\% - 8.35\\%$ & N/A & 20,003 \\\\
2. Axisymmetric Pipe & Laminar & $50 - 600$ & $2.20\\% - 4.16\\%$ & $1.36\\% - 2.76\\%$ & N/A & 20,003 \\\\
3. 3D Rectangular Duct & Laminar & $50 - 600$ & $8.81\\% - 12.69\\%$ & $56.8\\% - 106.4\\%$ & N/A & 20,164 \\\\
4. 2D Turbulent Channel & RANS+SA & $10^4 - 1.2\\times 10^5$ & $0.42\\% - 0.58\\%$ & $0.38\\% - 0.58\\%$ & $99.2\\% - 99.5\\%$ & 13,060 \\\\
5. Axisymmetric Turb.\\ Pipe & RANS+SA & $10^4 - 1.2\\times 10^5$ & $0.36\\% - 0.57\\%$ & $0.65\\% - 2.57\\%$ & $47.1\\% - 79.6\\%$ & 13,060 \\\\
\\bottomrule
\\end{tabular}'''

new_tab11 = '''\\begin{table}[H]
\\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Stage \\& Configuration} & \\textbf{Regime} & \\textbf{Re Range} & \\textbf{Streamwise $u$ Err} & \\textbf{Pressure $p$ Err} & \\textbf{Viscosity $\\tilde{\\nu}$ Err} & \\textbf{Params} \\\\
\\midrule
1. 2D Laminar Channel & Laminar & $50 - 600$ & $3.88\\% - 5.69\\%$ & $4.95\\% - 8.35\\%$ & N/A & 20,003 \\\\
2. Axisymmetric Pipe & Laminar & $50 - 600$ & $2.20\\% - 4.16\\%$ & $1.36\\% - 2.76\\%$ & N/A & 20,003 \\\\
3. 3D Rectangular Duct & Laminar & $50 - 600$ & $8.81\\% - 12.69\\%$ & $56.8\\% - 106.4\\%$ & N/A & 20,164 \\\\
4. 2D Turbulent Channel & RANS+SA & $10^4 - 1.2\\times 10^5$ & $0.42\\% - 0.58\\%$ & $0.38\\% - 0.58\\%$ & $99.2\\% - 99.5\\%$ & 13,060 \\\\
5. Axisymmetric Turb.\\ Pipe & RANS+SA & $10^4 - 1.2\\times 10^5$ & $0.36\\% - 0.57\\%$ & $0.65\\% - 2.57\\%$ & $47.1\\% - 79.6\\%$ & 13,060 \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab11 in text, 'old_tab11 not found'
text = text.replace(old_tab11, new_tab11)

# 18. Table 12 (Ablation study)
old_tab12 = '''\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Ablation Variant} & \\textbf{Data Grid} & \\textbf{Physics Loss} & \\textbf{Train Error} & \\textbf{Test Error} & \\textbf{Gen.\\ Ratio} & \\textbf{PDE Residual} \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{Stage 4: Turbulent Channel Flow}} \\\\
A. Pure Data Fit (No PDE) & Dense ($100\\%$) & $\\times$ & $0.49\\%$ & $0.49\\%$ & $1.00$ & $4.82$ \\\\
B. PINN Baseline (Data + PDE) & Dense ($100\\%$) & $\\checkmark$ & $0.49\\%$ & $0.49\\%$ & $1.00$ & $4.75$ \\\\
\\midrule
C. Pure Data Fit (No PDE) & Sparse ($1\\%$) & $\\times$ & $0.48\\%$ & $1.42\\%$ & $2.96$ & $\\mathbf{699.22}$ \\\\
D. PINN Baseline (Data + PDE) & Sparse ($1\\%$) & $\\checkmark$ & $0.51\\%$ & $0.98\\%$ & $1.92$ & $\\mathbf{4.75}$ \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{Stage 5: Turbulent Pipe Flow}} \\\\
E. Pure Data Fit (No PDE) & Sparse ($1\\%$) & $\\times$ & $0.46\\%$ & $1.85\\%$ & $4.02$ & $\\mathbf{4160.51}$ \\\\
F. PINN Baseline (Data + PDE) & Sparse ($1\\%$) & $\\checkmark$ & $0.49\\%$ & $0.82\\%$ & $1.67$ & $\\mathbf{11.34}$ \\\\
\\bottomrule
\\end{tabular}'''

new_tab12 = '''\\begin{table}[H]
\\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Ablation Variant} & \\textbf{Data Grid} & \\textbf{Physics Loss} & \\textbf{Train Error} & \\textbf{Test Error} & \\textbf{Gen.\\ Ratio} & \\textbf{PDE Residual} \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{Stage 4: Turbulent Channel Flow}} \\\\
A. Pure Data Fit (No PDE) & Dense ($100\\%$) & $\\times$ & $0.49\\%$ & $0.49\\%$ & $1.00$ & $4.82$ \\\\
B. PINN Baseline (Data + PDE) & Dense ($100\\%$) & $\\checkmark$ & $0.49\\%$ & $0.49\\%$ & $1.00$ & $4.75$ \\\\
\\midrule
C. Pure Data Fit (No PDE) & Sparse ($1\\%$) & $\\times$ & $0.48\\%$ & $1.42\\%$ & $2.96$ & $\\mathbf{699.22}$ \\\\
D. PINN Baseline (Data + PDE) & Sparse ($1\\%$) & $\\checkmark$ & $0.51\\%$ & $0.98\\%$ & $1.92$ & $\\mathbf{4.75}$ \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{Stage 5: Turbulent Pipe Flow}} \\\\
E. Pure Data Fit (No PDE) & Sparse ($1\\%$) & $\\times$ & $0.46\\%$ & $1.85\\%$ & $4.02$ & $\\mathbf{4160.51}$ \\\\
F. PINN Baseline (Data + PDE) & Sparse ($1\\%$) & $\\checkmark$ & $0.49\\%$ & $0.82\\%$ & $1.67$ & $\\mathbf{11.34}$ \\\\
\\bottomrule
\\end{tabular}%
}'''
assert old_tab12 in text, 'old_tab12 not found'
text = text.replace(old_tab12, new_tab12)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: All 18 Step 7 layout changes written into report.tex')

