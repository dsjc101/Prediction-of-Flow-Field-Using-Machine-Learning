"""
build_master_latex_report.py
Generates the comprehensive, rigorous academic LaTeX report for the 5-Stage PINN pipeline:
- Single-column standard article (11pt, margin=2cm)
- Plain language for a fluid mechanics professor (all ML terms defined with physical analogies)
- Identical 8-subsection structure for Stages 1 through 5
- Worked 4-decimal numerical forward pass for one point in Subsection D for each stage
- Single comprehensive comparison table in Section 8
- Limitations and honest 3-5 sentence note on Stage 6 in Section 9
- Open-access references ending in #page=XX
- Trimmed PyTorch code listings in Appendix
"""

import os
import json

base_dir = r"."

# Load facts JSONs
facts = {}
for i in range(1, 6):
    fp = os.path.join(base_dir, f"stage_facts_{i}.json")
    with open(fp, "r") as f:
        facts[i] = json.load(f)

print("Loaded facts for all 5 stages.")
"""
build_master_latex_report.py
Generates the comprehensive, rigorous academic LaTeX report for the 5-Stage PINN pipeline:
- Single-column standard article (11pt, margin=2cm)
- Plain language for a fluid mechanics professor (all ML terms defined with physical analogies)
- Identical 8-subsection structure for Stages 1 through 5
- Worked 4-decimal numerical forward pass for one point in Subsection D for each stage
- Single comprehensive comparison table in Section 8
- Limitations and honest 3-5 sentence note on Stage 6 in Section 9
- Open-access references ending in #page=XX
- Trimmed PyTorch code listings in Appendix
"""

import os
import json

base_dir = r"."

# Load facts JSONs
facts = {}
for i in range(1, 6):
    fp = os.path.join(base_dir, f"stage_facts_{i}.json")
    with open(fp, "r") as f:
        facts[i] = json.load(f)

print("Loaded facts for all 5 stages.")

header = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs,tabularx,multirow,array}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{siunitx}
\usepackage{listings}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,calc}
\usepackage{float}
\usepackage{subcaption}
\usepackage{tcolorbox}
\usepackage{enumitem}

\definecolor{primary}{RGB}{18,48,96}
\definecolor{accent}{RGB}{180,40,40}
\definecolor{shade}{RGB}{245,247,250}
\definecolor{codebg}{RGB}{248,249,251}

\hypersetup{
    colorlinks=true,
    linkcolor=primary,
    citecolor=primary,
    urlcolor=primary,
    pdftitle={Physics-Informed Neural Network Pipeline for Internal Fluid Flows: From Laminar Channels to Turbulent Pipe Flow},
    pdfauthor={Computational Fluid Dynamics and Machine Learning Research Group}
}

\lstset{
    backgroundcolor=\color{codebg},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    captionpos=b,
    commentstyle=\color{gray},
    keywordstyle=\color{primary}\bfseries,
    stringstyle=\color{accent},
    frame=single,
    rulecolor=\color{gray!30},
    numbers=left,
    numberstyle=\tiny\color{gray},
    stepnumber=1,
    numbersep=8pt,
    tabsize=4,
    showstringspaces=false
}

\title{\LARGE\textbf{A Rigorous Multi-Stage Physics-Informed Neural Network (PINN) Pipeline for Developing Internal Fluid Flows: From 2D Laminar Channels to Axisymmetric Turbulent Pipe Flow}}
\author{\textbf{Computational Fluid Dynamics and Machine Learning Research Team}}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
\noindent
This study presents a unified, five-stage Physics-Informed Neural Network (PINN) framework for predicting fluid flow development across progressively complex internal geometries and flow regimes. In Stages~1 through~3, we investigate steady laminar developing flows in a two-dimensional parallel-plate channel, an axisymmetric circular pipe, and a fully three-dimensional rectangular duct ($H=0.10\,\text{m}, W=0.20\,\text{m}$). In Stages~4 and~5, we advance to high-Reynolds-number turbulent developing flows in channels and pipes, coupling the Reynolds-Averaged Navier--Stokes (RANS) equations with the one-equation Spalart--Allmaras (SA) turbulence model. Using deep multilayer perceptrons trained via hybrid data-loss and autograd-derived PDE residual minimization, the models predict primary axial velocities with remarkable precision, achieving relative $L_2$ errors of $0.36\%$ to $5.69\%$ across training, interpolation, and extrapolation Reynolds numbers ($50 \le \mathrm{Re} \le 120{,}000$). Wall skin friction is captured to within $3.8\%$ to $6.1\%$ against established empirical correlations (Dean, Prandtl, McKeon) and direct numerical simulation (DNS) datasets (Moser et al., Princeton Superpipe) without direct shear supervision. However, we document notable physical stiffness: secondary transverse velocities ($v$) and turbulence working variables ($\tilde{\nu}$) exhibit substantial relative errors ($16\%$ to $96\%$ and $>100\%$, respectively) due to severe magnitude disparities ($O(10^{-3})$) and logarithmic near-wall boundary layer gradients. Preliminary development was initiated for an exploratory sixth stage addressing a 3D turbulent rectangular duct, but this stage was not completed due to time constraints and carries no claimed results. All quantitative parameters, loss weights, and architectural configurations are documented from saved training checkpoints and validated against literature benchmarks.
\end{abstract}

\vspace{1em}
\noindent\textbf{Keywords:} Physics-Informed Neural Networks (PINNs); Internal Flow Development; Navier--Stokes Equations; Spalart--Allmaras Turbulence Model; Boundary Layer Transition; Automatic Differentiation.

\tableofcontents
\newpage
"""

print("Header ready.")
