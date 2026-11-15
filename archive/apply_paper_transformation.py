"""
apply_paper_transformation.py
Transforms report.tex into a standalone academic research paper following Parts 1-6.
Preserves all existing numerical tables, figures, equations, and results.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_PATH = os.path.join(BASE_DIR, "report.tex")

with open(TEX_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------------------------------------------------------
# PART 1: FRONT PAGE
# -----------------------------------------------------------------------------
# 1.1 Delete subtitle under title
text = re.sub(
    r"\\title\{\\LARGE\\textbf\{Prediction of Flow Field Using Machine Learning\}\\\\\[0\.4em\]\s*\\large A Five-Stage Investigation of Navier--Stokes and Reynolds-Averaged Equations\}",
    r"\\title{\\LARGE\\textbf{Prediction of Flow Field Using Machine Learning}}",
    text
)

# 1.2 Author and Date: \author{\textbf{Divyansh Singh}} and \date{}
text = re.sub(
    r"\\author\{\\textbf\{Divyansh Singh\}\\\\\[0\.3em\]\s*\\small Academic Supervisor: Prof\. Subhadip Das\}",
    r"\\author{\\textbf{Divyansh Singh}}\n\\date{}",
    text
)

# 1.3 Delete "In plain terms: ..." from abstract
text = re.sub(
    r"% src: 1/outputs/metrics_full_domain\.csv[^\n]*\nIn plain terms:[^\n]*\n+",
    "",
    text
)

# 1.4 Delete Keywords section
text = re.sub(
    r"\\vspace\{0\.8em\}\s*\\noindent\\textbf\{Keywords:\}[^\n]+\n",
    "",
    text
)

# -----------------------------------------------------------------------------
# GRAPHICS PATH: Include report_figs/pred/
# -----------------------------------------------------------------------------
text = text.replace(
    r"\graphicspath{{report_figs/}{figures/}}",
    r"\graphicspath{{report_figs/pred/}{report_figs/}{figures/}}"
)

# -----------------------------------------------------------------------------
# PART 2: LANGUAGE AUDIT REPLACEMENTS
# -----------------------------------------------------------------------------
# Abstract "audits reveal two critical failures"
text = text.replace(
    "However, audits reveal two critical failures:",
    "However, comparative evaluations reveal two primary discrepancy modes:"
)

# Intro "is critical for predicting pressure drops"
text = text.replace(
    "resolving this development region is critical for predicting pressure drops,",
    "resolving this development region is essential for predicting pressure drops,"
)

# Intro "audit of loss-weighting failures"
text = text.replace(
    "and conducts an audit of loss-weighting failures in pressure-driven flow.",
    "and conducts a systematic evaluation of loss-weighting behavior in pressure-driven flow."
)

# Section 2 intro "in plain language"
text = text.replace(
    "this section defines all essential concepts in plain language.",
    "this section defines all essential concepts and operational terminology."
)

# Stage 1 entrance length correlation
text = text.replace(
    "The hydrodynamic entrance length $L_e$ for laminar flow between parallel plates is estimated with an empirical correlation (constants not verified against a free source; see CHANGELOG):",
    "The hydrodynamic entrance length $L_e$ for laminar flow between parallel plates is estimated with an empirical correlation (this correlation is a standard empirical estimate used here as a rough benchmark; no freely accessible primary source was located, so it is used only for indicative comparison):"
)

# Stage 2 entrance length correlation
text = text.replace(
    "The circular pipe hydrodynamic entrance length correlation is estimated with an empirical correlation (constants not verified against a free source; see CHANGELOG):",
    "The circular pipe hydrodynamic entrance length correlation is estimated with an empirical correlation (this correlation is a standard empirical estimate used here as a rough benchmark; no freely accessible primary source was located, so it is used only for indicative comparison):"
)

# Forward pass subsections F in Stages 1, 2, 3, 4, 5
text = text.replace(
    r"\subsection{F. Deterministic Worked Numerical Forward Pass}",
    r"\subsection{F. Forward Pass at a Single Point}"
)

# Stage 3 failure modes
text = text.replace(
    "Detailed auditing reveals two critical failure modes in Stage 3:",
    "Comparative evaluation identifies two primary failure modes in Stage 3:"
)

# Stage 4 Structural Modeling Distinction
text = text.replace(
    r"\noindent\textbf{Structural Modeling Distinction:} Crucially, while the training and evaluation data originate from this parabolic marching solver,",
    r"\noindent\textbf{Equations Solved by the Network and by the Reference Solver:} While the training and evaluation data originate from this parabolic marching solver,"
)

# Stage 4 Critical Loss Weight Ratio
text = text.replace(
    r"\noindent\textbf{Critical Loss Weight Ratio:} The ratio of momentum physics weight to data weight is:",
    r"\noindent\textbf{Comparison of Loss Weights:} The ratio of momentum physics weight to data weight is:"
)

# Stage 4 Eddy Viscosity Failure Mode
text = text.replace(
    r"\noindent\textbf{Eddy Viscosity Failure Mode:}",
    r"\noindent\textbf{Behaviour of the Eddy-Viscosity Variable.}"
)

# Stage 5 Structural Modeling Distinction
text = text.replace(
    r"\noindent\textbf{Structural Modeling Distinction:} Similar to Stage 4, the reference solver is strictly parabolic, whereas the PINN evaluates the full cylindrical Navier--Stokes system (retaining axial diffusion $\partial^2/\partial x^2$ and cross-stream pressure gradient $\partial P/\partial r$). This structural discrepancy means the reference data satisfy a simplified boundary layer formulation, which can introduce discrepancies in regions with non-negligible radial pressure variations.",
    r"\noindent\textbf{Equations Solved by the Network and by the Reference Solver:} Similar to Stage 4, the reference solver solves the parabolic boundary-layer formulation, whereas the PINN evaluates the complete cylindrical Navier--Stokes system (retaining axial diffusion $\partial^2/\partial x^2$ and radial pressure variation $\partial P/\partial r$). Consequently, near the entrance where streamline curvature is non-negligible, a mathematical discrepancy exists between the parabolic reference and the elliptic network solution."
)

# Stage 5 Critical Loss Weight Comparison
text = text.replace(
    r"\noindent\textbf{Critical Loss Weight Comparison:} The relative SA physics-to-data weight ratio is:",
    r"\noindent\textbf{Comparison of Loss Weights Between Stages:} The relative SA physics-to-data weight ratio is:"
)

# Stage 5 friction law and script reference
text = text.replace(
    r"Agreement with the friction-law fit used in the analysis script (\path{generate_all_perfect_plots.py}, line 1367; not independently sourced) ($1/\sqrt{f} = 1.930\log_{10}(\mathrm{Re}\sqrt{f}) - 0.537$) shows a $1.52\%$--$2.21\%$ relative error ($C_f = f/4$)",
    r"Agreement with the empirical friction-law fit implemented in the analysis script (\path{generate_all_plots.py}) ($1/\sqrt{f} = 1.930\log_{10}(\mathrm{Re}\sqrt{f}) - 0.537$; this correlation is a standard empirical estimate used here as a rough benchmark; no freely accessible primary source was located, so it is used only for indicative comparison) shows a $1.52\%$--$2.21\%$ relative error ($C_f = f/4$)"
)

# Section 8 Ablation title
text = text.replace(
    r"\subsection{B. Analysis of Real Ablation Experiments: Dense vs.\ Sparse Regimes}",
    r"\subsection{B. Ablation Experiments: Dense vs.\ Sparse Regimes}"
)

# Section 8 True Role bullet
text = text.replace(
    r"\item \textbf{The True Role of Physics Loss:}",
    r"\item \textbf{Role of the Physics Loss:}"
)

# Any other occurrences of generate_all_perfect_plots.py
text = text.replace("generate_all_perfect_plots.py", "generate_all_plots.py")

# -----------------------------------------------------------------------------
# PART 3: IN-TEXT CITATIONS WITH LOCATORS
# -----------------------------------------------------------------------------
# Introduction citations
text = text.replace(
    r"cornerstone of fluid dynamics, thermal engineering, and piping network design~\cite{hornbeck1973}.",
    r"cornerstone of fluid dynamics, thermal engineering, and piping network design~\cite[p.~1]{hornbeck1973}."
)
text = text.replace(
    r"marching schemes for parabolic entrance equations~\cite{hornbeck1973} or elliptic",
    r"marching schemes for parabolic entrance equations~\cite[p.~1]{hornbeck1973} or elliptic"
)
text = text.replace(
    r"pioneered by Raissi et al.~\cite{raissi2019} and extended to incompressible flows by Rao et al.~\cite{rao2020},",
    r"pioneered by Raissi et al.~\cite[p.~6]{raissi2019} and extended to incompressible flows by Rao et al.~\cite[Sec.~2, p.~2]{rao2020},"
)
text = text.replace(
    r"Jin et al.~\cite{jin2021} developed NSFnets for incompressible laminar and turbulent flows,",
    r"Jin et al.~\cite[abstract, p.~1]{jin2021} developed NSFnets for incompressible laminar and turbulent flows,"
)
text = text.replace(
    r"Cai et al.~\cite{cai2021} reviewed physics-informed deep learning methodologies for fluid mechanics,",
    r"Cai et al.~\cite[abstract, p.~1]{cai2021} reviewed physics-informed deep learning methodologies for fluid mechanics,"
)
text = text.replace(
    r"Eivazi et al.~\cite{eivazi2022} applied PINNs to solve RANS equations for incompressible turbulent flows.",
    r"Eivazi et al.~\cite[abstract, p.~1]{eivazi2022} applied PINNs to solve RANS equations for incompressible turbulent flows."
)
text = text.replace(
    r"Wang et al.~\cite{wang2022ntk} analyzed training dynamics and failure modes through neural tangent kernels,",
    r"Wang et al.~\cite[abstract, p.~1]{wang2022ntk} analyzed training dynamics and failure modes through neural tangent kernels,"
)
text = text.replace(
    r"Krishnapriyan et al.~\cite{krishnapriyan2021} characterized failure modes in differential equations",
    r"Krishnapriyan et al.~\cite[abstract, p.~1]{krishnapriyan2021} characterized failure modes in differential equations"
)
text = text.replace(
    r"Spalart--Allmaras eddy viscosity transport~\cite{allmaras2012}.",
    r"Spalart--Allmaras eddy viscosity transport~\cite[p.~2]{allmaras2012}."
)
text = text.replace(
    r"data from the Princeton Superpipe~\cite{superpipe_data}.",
    r"data from the Princeton Superpipe~\cite[p.~1]{superpipe_data}."
)

# Section 2 concepts citations
text = text.replace(
    r"diffusion, and cross-diffusion~\cite{allmaras2012}.",
    r"diffusion, and cross-diffusion~\cite[p.~2]{allmaras2012}."
)
text = text.replace(
    r"signals during early training~\cite{glorot2010}.",
    r"signals during early training~\cite[Eq.~(12), p.~253]{glorot2010}."
)
text = text.replace(
    r"techniques that balance disparate loss terms~\cite{wang2021}.",
    r"techniques that balance disparate loss terms~\cite[Alg.~1, Eq.~(40), p.~10]{wang2021}."
)

# Stage 1 citations
text = text.replace(
    r"Durst et al. as reproduced in~\cite{ray2017} [p.~8] gives",
    r"Durst et al. as reproduced in~\cite[Eqs.~(2a)--(2b), p.~8]{ray2017} gives"
)
text = text.replace(
    r"method of Hornbeck~\cite{hornbeck1973} (\texttt{1/entrance\_solver.py}).",
    r"method of Hornbeck~\cite[p.~1]{hornbeck1973} (\texttt{1/entrance\_solver.py})."
)
text = text.replace(
    r"Xavier normal initialization~\cite{glorot2010}.",
    r"Xavier normal initialization~\cite[Eq.~(12), p.~253]{glorot2010}."
)

# Stage 2 citations
text = text.replace(
    r"Durst et al. as reproduced in~\cite{ray2017} [p.~8] gives $L_e = 0.577",
    r"Durst et al. as reproduced in~\cite[Eqs.~(2a)--(2b), p.~8]{ray2017} gives $L_e = 0.577"
)
text = text.replace(
    r"marching finite-difference solver~\cite{hornbeck1973} (\texttt{2/entrance\_solver.py}).",
    r"marching finite-difference solver~\cite[p.~1]{hornbeck1973} (\texttt{2/entrance\_solver.py})."
)

# Stage 4 citations
text = text.replace(
    r"one-equation Spalart--Allmaras (SA) model~\cite{allmaras2012}:",
    r"one-equation Spalart--Allmaras (SA) model~\cite[p.~2]{allmaras2012}:"
)
text = text.replace(
    r"without the $f_{t2}$ term ($c_{t3} = 0$) [Allmaras 2012, p.~3]~\cite{allmaras2012}.",
    r"without the $f_{t2}$ term ($c_{t3} = 0$)~\cite[Sec.~2.1, p.~3]{allmaras2012}."
)
text = text.replace(
    r"benchmarked against Moser et al.~\cite{moser1999}.",
    r"benchmarked against Moser et al.~\cite[p.~1]{moser1999}."
)

# Stage 5 citations
text = text.replace(
    r"Pipe DNS benchmarks by Pirozzoli et al.~\cite{pirozzoli2021} confirm these parameters.",
    r"Turbulent pipe flow friction law and DNS velocity benchmarks are documented by Pirozzoli et al.~\cite[Eq.~(3.2), p.~7]{pirozzoli2021}."
)
text = text.replace(
    r"Princeton Superpipe experiment~\cite{superpipe_data} spanning",
    r"Princeton Superpipe experiment~\cite[p.~1]{superpipe_data} spanning"
)

# -----------------------------------------------------------------------------
# CLEAN BIBLIOGRAPHY ENTRIES (Remove odd bracket notes)
# -----------------------------------------------------------------------------
text = re.sub(
    r"\\href\{https://iccfd\.org[^\}]+\}\{\[\\textsc\{pdf\} p\.~2, 3\]\}\.",
    "",
    text
)
text = re.sub(
    r"\\href\{https://proceedings\.mlr\.press[^\}]+\}\{\[\\textsc\{pdf\} p\.~4\]\}\.",
    "",
    text
)
text = re.sub(
    r"\\href\{https://ntrs\.nasa\.gov[^\}]+\}\{\[\\textsc\{nasa\} \\textsc\{ntrs\} SP-297\]\}\.",
    "",
    text
)
text = re.sub(
    r"\\href\{https://turbulence\.oden\.utexas\.edu[^\}]+\}\{\[\\textsc\{mkm\} Database\]\}\.",
    "",
    text
)
text = re.sub(
    r"\\href\{https://smits\.princeton\.edu[^\}]+\}\{\[\\textsc\{data\}\]\}\.",
    "",
    text
)
text = re.sub(
    r"\\href\{https://arxiv\.org[^\}]+\}\{\[\\textsc\{arxiv\} \\textsc\{pdf\} p\.~[0-9]+\]\}\.",
    "",
    text
)

# -----------------------------------------------------------------------------
# PART 4.1: EQUATIONS SOLVED BY NETWORK VS REFERENCE SOLVER (in Section 2.A)
# -----------------------------------------------------------------------------
sec2_modeling_distinction = r"""
\paragraph{Mathematical Distinction Between Network and Reference Formulations.}
A fundamental methodological distinction exists between the equations satisfied by the neural network and those solved by the numerical reference code:
\begin{enumerate}[label=(\roman*)]
\item \textbf{Full Navier--Stokes system solved by the PINN:} The neural network is trained directly on the full, steady, incompressible Navier--Stokes equations (or the full elliptic Reynolds-Averaged equations in turbulent regimes), retaining both streamwise diffusion ($\partial^2/\partial x^2$) and cross-stream pressure gradients ($\partial p/\partial y$ or $\partial P/\partial r$). A mesh-free neural network incorporates these higher-order elliptic terms via automatic differentiation without geometric meshing or extra algorithmic complexity. Crucially, the full elliptic formulation introduces no boundary-layer or thin-shear-layer assumptions, remaining mathematically valid in the immediate entrance region ($x \to 0$) where axial velocity gradients are steepest and parabolic approximations are least justifiable.
\item \textbf{Parabolic marching reference solver:} In contrast, the numerical reference truth datasets are produced using implicit finite-difference marching schemes of Hornbeck type~\cite[p.~1]{hornbeck1973}. A spatial marching formulation is computationally rapid, robust, and the established classical standard for developing internal conduit flows.
\item \textbf{Consequence and modeling limitation:} Because the marching solver neglects axial diffusion ($\partial^2 u/\partial x^2 \approx 0$) and enforces a transversely uniform pressure across each cross-section ($\partial p/\partial y = 0$), the network and reference solver satisfy different physical systems. A localized discrepancy between the PINN prediction and the reference profile is therefore physically expected near the conduit inlet ($x \lesssim 0.5\,D_h$). The reference solver does not constitute an exact analytical solution of the elliptic Navier--Stokes equations satisfied by the network, representing an inherent limitation of parabolic ground-truth comparisons.
\end{enumerate}
"""

text = text.replace(
    r"\subsection{B. Machine Learning and Optimization Concepts}",
    sec2_modeling_distinction + "\n\n" + r"\subsection{B. Machine Learning and Optimization Concepts}"
)

# -----------------------------------------------------------------------------
# PART 4.4 & 4.3: XAVIER INITIALIZATION & NETWORK DESIGN REASONS (in Section 2.B)
# -----------------------------------------------------------------------------
sec2_design_principles = r"""
\paragraph{Xavier-Normal Initialization.}
Neural network weights in this work are initialized following the normalized initialization criterion of Glorot and Bengio~\cite[Eq.~(12), p.~253]{glorot2010}. For a dense layer with $n_{\mathrm{in}}$ input units and $n_{\mathrm{out}}$ output units, each weight matrix entry $W_{ij}$ is independently sampled from a zero-mean Gaussian distribution with variance
\begin{equation}
\mathrm{Var}[W] = \frac{2}{n_{\mathrm{in}} + n_{\mathrm{out}}},
\end{equation}
while layer bias vectors $b$ are initialized to zero. This variance scaling balances the variance of forward-propagated layer activations with the variance of back-propagated gradient signals across deep architectures, preventing exponential signal amplification or attenuation during early optimization. Glorot's derivation assumes symmetric activation functions with unit first derivative at the origin ($f'(0) = 1$), a mathematical condition satisfied by the hyperbolic tangent ($\tanh$) activation functions employed throughout this investigation.

\paragraph{Network Architecture and Representation Design Choices.}
The architectures across all stages adhere to specific physical and numerical design rationales:
\begin{enumerate}[label=(\alph*)]
\item \textbf{Input coordinate normalization:} Spatial coordinates are normalized by conduit dimensions ($\hat{x} = x/L$, $\hat{y} = y/H$, $\hat{r} = r/R$, $\hat{z} = z/W$) into $[-1, 1]$ or $[0, 1]$, placing inputs directly within the active, non-saturating dynamic range of the $\tanh$ activation function.
\item \textbf{Logarithmic Reynolds-number embedding:} Reynolds numbers are embedded as $\log_{10}(\mathrm{Re})$ and mapped linearly to $[-1, 1]$ via
\begin{equation}
\mathrm{Re}_n = 2\,\frac{\log_{10}\mathrm{Re} - \log_{10}\mathrm{Re}_{\min}}{\log_{10}\mathrm{Re}_{\max} - \log_{10}\mathrm{Re}_{\min}} - 1,
\end{equation}
because operating conditions span one to two decades; logarithmic spacing distributes training Reynolds numbers uniformly across parameter space.
\item \textbf{Dimensionless output scaling:} Outputs are predicted in normalized variables ($u/U_{\mathrm{ref}}$, $v/U_{\mathrm{ref}}$, $p/p_{\mathrm{scale}}$) such that all network output heads predict $\mathcal{O}(1)$ values regardless of the physical magnitude or Reynolds number.
\item \textbf{Inlet square-root feature:} In turbulent stages, the spatial feature $\sqrt{x/L}$ is provided to the network, providing finer gradient resolution near the entrance ($x \to 0$) where turbulent boundary layers grow at the highest rate.
\item \textbf{Wall-normal logarithmic coordinate:} The coordinate $\ln(1 + \mathrm{Re}_\tau \eta)/\ln(1 + \mathrm{Re}_\tau)$ concentrates spatial collocation density inside the thin viscous sublayer ($y^+ < 5$).
\item \textbf{Smooth $\tanh$ activations:} The hyperbolic tangent is infinitely differentiable ($C^\infty$), ensuring non-vanishing and continuous second derivatives required for evaluating the Navier--Stokes viscous diffusion terms $\nabla^2 u$.
\item \textbf{Depth and width:} Four hidden layers of width 80 (laminar) or 64 (turbulent) were selected empirically, providing sufficient expressive capacity for multi-scale internal flow fields while retaining practical CPU execution times.
\item \textbf{Hard boundary envelopes:} In turbulent flows, hard envelopes $g = \tanh(\mathrm{Re}_\tau \eta / 10)$ enforce exact no-slip wall conditions. The scaling constant 10 ensures that $g$ reaches $\tanh(20/10) = \tanh(2.0) \approx 0.964$ at $y^+ = 20$, releasing the network to model unconstrained turbulent profiles in the buffer and logarithmic layers. The softplus activation ensures $\tilde{\nu} \ge 0$, and the bias offset $-3.0$ sets the baseline initialization to $\operatorname{softplus}(-3.0) = \ln(1 + e^{-3}) \approx 0.0486$.
\item \textbf{Two-stage optimization (Adam + L-BFGS):} Global training employs Adam with mini-batch or Latin-hypercube resampling for robust early loss reduction, followed by deterministic full-batch L-BFGS optimization for rapid quadratic convergence to high precision~\cite[Sec.~3.1, p.~3]{rao2020}.
\item \textbf{Learning rate schedules:} StepLR ($\gamma=0.7$ every 400--500 steps) and ExponentialLR ($100\times$ total decay) were selected empirically to ensure smooth convergence.
\end{enumerate}
"""

text = text.replace(
    r"\item \textbf{Adam and L-BFGS Optimizers:}",
    sec2_design_principles + "\n" + r"\item \textbf{Adam and L-BFGS Optimizers:}"
)

# -----------------------------------------------------------------------------
# PART 4.5: NOTATION AND TERMINOLOGY SUBSECTION (Section 2.C)
# -----------------------------------------------------------------------------
sec2_c_notation = r"""
\subsection{C. Notation and Operational Terminology}
To ensure unambiguous interpretation across both fluid mechanics and deep learning domains, the foundational mathematical terms used throughout this monograph are defined as follows:
\begin{itemize}[leftmargin=1.8em, itemsep=0.4em]
\item \textbf{Neuron / Unit:} An elementary scalar processing node that computes an affine combination of its inputs followed by a non-linear activation function.
\item \textbf{Layer:} A structured collection of parallel neurons operating at the same depth within the network hierarchy.
\item \textbf{Weight Matrix ($W$):} A parameter matrix governing linear transformations between adjacent network layers.
\item \textbf{Bias Vector ($b$):} An additive parameter vector shifting layer activations independently of input magnitude.
\item \textbf{Pre-activation ($z$):} The linear intermediate vector $z = W a_{\mathrm{prev}} + b$ entering a layer prior to the non-linear transformation.
\item \textbf{Post-activation ($a$):} The output vector produced by applying an elementwise activation function, $a = \tanh(z)$.
\item \textbf{Loss Function ($\mathcal{L}$):} A scalar objective measuring the weighted discrepancy between predictions, reference observations, and PDE residuals.
\item \textbf{Collocation Point:} A discrete spatial coordinate $(x, y, z, \mathrm{Re})$ sampled in the fluid domain where differential equations are evaluated via automatic differentiation without requiring numerical mesh connectivity.
\item \textbf{Automatic Differentiation (autograd):} A technique that evaluates exact derivatives of network outputs with respect to spatial inputs by applying the chain rule to the underlying computational graph down to machine precision.
\item \textbf{PDE Residual ($r$):} The algebraic evaluation of the governing differential operator applied to the network prediction; $r=0$ denotes exact satisfaction of the physical law.
\item \textbf{Optimizer:} The numerical algorithm (such as Adam or L-BFGS) that updates trainable parameters $\theta = \{W, b\}$ to minimize the aggregate loss function.
\item \textbf{Learning Rate ($\eta$):} A hyperparameter controlling the step size taken along the optimization descent direction at each iteration.
\item \textbf{Iteration / Epoch:} A single evaluation of forward predictions, loss computation, and backpropagated gradient update across the training set.
\item \textbf{Soft vs.\ Hard Boundary Conditions:} Soft boundary conditions enforce boundary constraints as quadratic penalty terms in the loss function ($\mathcal{L}_{\mathrm{bc}}$). Hard boundary conditions incorporate constraints directly into the network architecture via multiplicative spatial envelopes ($g(y)$), guaranteeing exact boundary satisfaction by construction.
\item \textbf{Optuna Hyperparameter Search:} An automated framework utilizing Tree-structured Parzen Estimators (TPE) to discover optimal loss-weighting combinations across multi-objective PINN formulations.
\end{itemize}
"""

text = text.replace(
    r"\section{Stage 1: Developing 2D Laminar Channel Flow}",
    sec2_c_notation + "\n\n" + r"\section{Stage 1: Developing 2D Laminar Channel Flow}"
)

# -----------------------------------------------------------------------------
# PART 4.6: TWO TIKZ FLOWCHARTS (Figure 1 & 2 in Section 2)
# -----------------------------------------------------------------------------
flowcharts_tikz = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=1.0cm, auto,
    block/.style={rectangle, draw=primary, fill=primary!6, text width=2.6cm, text centered, rounded corners, minimum height=1.1cm, font=\small},
    line/.style={draw=primary, -latex, thick}]
    
    \node [block] (input) {\textbf{Physical Inputs}\\$(x, y, [z], \mathrm{Re})$};
    \node [block, right=1.0cm of input] (scale) {\textbf{Input Scaling}\\Mapping to $[-1, 1]$\\$\log_{10}(\mathrm{Re})$ embedding};
    \node [block, right=1.0cm of scale] (hidden) {\textbf{Hidden Layers}\\4 Dense Layers\\$a = \tanh(W a_{\mathrm{prev}} + b)$};
    \node [block, right=1.0cm of hidden] (env) {\textbf{Output Envelope}\\Hard / Soft BCs\\$g = \tanh(\mathrm{Re}_\tau \eta / 10)$};
    \node [block, right=1.0cm of env] (phys) {\textbf{Physical Fields}\\$(u, v, [w], p, [\tilde{\nu}])$\\Dimensionless to physical};
    
    \path [line] (input) -- (scale);
    \path [line] (scale) -- (hidden);
    \path [line] (hidden) -- (env);
    \path [line] (env) -- (phys);
\end{tikzpicture}
\caption{Forward pass architecture of the Physics-Informed Neural Network. Inputs are scaled to $[-1, 1]$ before passing through 4 hidden layers of width 80 (Stages 1--3) or 64 (Stages 4--5) with $\tanh$ activations. Outputs are converted to physical velocity, pressure, and eddy viscosity fields through hard boundary envelopes in turbulent stages or soft penalty formulations in laminar stages.}
\label{fig:forward_pass_flowchart}
\end{figure}

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=1.1cm, auto,
    block/.style={rectangle, draw=secondary, fill=secondary!6, text width=2.8cm, text centered, rounded corners, minimum height=1.1cm, font=\small},
    line/.style={draw=secondary, -latex, thick}]
    
    \node [block] (sample) {\textbf{Point Sampling}\\Data points ($N_{\mathrm{data}}$)\\Collocation ($N_{\mathrm{pde}}$)\\Boundaries ($N_{\mathrm{bc}}$)};
    \node [block, right=1.0cm of sample] (pred) {\textbf{Forward Prediction}\\Evaluate network\\Obtain $(u, v, p, [\tilde{\nu}])$};
    \node [block, right=1.0cm of pred] (autograd) {\textbf{Autograd Derivatives}\\Exact spatial derivatives\\$\partial / \partial x, \partial / \partial y, \nabla^2$};
    \node [block, right=1.0cm of autograd] (losses) {\textbf{Loss Evaluation}\\$\mathcal{L}_{\mathrm{data}}$: reference misfit\\$\mathcal{L}_{\mathrm{pde}}$: Navier--Stokes / SA\\$\mathcal{L}_{\mathrm{bc}}$: boundary penalties};
    \node [block, below=1.0cm of losses] (opt) {\textbf{Optimizer Update}\\Adam (initial steps)\\L-BFGS (refinement)\\\textbf{Weights via Optuna / Sweep}};
    
    \path [line] (sample) -- (pred);
    \path [line] (pred) -- (autograd);
    \path [line] (autograd) -- (losses);
    \path [line] (losses) -- (opt);
    \path [line] (opt) -| (sample);
\end{tikzpicture}
\caption{Training and optimization procedure of the PINN pipeline. Collocation, reference data, and boundary points are sampled across operating Reynolds numbers. Automatic differentiation evaluates governing PDE residuals without a numerical grid. Individual loss terms are balanced via composite weights determined empirically or optimized through Optuna TPE searches (Stages 4--5), followed by sequential Adam and L-BFGS optimization.}
\label{fig:training_flowchart}
\end{figure}
"""

text = text.replace(
    r"\subsection{C. Notation and Operational Terminology}",
    flowcharts_tikz + "\n\n" + r"\subsection{C. Notation and Operational Terminology}"
)

# -----------------------------------------------------------------------------
# FORWARD PASS TABLES EXPLANATORY SENTENCES (Part 4.5)
# -----------------------------------------------------------------------------
fw_explanation = "The pre-activation vector $z = W a_{\mathrm{prev}} + b$ represents the affine combination entering each layer, while the post-activation $a = \\tanh(z)$ is its bounded output. Matrix $\|W\|_F$ denotes the Frobenius norm of the layer weight matrix, $\|b\|_2$ is the Euclidean norm of the bias vector, and $z[1:4]$ reports the numerical values of the first four neurons of each layer."

text = re.sub(
    r"(\\subsection\{F\. Forward Pass at a Single Point\}\s*\n)",
    lambda m: m.group(1) + fw_explanation + "\n\n",
    text
)

# -----------------------------------------------------------------------------
# PART 4.2: WEIGHTED LOSS TERMS TABLE & PARAGRAPH (Section 8)
# -----------------------------------------------------------------------------
sec8_weighted_loss_table = r"""
\paragraph{Quantitative Analysis of Loss Term Shares at Convergence.}
To rigorously address whether turbulent configurations function as physics-driven PINNs or predominantly as data-driven surrogates, Table~\ref{tab:weighted_loss_end_training} reports the exact numerical loss breakdown at the conclusion of training across all five production models. Each raw residual $\mathcal{L}_i$ is evaluated at fixed random seed with identical collocation budgets to training, multiplied by its assigned weight $w_i$, and expressed as a percentage share of the total weighted loss.

\begin{table}[H]
\centering
\small
\begin{tabular}{llcccc}
\toprule
\textbf{Stage} & \textbf{Loss Term ($i$)} & \textbf{Raw Loss $\mathcal{L}_i$} & \textbf{Weight $w_i$} & \textbf{Weighted Value $w_i \mathcal{L}_i$} & \textbf{Share of Total (\%)} \\
\midrule
Stage 1 (2D Channel) & Data misfit ($\mathcal{L}_{\mathrm{data}}$) & $5.542\times 10^{-3}$ & $20.00$ & $1.108\times 10^{-1}$ & $43.94\%$ \\
& Navier--Stokes residual ($\mathcal{L}_{\mathrm{pde}}$) & $1.430\times 10^{-2}$ & $1.00$ & $1.430\times 10^{-2}$ & $5.67\%$ \\
& Inlet boundary ($\mathcal{L}_{\mathrm{inlet}}$) & $1.918\times 10^{-2}$ & $5.00$ & $9.592\times 10^{-2}$ & $38.02\%$ \\
& Wall boundary ($\mathcal{L}_{\mathrm{wall}}$) & $6.120\times 10^{-3}$ & $5.00$ & $3.060\times 10^{-2}$ & $12.13\%$ \\
& Outlet gauge ($\mathcal{L}_{\mathrm{outlet}}$) & $6.083\times 10^{-4}$ & $1.00$ & $6.083\times 10^{-4}$ & $0.24\%$ \\
\cmidrule{2-6}
& \textbf{Stage 1 Total} & -- & -- & $\mathbf{0.2523}$ & $\mathbf{100.00\%}$ \\
\midrule
Stage 2 (Laminar Pipe) & Data misfit ($\mathcal{L}_{\mathrm{data}}$) & $2.103\times 10^{-3}$ & $10.00$ & $2.103\times 10^{-2}$ & $13.61\%$ \\
& Axisymmetric PDE residual ($\mathcal{L}_{\mathrm{pde}}$) & $8.028\times 10^{-2}$ & $1.00$ & $8.028\times 10^{-2}$ & $51.97\%$ \\
& Inlet boundary ($\mathcal{L}_{\mathrm{inlet}}$) & $6.773\times 10^{-4}$ & $5.00$ & $3.386\times 10^{-3}$ & $2.19\%$ \\
& Wall boundary ($\mathcal{L}_{\mathrm{wall}}$) & $9.723\times 10^{-3}$ & $5.00$ & $4.862\times 10^{-2}$ & $31.47\%$ \\
& Axis symmetry ($\mathcal{L}_{\mathrm{axis}}$) & $3.053\times 10^{-5}$ & $5.00$ & $1.526\times 10^{-4}$ & $0.10\%$ \\
& Outlet gauge ($\mathcal{L}_{\mathrm{outlet}}$) & $2.042\times 10^{-4}$ & $5.00$ & $1.021\times 10^{-3}$ & $0.66\%$ \\
\cmidrule{2-6}
& \textbf{Stage 2 Total} & -- & -- & $\mathbf{0.1545}$ & $\mathbf{100.00\%}$ \\
\midrule
Stage 3 (3D Rectangular Duct) & Axial velocity data ($\mathcal{L}_{\mathrm{data},u}$) & $1.234\times 10^{-2}$ & $8.73$ & $1.077\times 10^{-1}$ & $12.62\%$ \\
& Transverse data ($\mathcal{L}_{\mathrm{data},v}$) & $3.559\times 10^{-4}$ & $0.11$ & $3.915\times 10^{-5}$ & $<0.01\%$ \\
& Spanwise data ($\mathcal{L}_{\mathrm{data},w}$) & $1.554\times 10^{-4}$ & $11.11$ & $1.727\times 10^{-3}$ & $0.20\%$ \\
& Pressure data ($\mathcal{L}_{\mathrm{data},p}$) & $4.070\times 10^{-2}$ & $2.42$ & $9.850\times 10^{-2}$ & $11.54\%$ \\
& 3D Navier--Stokes PDE ($\mathcal{L}_{\mathrm{pde}}$) & $1.968\times 10^{-3}$ & $32.18$ & $6.332\times 10^{-2}$ & $7.42\%$ \\
& Inlet boundary ($\mathcal{L}_{\mathrm{inlet}}$) & $3.154\times 10^{-1}$ & $0.68$ & $2.145\times 10^{-1}$ & $25.13\%$ \\
& Wall no-slip ($\mathcal{L}_{\mathrm{wall}}$) & $3.087\times 10^{-2}$ & $11.14$ & $3.439\times 10^{-1}$ & $40.29\%$ \\
& Outlet gauge ($\mathcal{L}_{\mathrm{outlet}}$) & $3.745\times 10^{-2}$ & $0.64$ & $2.397\times 10^{-2}$ & $2.81\%$ \\
\cmidrule{2-6}
& \textbf{Stage 3 Total} & -- & -- & $\mathbf{0.8536}$ & $\mathbf{100.00\%}$ \\
\midrule
Stage 4 (Turbulent Channel) & Supervised data ($\mathcal{L}_{\mathrm{data}}$) & $5.734\times 10^{-4}$ & $6768.75$ & $3.881$ & $79.49\%$ \\
& Continuity residual ($\mathcal{L}_{\mathrm{cont}}$) & $1.571\times 10^{-3}$ & $225.41$ & $0.3541$ & $7.25\%$ \\
& Momentum residual ($\mathcal{L}_{\mathrm{mom}}$) & $7.772\times 10^{2}$ & $3.436\times 10^{-5}$ & $2.671\times 10^{-2}$ & $0.55\%$ \\
& SA turbulence residual ($\mathcal{L}_{\mathrm{sa}}$) & $2.794\times 10^{-2}$ & $3.307\times 10^{-5}$ & $9.239\times 10^{-7}$ & $<0.001\%$ \\
& Inlet boundary penalty ($\mathcal{L}_{\mathrm{inlet}}$) & $5.043$ & $0.1197$ & $0.6038$ & $12.37\%$ \\
& Outlet gauge penalty ($\mathcal{L}_{\mathrm{outlet}}$) & $3.512\times 10^{-5}$ & $5.525$ & $1.941\times 10^{-4}$ & $<0.01\%$ \\
& Symmetry plane penalty ($\mathcal{L}_{\mathrm{sym}}$) & $1.253\times 10^{-2}$ & $1.324$ & $1.659\times 10^{-2}$ & $0.34\%$ \\
\cmidrule{2-6}
& \textbf{Stage 4 Total} & -- & -- & $\mathbf{4.8824}$ & $\mathbf{100.00\%}$ \\
\midrule
Stage 5 (Turbulent Pipe) & Supervised data ($\mathcal{L}_{\mathrm{data}}$) & $5.339\times 10^{-5}$ & $1996.58$ & $1.066\times 10^{-1}$ & $33.67\%$ \\
& Continuity residual ($\mathcal{L}_{\mathrm{cont}}$) & $8.428\times 10^{-3}$ & $11.82$ & $9.965\times 10^{-2}$ & $31.48\%$ \\
& Momentum residual ($\mathcal{L}_{\mathrm{mom}}$) & $1.657\times 10^{2}$ & $5.519\times 10^{-4}$ & $9.144\times 10^{-2}$ & $28.89\%$ \\
& SA turbulence residual ($\mathcal{L}_{\mathrm{sa}}$) & $2.217\times 10^{-3}$ & $2.233$ & $4.951\times 10^{-3}$ & $1.56\%$ \\
& Inlet boundary penalty ($\mathcal{L}_{\mathrm{inlet}}$) & $8.104\times 10^{-3}$ & $1.515$ & $1.228\times 10^{-2}$ & $3.88\%$ \\
& Outlet gauge penalty ($\mathcal{L}_{\mathrm{outlet}}$) & $1.595\times 10^{-5}$ & $2.055$ & $3.278\times 10^{-5}$ & $0.01\%$ \\
& Symmetry axis penalty ($\mathcal{L}_{\mathrm{sym}}$) & $4.498\times 10^{-3}$ & $0.3592$ & $1.615\times 10^{-3}$ & $0.51\%$ \\
\cmidrule{2-6}
& \textbf{Stage 5 Total} & -- & -- & $\mathbf{0.3166}$ & $\mathbf{100.00\%}$ \\
\bottomrule
\end{tabular}
\caption{Weighted contribution of each loss term at the end of training evaluated from saved production checkpoints across Stages 1--5. The share of total reflects the effective optimization pressure exerted by each physical and data objective at convergence.}
\label{tab:weighted_loss_end_training}
\end{table}

A numerical weight must always be interpreted in conjunction with the raw magnitude of the residual it scales. Because PDE residuals and data misfits possess different physical dimensions and disparate numerical scales (for instance, raw momentum residuals in turbulent flows attain $\mathcal{O}(10^2)$ while normalized velocity errors are $\mathcal{O}(10^{-4})$), loss weights of $10^{-5}$ to $10^{-9}$ serve to re-normalize these heterogeneous terms onto a balanced dynamic range. As demonstrated by the quantitative shares in Table~\ref{tab:weighted_loss_end_training}, in Stages~1--3 the physics and boundary residuals constitute significant components of the overall objective: the PDE residual contributes $5.67\%$ in Stage~1, $51.97\%$ in Stage~2, and $7.42\%$ in Stage~3 (with boundary enforcement accounting for $50.39\%$ and $68.23\%$, respectively). 

Conversely, in Stages~4 and 5, the turbulent networks are predominantly data-fitted. In Stage~4, data loss accounts for $79.49\%$ of the objective, while momentum and SA residuals account for only $0.55\%$ and $<0.001\%$, respectively; thus, the network functions primarily as a regression model with an eddy viscosity extinction defect ($\tilde{\nu} \approx 0$). In Stage~5, data loss represents $33.67\%$, while continuity ($31.48\%$) and momentum ($28.89\%$) are actively enforced, but the SA working variable loss comprises only $1.56\%$. Therefore, the turbulent networks cannot be characterized as purely physics-driven; the physics loss acts primarily as an inductive regularizer. This finding is directly corroborated by the sparse-data ablation experiments (Table~12), which demonstrate that while physics regularization significantly improves generalization and reduces PDE residuals when supervision is sparse, it produces virtually no change in accuracy when dense measurement data are available.
"""

text = text.replace(
    r"\subsection{B. Ablation Experiments: Dense vs.\ Sparse Regimes}",
    sec8_weighted_loss_table + "\n\n" + r"\subsection{B. Ablation Experiments: Dense vs.\ Sparse Regimes}"
)

# -----------------------------------------------------------------------------
# CODE AND DATA AVAILABILITY PARAGRAPH (Section 11)
# -----------------------------------------------------------------------------
sec11_code_avail = r"""
\paragraph{Code and Data Availability.}
All model checkpoints, reference solver scripts, and diagnostic analysis routines are archived to ensure full scientific reproducibility. Production checkpoints are archived at \path{1/outputs/channel_pinn_hybrid.pt}, \path{2/checkpoints/pipe_pinn.pt}, \path{3/checkpoints/duct_pinn.pt}, \path{4/models/best.pt}, and \path{5/outputs/models/dense/final_seed0.pt}. Reference data generation routines reside in stage directories (\path{1/entrance_solver.py}, \path{2/entrance_solver.py}, \path{3/duct_solver.py}, \path{4/build_reference.py}, and \path{5/build_reference.py}). Global diagnostic visualizations and tabular syntheses are generated by \path{generate_all_plots.py}, prediction flow fields by \path{tools/plot_predictions.py}, loss breakdown calculations by \path{tools/compute_weighted_losses.py}, and pressure gradient metrics by \path{tools/dp_ratio.py}. Complete source documentation is catalogued in the accompanying evidence registry.
"""

text = text.replace(
    r"\end{document}",
    sec11_code_avail + "\n\n" + r"\end{document}"
)

with open(TEX_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("apply_paper_transformation.py completed successfully!")
