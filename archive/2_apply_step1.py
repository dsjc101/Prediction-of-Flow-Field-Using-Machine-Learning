import os

ROOT = r".\Downloads\openfoam_pinn_pipeline"
report_path = os.path.join(ROOT, "report.tex")

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Line 77: remove white2006, shah1978
old_l77 = r"Predicting the hydrodynamic development of internal flows in ducts, channels, and pipelines is a cornerstone of fluid dynamics, thermal engineering, and piping network design~\cite{white2006,shah1978,hornbeck1973}."
new_l77 = r"Predicting the hydrodynamic development of internal flows in ducts, channels, and pipelines is a cornerstone of fluid dynamics, thermal engineering, and piping network design~\cite{hornbeck1973}."
assert old_l77 in text, "old_l77 not found"
text = text.replace(old_l77, new_l77)

# 2. Line 79: replace bodoia1961 by hornbeck1973, drop openfoam2019 citation
old_l79 = r"Traditional computational approaches rely on spatial discretization: finite-difference marching schemes for parabolic entrance equations~\cite{bodoia1961,hornbeck1973} or elliptic finite-volume packages such as OpenFOAM~\cite{openfoam2019}."
new_l79 = r"Traditional computational approaches rely on spatial discretization: finite-difference marching schemes for parabolic entrance equations~\cite{hornbeck1973} or elliptic finite-volume solvers such as OpenFOAM."
assert old_l79 in text, "old_l79 not found"
text = text.replace(old_l79, new_l79)

# 3. Line 81: remove schlichting2017
old_l81 = r"Boundary layer theories by Schlichting and Gersten~\cite{schlichting2017} provide theoretical foundations for analyzing these entrance shear layers."
new_l81 = r"Classical boundary layer theory provides theoretical foundations for analyzing these entrance shear layers."
assert old_l81 in text, "old_l81 not found"
text = text.replace(old_l81, new_l81)

# 4. Line 108: remove spalart1992
old_l108 = r"\item \textbf{Stage 4 (Turbulent Developing Channel):} 2D turbulent channel flow ($L/h = 160$, $\mathrm{Re} \in [10^4, 1.2\times 10^5]$) governed by the elliptic RANS equations with Spalart--Allmaras eddy viscosity transport~\cite{spalart1992,allmaras2012}."
new_l108 = r"\item \textbf{Stage 4 (Turbulent Developing Channel):} 2D turbulent channel flow ($L/h = 160$, $\mathrm{Re} \in [10^4, 1.2\times 10^5]$) governed by the elliptic RANS equations with Spalart--Allmaras eddy viscosity transport~\cite{allmaras2012}."
assert old_l108 in text, "old_l108 not found"
text = text.replace(old_l108, new_l108)

# 5. Line 109: replace zagarola1998,mckeon2004 with superpipe_data
old_l109 = r"\item \textbf{Stage 5 (Turbulent Developing Pipe):} Cylindrical RANS+SA pipe flow ($L/R = 160$, $\mathrm{Re} \in [10^4, 1.2\times 10^5]$), benchmarked against high-Reynolds-number experimental data from the Princeton Superpipe~\cite{zagarola1998,mckeon2004}."
new_l109 = r"\item \textbf{Stage 5 (Turbulent Developing Pipe):} Cylindrical RANS+SA pipe flow ($L/R = 160$, $\mathrm{Re} \in [10^4, 1.2\times 10^5]$), benchmarked against high-Reynolds-number experimental data from the Princeton Superpipe~\cite{superpipe_data}."
assert old_l109 in text, "old_l109 not found"
text = text.replace(old_l109, new_l109)

# 6. Line 131: remove spalart1992
old_l131 = r"\item \textbf{Spalart--Allmaras (SA) Model:} A robust one-equation turbulence model solving a transport equation for a modified kinematic eddy viscosity $\tilde{\nu}$, accounting for turbulent production near walls, viscous destruction, diffusion, and cross-diffusion~\cite{spalart1992,allmaras2012}."
new_l131 = r"\item \textbf{Spalart--Allmaras (SA) Model:} A robust one-equation turbulence model solving a transport equation for a modified kinematic eddy viscosity $\tilde{\nu}$, accounting for turbulent production near walls, viscous destruction, diffusion, and cross-diffusion~\cite{allmaras2012}."
assert old_l131 in text, "old_l131 not found"
text = text.replace(old_l131, new_l131)

# 7. Line 161: remove shah1978
old_l161 = r"The hydraulic diameter for infinite parallel plates is $D_h = 4A/P = 4(2h)/2 = 4h = 0.20\,\text{m}$~\cite{shah1978}."
new_l161 = r"The hydraulic diameter for infinite parallel plates is $D_h = 4A/P = 4(2h)/2 = 4h = 0.20\,\text{m}$."
assert old_l161 in text, "old_l161 not found"
text = text.replace(old_l161, new_l161)

# 8. Line 219: replace Bodoia and Osterle by Hornbeck
old_l219 = r"Reference truth data was generated using an independent Python solver implementing the classical finite-difference marching method of Bodoia and Osterle~\cite{bodoia1961} (\texttt{1/entrance\_solver.py})."
new_l219 = r"Reference truth data was generated using an independent Python solver implementing the classical implicit marching method of Hornbeck~\cite{hornbeck1973} (\texttt{1/entrance\_solver.py})."
assert old_l219 in text, "old_l219 not found"
text = text.replace(old_l219, new_l219)

# 9. Line 461: remove shah1978
old_l461 = r"Hydraulic diameter is $D_h = 4(H W) / [2(H+W)] = 4(0.02) / 0.60 = 0.1333\,\text{m}$~\cite{shah1978}."
new_l461 = r"Hydraulic diameter is $D_h = 4(H W) / [2(H+W)] = 4(0.02) / 0.60 = 0.1333\,\text{m}$."
assert old_l461 in text, "old_l461 not found"
text = text.replace(old_l461, new_l461)

# 10. Line 482: remove shah1978
old_l482 = r"Downstream, the flow converges to the analytical Fourier series solution of Shah and London~\cite{shah1978}, where the peak centerline velocity reaches $u_{\max}/U_{\mathrm{mean}} = 1.992$ for an $\mathrm{AR}=2.0$ duct."
new_l482 = r"Downstream, the flow converges to the analytical Fourier series solution, where the peak centerline velocity reaches $u_{\max}/U_{\mathrm{mean}} = 1.992$ for an $\mathrm{AR}=2.0$ duct."
assert old_l482 in text, "old_l482 not found"
text = text.replace(old_l482, new_l482)

# 11. Line 614: remove spalart1992
old_l614 = r"closed with the one-equation Spalart--Allmaras (SA) model~\cite{spalart1992,allmaras2012}:"
new_l614 = r"closed with the one-equation Spalart--Allmaras (SA) model~\cite{allmaras2012}:"
assert old_l614 in text, "old_l614 not found"
text = text.replace(old_l614, new_l614)

# 12. Line 634: SA-noft2 phrasing
old_l634 = r"using the SA-noft2 closure~\cite{allmaras2012}."
new_l634 = r"using the Spalart--Allmaras model without the $f_{t2}$ term ($c_{t3} = 0$) [Allmaras 2012, p.~3]~\cite{allmaras2012}."
assert old_l634 in text, "old_l634 not found"
text = text.replace(old_l634, new_l634)

# 13. Line 865: friction law phrasing
old_l865 = r"Agreement with the McKeon et al.~\cite{mckeon2004} Superpipe friction law ($1/\sqrt{f} = 1.930\log_{10}(\mathrm{Re}\sqrt{f}) - 0.537$) shows a $1.52\%$--$2.21\%$ relative error ($C_f = f/4$)."
new_l865 = r"Agreement with the Superpipe friction-law fit used here ($1/\sqrt{f} = 1.930\log_{10}(\mathrm{Re}\sqrt{f}) - 0.537$) shows a $1.52\%$--$2.21\%$ relative error ($C_f = f/4$)."
assert old_l865 in text, "old_l865 not found"
text = text.replace(old_l865, new_l865)

# 14. Line 869: replace zagarola1998,mckeon2004 with superpipe_data
old_l869 = r"from the Princeton Superpipe experiment~\cite{zagarola1998,mckeon2004}"
new_l869 = r"from the Princeton Superpipe experiment~\cite{superpipe_data}"
assert old_l869 in text, "old_l869 not found"
text = text.replace(old_l869, new_l869)

# 15. Bibliography replacement (keeping atkinson1969, chen1973, dean1978 until steps 2 and 3)
bib_start = r"\begin{thebibliography}{19}"
bib_end = r"\end{thebibliography}"
start_idx = text.find(bib_start)
end_idx = text.find(bib_end) + len(bib_end)
assert start_idx != -1 and end_idx != -1, "thebibliography not found"

new_bib = r"""\begin{thebibliography}{12}

\bibitem{allmaras2012}
S.~R. Allmaras, F.~T. Johnson, and P.~R. Spalart.
\newblock \href{https://iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf#page=2}{Modifications and clarifications for the implementation of the {Spalart-Allmaras} turbulence model}.
\newblock In {\em Seventh International Conference on Computational Fluid Dynamics (ICCFD7)}, pages 1--11, 2012. \href{https://iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf#page=2}{[\textsc{pdf} p.~2, 3]}.

\bibitem{atkinson1969}
B.~Atkinson, M.~P. Brocklebank, M.~H. Cardew, and J.~M. Smith.
\newblock \href{https://doi.org/10.1002/aic.690150414}{Low {Reynolds} number developing flows}.
\newblock {\em AIChE Journal}, 15(4):548--554, 1969.

\bibitem{chen1973}
R.-Y. Chen.
\newblock \href{https://doi.org/10.1115/1.3446948}{Flow in the entrance region at low {Reynolds} numbers}.
\newblock {\em ASME Journal of Fluids Engineering}, 95(1):153--158, 1973.

\bibitem{dean1978}
R.~B. Dean.
\newblock \href{https://doi.org/10.1115/1.3448633}{Reynolds number dependence of skin friction and other bulk flow variables in two-dimensional rectangular duct flow}.
\newblock {\em ASME Journal of Fluids Engineering}, 100(2):215--223, 1978.

\bibitem{glorot2010}
X.~Glorot and Y.~Bengio.
\newblock \href{https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf#page=4}{Understanding the difficulty of training deep feedforward neural networks}.
\newblock In {\em Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS)}, pages 249--256, 2010. \href{https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf#page=4}{[\textsc{pdf} p.~4]}.

\bibitem{hornbeck1973}
R.~W. Hornbeck.
\newblock \href{https://ntrs.nasa.gov/citations/19740008333}{Numerical marching techniques for fluid flows with heat transfer}.
\newblock Special Publication SP-297, NASA, 1973. \href{https://ntrs.nasa.gov/citations/19740008333}{[\textsc{nasa} \textsc{ntrs} SP-297]}.

\bibitem{moser1999}
R.~D. Moser, J.~Kim, and N.~N. Mansour.
\newblock \href{https://turbulence.oden.utexas.edu/data/MKM/}{Direct numerical simulation of turbulent channel flow up to $\mathrm{Re}_\tau = 590$}.
\newblock {\em DNS data set}, \url{https://turbulence.oden.utexas.edu/data/MKM/}, 1999. \href{https://turbulence.oden.utexas.edu/data/MKM/}{[\textsc{mkm} Database]}.

\bibitem{pirozzoli2021}
S.~Pirozzoli, M.~Romero, M.~Fatica, R.~Verzicco, and P.~Orlandi.
\newblock \href{https://arxiv.org/pdf/2103.13383.pdf#page=7}{One-point statistics for turbulent pipe flow up to $\mathrm{Re}_\tau = 6000$}.
\newblock {\em Journal of Fluid Mechanics}, 926:A28, 2021. \href{https://arxiv.org/pdf/2103.13383.pdf#page=7}{[\textsc{arxiv} \textsc{pdf} p.~7]}.

\bibitem{superpipe_data}
Princeton University Gas Dynamics Laboratory.
\newblock \href{https://smits.princeton.edu/superpipe-pitot-data}{Princeton Superpipe Pitot Data}.
\newblock Experimental data repository, \url{https://smits.princeton.edu/superpipe-pitot-data}, 2026. \href{https://smits.princeton.edu/superpipe-pitot-data}{[\textsc{data}]}.

\bibitem{raissi2019}
M.~Raissi, P.~Perdikaris, and G.~E. Karniadakis.
\newblock \href{https://arxiv.org/pdf/1711.10561.pdf#page=6}{Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations}.
\newblock {\em Journal of Computational Physics}, 378:686--707, 2019. \href{https://arxiv.org/pdf/1711.10561.pdf#page=6}{[\textsc{arxiv} \textsc{pdf} p.~6]}.

\bibitem{rao2020}
C.~Rao, H.~Sun, and Y.~Liu.
\newblock \href{https://arxiv.org/pdf/2002.10558.pdf#page=2}{Physics-informed deep learning for incompressible laminar flows}.
\newblock {\em Theoretical and Applied Mechanics Letters}, 10(3):207--212, 2020. \href{https://arxiv.org/pdf/2002.10558.pdf#page=2}{[\textsc{arxiv} \textsc{pdf} p.~2]}.

\bibitem{wang2021}
S.~Wang, Y.~Teng, and P.~Perdikaris.
\newblock \href{https://arxiv.org/pdf/2001.04536.pdf#page=10}{Understanding and mitigating gradient pathologies in physics-informed neural networks}.
\newblock {\em SIAM Journal on Scientific Computing}, 43(5):A3055--A3081, 2021. \href{https://arxiv.org/pdf/2001.04536.pdf#page=10}{[\textsc{arxiv} \textsc{pdf} p.~10]}.

\end{thebibliography}"""

text = text[:start_idx] + new_bib + text[end_idx:]

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Step 1 changes applied to report.tex successfully.")
