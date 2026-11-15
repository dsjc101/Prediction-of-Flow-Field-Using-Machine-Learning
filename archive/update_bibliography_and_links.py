import os, re

ROOT = r"."
report_path = os.path.join(ROOT, "report.tex")

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# Update OpenFOAM entry
old_of = r"""\bibitem{openfoam2019}
OpenCFD Ltd.
\newblock \href{https://www.openfoam.com}{{\em OpenFOAM v1912 User Guide}}, 2019. \href{https://www.openfoam.com}{[\textsc{url}]};."""
if old_of in text:
    text = text.replace(old_of, r"""\bibitem{openfoam2019}
OpenCFD Ltd.
\newblock \href{https://www.openfoam.com/documentation/user-guide}{{\em OpenFOAM v1912 User Guide}}, 2019. \href{https://www.openfoam.com/documentation/user-guide}{[\textsc{openfoam} User Guide]}.""")
else:
    # try without semicolon
    old_of2 = r"""\bibitem{openfoam2019}
OpenCFD Ltd.
\newblock \href{https://www.openfoam.com}{{\em OpenFOAM v1912 User Guide}}, 2019. \href{https://www.openfoam.com}{[\textsc{url}]}."""
    assert old_of2 in text, "old_of2 not found"
    text = text.replace(old_of2, r"""\bibitem{openfoam2019}
OpenCFD Ltd.
\newblock \href{https://www.openfoam.com/documentation/user-guide}{{\em OpenFOAM v1912 User Guide}}, 2019. \href{https://www.openfoam.com/documentation/user-guide}{[\textsc{openfoam} User Guide]}.""")

# Update Moser entry
old_moser = r"""\bibitem{moser1999}
R.~D. Moser, J.~Kim, and N.~N. Mansour.
\newblock \href{https://turbulence.ices.utexas.edu}{Direct numerical simulation of turbulent channel flow up to $\mathrm{Re}_\tau = 590$}.
\newblock {\em Physics of Fluids}, 11(4):943--945, 1999. \href{https://turbulence.ices.utexas.edu}{[\textsc{dns} Database]}."""
assert old_moser in text, "old_moser not found"
text = text.replace(old_moser, r"""\bibitem{moser1999}
R.~D. Moser, J.~Kim, and N.~N. Mansour.
\newblock \href{https://turbulence.ices.utexas.edu/data/MKM/}{Direct numerical simulation of turbulent channel flow up to $\mathrm{Re}_\tau = 590$}.
\newblock {\em Physics of Fluids}, 11(4):943--945, 1999. \href{https://turbulence.ices.utexas.edu/data/MKM/}{[\textsc{mkm} Database]}.""")

# Update White entry (remove generic homepage)
old_white = r"""\bibitem{white2006}
F.~M. White.
\newblock \href{https://www.mheducation.com}{{\em Viscous Fluid Flow}}.
\newblock McGraw-Hill, New York, 3rd edition, 2006. \href{https://www.mheducation.com}{[\textsc{isbn}: 0-07-240231-8]}."""
assert old_white in text, "old_white not found"
text = text.replace(old_white, r"""\bibitem{white2006}
F.~M. White.
\newblock {\em Viscous Fluid Flow}.
\newblock McGraw-Hill, New York, 3rd edition, 2006. [\textsc{isbn}: 0-07-240231-8].""")

# Update Raissi entry to arXiv PDF #page=6
old_raissi = r"""\bibitem{raissi2019}
M.~Raissi, P.~Perdikaris, and G.~E. Karniadakis.
\newblock \href{https://doi.org/10.1016/j.jcp.2018.10.045}{Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations}.
\newblock {\em Journal of Computational Physics}, 378:686--707, 2019. \href{https://doi.org/10.1016/j.jcp.2018.10.045}{[\textsc{doi}: \texttt{10.1016/j.jcp.2018.10.045}]}."""
assert old_raissi in text, "old_raissi not found"
text = text.replace(old_raissi, r"""\bibitem{raissi2019}
M.~Raissi, P.~Perdikaris, and G.~E. Karniadakis.
\newblock \href{https://arxiv.org/pdf/1711.10561.pdf#page=6}{Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations}.
\newblock {\em Journal of Computational Physics}, 378:686--707, 2019. \href{https://arxiv.org/pdf/1711.10561.pdf#page=6}{[\textsc{arxiv} \textsc{pdf} p.~6]}.""")

# Update Wang entry to arXiv PDF #page=10
old_wang = r"""\bibitem{wang2021}
S.~Wang, Y.~Teng, and P.~Perdikaris.
\newblock \href{https://arxiv.org/abs/2001.04536}{Understanding and mitigating gradient pathologies in physics-informed neural networks}.
\newblock {\em SIAM Journal on Scientific Computing}, 43(5):A3055--A3081, 2021. \href{https://arxiv.org/abs/2001.04536}{[arXiv:2001.04536]}."""
assert old_wang in text, "old_wang not found"
text = text.replace(old_wang, r"""\bibitem{wang2021}
S.~Wang, Y.~Teng, and P.~Perdikaris.
\newblock \href{https://arxiv.org/pdf/2001.04536.pdf#page=10}{Understanding and mitigating gradient pathologies in physics-informed neural networks}.
\newblock {\em SIAM Journal on Scientific Computing}, 43(5):A3055--A3081, 2021. \href{https://arxiv.org/pdf/2001.04536.pdf#page=10}{[\textsc{arxiv} \textsc{pdf} p.~10]}.""")

# Update Pirozzoli entry to arXiv PDF #page=7
old_piroz = r"""\bibitem{pirozzoli2021}
S.~Pirozzoli, M.~Romero, M.~Fatica, R.~Verzicco, and P.~Orlandi.
\newblock \href{https://arxiv.org/abs/2103.13383}{One-point statistics for turbulent pipe flow up to $\mathrm{Re}_\tau = 6000$}.
\newblock {\em Journal of Fluid Mechanics}, 926:A28, 2021. \href{https://arxiv.org/abs/2103.13383}{[arXiv:2103.13383]}."""
assert old_piroz in text, "old_piroz not found"
text = text.replace(old_piroz, r"""\bibitem{pirozzoli2021}
S.~Pirozzoli, M.~Romero, M.~Fatica, R.~Verzicco, and P.~Orlandi.
\newblock \href{https://arxiv.org/pdf/2103.13383.pdf#page=7}{One-point statistics for turbulent pipe flow up to $\mathrm{Re}_\tau = 6000$}.
\newblock {\em Journal of Fluid Mechanics}, 926:A28, 2021. \href{https://arxiv.org/pdf/2103.13383.pdf#page=7}{[\textsc{arxiv} \textsc{pdf} p.~7]}.""")

# Update Spalart entry to NASA TMR link
old_spalart = r"""\bibitem{spalart1992}
P.~R. Spalart and S.~R. Allmaras.
\newblock \href{https://ntrs.nasa.gov/api/citations/19930094770/downloads/19930094770.pdf#page=1}{A one-equation turbulence model for aerodynamic flows}.
\newblock {\em AIAA Paper 92-0439}, 1992. \href{https://ntrs.nasa.gov/api/citations/19930094770/downloads/19930094770.pdf#page=1}{[\textsc{pdf} p.~1]}."""
assert old_spalart in text, "old_spalart not found"
text = text.replace(old_spalart, r"""\bibitem{spalart1992}
P.~R. Spalart and S.~R. Allmaras.
\newblock \href{https://turbmodels.larc.nasa.gov/spalart.html}{A one-equation turbulence model for aerodynamic flows}.
\newblock {\em AIAA Paper 92-0439}, 1992. \href{https://turbmodels.larc.nasa.gov/spalart.html}{[\textsc{nasa} \textsc{tmr}; AIAA 92-0439]}.""")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated bibliography in report.tex successfully.")

