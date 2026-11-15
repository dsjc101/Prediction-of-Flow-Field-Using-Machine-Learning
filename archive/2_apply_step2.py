import os

ROOT = r".\Downloads\openfoam_pinn_pipeline"
report_path = os.path.join(ROOT, "report.tex")

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# Stage 1: parallel plates entrance length
old_s1 = r"""The hydrodynamic entrance length $L_e$ for laminar flow between parallel plates is estimated using the composite correlation (correlation, source not verified; cf.~Atkinson et al.~\cite{atkinson1969} and Shah and London~\cite{shah1978}):
\begin{equation}
\frac{L_e}{D_h} = \frac{0.315}{1 + 0.0175\,\mathrm{Re}} + 0.011\,\mathrm{Re}.
\end{equation}
For $\mathrm{Re}=100$, $L_e = 0.243\,\text{m}$; for $\mathrm{Re}=300$, $L_e = 0.670\,\text{m}$; and for $\mathrm{Re}=600$, $L_e = 1.325\,\text{m}$. All entrance lengths fit comfortably within $L = 4.0\,\text{m}$."""

new_s1 = r"""The hydrodynamic entrance length $L_e$ for laminar flow between parallel plates is estimated with an empirical correlation (constants not verified against a free source; see CHANGELOG):
\begin{equation}
\frac{L_e}{D_h} = \frac{0.315}{1 + 0.0175\,\mathrm{Re}} + 0.011\,\mathrm{Re}.
\end{equation}
For $\mathrm{Re}=100$, $L_e = 0.243\,\text{m}$; for $\mathrm{Re}=300$, $L_e = 0.670\,\text{m}$; and for $\mathrm{Re}=600$, $L_e = 1.325\,\text{m}$. All entrance lengths fit comfortably within $L = 4.0\,\text{m}$. For comparison, the free-literature correlation of Durst et al. as reproduced in~\cite{ray2017} [p.~8] gives $L_e = 0.239\,\text{m}$ at $\mathrm{Re} = 100$ and $1.332\,\text{m}$ at $\mathrm{Re} = 600$."""

assert old_s1 in text, "old_s1 not found"
text = text.replace(old_s1, new_s1)

# Stage 2: circular pipe entrance length
old_s2 = r"""The circular pipe hydrodynamic entrance length correlation is computed via the standard engineering correlation (not verified against~\cite{chen1973}):
\begin{equation}
\frac{L_e}{D} = \frac{0.6}{0.14\,\mathrm{Re} + 1.0} + 0.0575\,\mathrm{Re}.
\end{equation}
At $\mathrm{Re}=100$, $L_e = 0.579\,\text{m}$; at $\mathrm{Re}=600$, $L_e = 3.457\,\text{m}$."""

new_s2 = r"""The circular pipe hydrodynamic entrance length correlation is estimated with an empirical correlation (constants not verified against a free source; see CHANGELOG):
\begin{equation}
\frac{L_e}{D} = \frac{0.6}{0.14\,\mathrm{Re} + 1.0} + 0.0575\,\mathrm{Re}.
\end{equation}
At $\mathrm{Re}=100$, $L_e = 0.579\,\text{m}$; at $\mathrm{Re}=600$, $L_e = 3.457\,\text{m}$. For comparison, the free-literature correlation of Durst et al. as reproduced in~\cite{ray2017} [p.~8] gives $L_e = 0.577\,\text{m}$ at $\mathrm{Re} = 100$ and $3.405\,\text{m}$ at $\mathrm{Re} = 600$."""

assert old_s2 in text, "old_s2 not found"
text = text.replace(old_s2, new_s2)

# Update bibliography: remove atkinson1969, chen1973, shah1978; add ray2017
old_atkinson = r"""\bibitem{atkinson1969}
B.~Atkinson, M.~P. Brocklebank, M.~H. Cardew, and J.~M. Smith.
\newblock \href{https://doi.org/10.1002/aic.690150414}{Low {Reynolds} number developing flows}.
\newblock {\em AIChE Journal}, 15(4):548--554, 1969."""

old_chen = r"""\bibitem{chen1973}
R.-Y. Chen.
\newblock \href{https://doi.org/10.1115/1.3446948}{Flow in the entrance region at low {Reynolds} numbers}.
\newblock {\em ASME Journal of Fluids Engineering}, 95(1):153--158, 1973."""

old_shah = r"""\bibitem{shah1978}
R.~K. Shah and A.~L. London.
\newblock {\em Laminar Flow Forced Convection in Ducts}.
\newblock Academic Press, New York, 1978."""

ray_entry = r"""\bibitem{ray2017}
B.~Ray, F.~Durst, and S.~Ray.
\newblock \href{https://arxiv.org/pdf/1707.04947.pdf#page=8}{Pressure drop and flow development in the entrance region of micro-channels with second order velocity slip condition and the requirement for development length}.
\newblock {\em arXiv preprint arXiv:1707.04947}, 2017. \href{https://arxiv.org/pdf/1707.04947.pdf#page=8}{[\textsc{arxiv} \textsc{pdf} p.~8]}."""

assert old_atkinson in text, "old_atkinson not found"
assert old_chen in text, "old_chen not found"
# assert old_shah in text, "old_shah not found"

text = text.replace(old_atkinson + "\n\n", "")
text = text.replace(old_chen + "\n\n", "")
# text = text.replace(old_shah + "\n\n", "")

# insert ray_entry before superpipe_data
text = text.replace(r"\bibitem{superpipe_data}", ray_entry + "\n\n" + r"\bibitem{superpipe_data}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Step 2 changes applied successfully.")
