import os

ROOT = r".\Downloads\openfoam_pinn_pipeline"
report_path = os.path.join(ROOT, "report.tex")

with open(report_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Verify line 713
line_713 = lines[712]
assert "dean1978" in line_713, "dean1978 not in line 713"

# We want to replace Dean's experimental skin friction correlation ... \cite{dean1978}
# with Dean's skin friction correlation ($C_f = 0.073\,\mathrm{Re}_m^{-0.25}$, as quoted in Tsukahara et al.~\cite{tsukahara2014} [p.~6])
# Let's see the exact text in line_713:
start_str = "Agreement with Dean's experimental skin friction correlation"
end_str = r"\cite{dean1978}"

assert start_str in line_713, "start_str not in line 713"
assert end_str in line_713, "end_str not in line 713"

idx1 = line_713.index(start_str)
idx2 = line_713.index(end_str) + len(end_str)

old_segment = line_713[idx1:idx2]
print("Old segment:", repr(old_segment))

new_segment = r"Agreement with Dean's skin friction correlation ($C_f = 0.073\,\mathrm{Re}_m^{-0.25}$, as quoted in Tsukahara et al.~\cite{tsukahara2014} [p.~6])"

lines[712] = line_713[:idx1] + new_segment + line_713[idx2:]

# Now replace bibitem dean1978 with tsukahara2014
full_text = "".join(lines)

old_dean = r"""\bibitem{dean1978}
R.~B. Dean.
\newblock \href{https://doi.org/10.1115/1.3448633}{Reynolds number dependence of skin friction and other bulk flow variables in two-dimensional rectangular duct flow}.
\newblock {\em ASME Journal of Fluids Engineering}, 100(2):215--223, 1978."""

new_dean = r"""\bibitem{tsukahara2014}
T.~Tsukahara, Y.~Seki, H.~Kawamura, and D.~Tochio.
\newblock \href{https://arxiv.org/pdf/1406.0248.pdf#page=6}{DNS of turbulent channel flow at very low Reynolds numbers}.
\newblock {\em arXiv preprint arXiv:1406.0248}, 2014. \href{https://arxiv.org/pdf/1406.0248.pdf#page=6}{[\textsc{arxiv} \textsc{pdf} p.~6]}."""

assert old_dean in full_text, "old_dean not found in full_text"
full_text = full_text.replace(old_dean, new_dean)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("Step 3 applied successfully!")

