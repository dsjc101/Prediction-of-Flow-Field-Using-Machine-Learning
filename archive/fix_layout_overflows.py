import os

ROOT = r"."

# 1. Update tables/table_c4_weights.tex
table_c4_path = os.path.join(ROOT, "tables", "table_c4_weights.tex")
with open(table_c4_path, "r", encoding="utf-8") as f:
    c4_text = f.read()

# Wrap tabular in resizebox
c4_text = c4_text.replace(
    r"\begin{tabular}{lccc}",
    r"\resizebox{\textwidth}{!}{\begin{tabular}{lccc}"
).replace(
    r"\end{tabular}" + "\n" + r"\caption{Training configurations, collocation counts, and loss weights for laminar flow stages (Stages 1--3).}",
    r"\end{tabular}}" + "\n" + r"\caption{Training configurations, collocation counts, and loss weights for laminar flow stages (Stages 1--3).}"
)

c4_text = c4_text.replace(
    r"\begin{tabular}{lcc}",
    r"\resizebox{\textwidth}{!}{\begin{tabular}{lcc}"
).replace(
    r"\end{tabular}" + "\n" + r"\caption{Training configurations, collocation counts, and loss weights for turbulent RANS+SA stages (Stages 4--5).}",
    r"\end{tabular}}" + "\n" + r"\caption{Training configurations, collocation counts, and loss weights for turbulent RANS+SA stages (Stages 4--5).}"
)

with open(table_c4_path, "w", encoding="utf-8") as f:
    f.write(c4_text)

# 2. Update report.tex figures
report_path = os.path.join(ROOT, "report.tex")
with open(report_path, "r", encoding="utf-8") as f:
    rep_text = f.read()

# Replace Fig 3
old_fig3 = r"""\begin{figure}[H]
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

new_fig3 = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{stage1_error_u.png}
\caption{Stage 1: Linear-axis axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%] across the Reynolds sweep.}
\label{fig:stage1_errors}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{stage1_error_p.png}
\caption{Stage 1: Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%] across the Reynolds sweep.}
\label{fig:stage1_error_p}
\end{figure}"""
assert old_fig3 in rep_text, "old_fig3 not found"
rep_text = rep_text.replace(old_fig3, new_fig3)

# Replace Fig 5
old_fig5 = r"""\begin{figure}[H]
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

new_fig5 = r"""\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{stage2_error_u.png}
\caption{Stage 2: Axisymmetric axial velocity error $|u - u_{\mathrm{ref}}|/U_0$ [\%] across the Reynolds sweep.}
\label{fig:stage2_errors}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{stage2_error_p.png}
\caption{Stage 2: Kinematic pressure error $|p - p_{\mathrm{ref}}|/p_{\mathrm{scale}}$ [\%] across the Reynolds sweep.}
\label{fig:stage2_error_p}
\end{figure}"""
assert old_fig5 in rep_text, "old_fig5 not found"
rep_text = rep_text.replace(old_fig5, new_fig5)

# Replace duct_error_domain_u_linear.png width
old_duct_fig = r"\includegraphics[width=\textwidth]{duct_error_domain_u_linear.png}"
new_duct_fig = r"\includegraphics[width=0.92\textwidth]{duct_error_domain_u_linear.png}"
assert old_duct_fig in rep_text, "old_duct_fig not found"
rep_text = rep_text.replace(old_duct_fig, new_duct_fig)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(rep_text)

print("Applied layout fixes successfully.")

