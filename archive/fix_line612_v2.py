path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

old_s = 'Reference truth data was computed using an independent parabolic marching solver (\\texttt{4/} \\texttt{reference\\_solver.py}) with the SA-noft2 closure and the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}. The solver solves the thin-shear-layer (parabolic) boundary-layer equations:'
new_s = 'Reference truth data was computed with an independent marching solver (\\texttt{4/}\\allowbreak\\texttt{reference\\_solver.py}) using the SA-noft2 closure~\cite{allmaras2012}. The code integrates the parabolic thin-shear-layer equations:'

assert old_s in text, 'old_s not found'
text = text.replace(old_s, new_s)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Updated line 612.')

