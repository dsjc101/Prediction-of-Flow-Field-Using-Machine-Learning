path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Line 612
old_612 = 'Reference truth data was computed with an independent parabolized RANS marching solver (\\texttt{4/reference\\_}\\allowbreak\\texttt{solver.py}) using the SA-noft2 closure with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
new_612 = 'Reference truth data was computed using an independent parabolized RANS marching solver (\\texttt{4/reference\\_solver.py}) with the SA-noft2 closure and the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
assert old_612 in text, 'old_612 not found'
text = text.replace(old_612, new_612)

# 2. Line 765
old_765 = '\\noindent\\textbf{Terms Dropped in Parabolic Formulation:} The marching solver drops all axial diffusion ($\\partial^2 U/\\partial x^2 = 0, \\partial^2 \\tilde{\\nu}/\\partial x^2 = 0$) and drops radial momentum balance ($\\partial P/\\partial r = 0 \\implies P = P(x)$).'
new_765 = '\\noindent\\textbf{Terms Dropped in Parabolic Formulation:} The marching solver drops all axial diffusion ($\\partial^2 U/\\partial x^2 = 0$, $\\partial^2 \\tilde{\\nu}/\\partial x^2 = 0$) and drops radial momentum balance ($\\partial P/\\partial r = 0 \\implies P = P(x)$).'
assert old_765 in text, 'old_765 not found'
text = text.replace(old_765, new_765)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Fixed 2 overfull lines.')

