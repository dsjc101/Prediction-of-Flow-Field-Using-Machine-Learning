path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Line 275
old_275 = 'Normalized coordinates evaluate to $\\hat{x} = 0.5/4.0 = 0.125$, $\\hat{y} = 0.02/0.05 = 0.400$, and $\\mathrm{Re}_n = 2\\log_{10}(100/50)/1.0 - 1 = -0.3979$. The network outputs normalized vector $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$.\nRescaling by $U_0 = 0.05\\,\\text{m/s}$ and dynamic pressure scale $p_{\\mathrm{scale}} = 0.5 U_0^2 = 0.00125\\,\\text{m}^2/\\text{s}^2$ yields:'
new_275 = 'Normalized coordinates evaluate to $\\hat{x} = 0.125$, $\\hat{y} = 0.400$, and $\\mathrm{Re}_n = -0.3979$ (from $2\\log_{10}(100/50) - 1$). The network outputs $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$.\nRescaling by $U_0 = 0.05\\,\\text{m/s}$ and dynamic pressure scale $p_{\\mathrm{scale}} = 0.5 U_0^2 = 0.00125\\,\\text{m}^2/\\text{s}^2$ yields:'
assert old_275 in text, 'old_275 not found'
text = text.replace(old_275, new_275)

# 2. Line 378
old_378 = 'The axis symmetry penalty is $\\mathcal{L}_{\\mathrm{axis}} = \\frac{1}{N_{\\mathrm{ax}}}\\sum (v_k/U_0)^2$.\nThe final loss weights read from \\texttt{2/train\\_hybrid.py} are:'
new_378 = 'The axis symmetry penalty is $\\mathcal{L}_{\\mathrm{axis}} = \\frac{1}{N_{\\mathrm{ax}}}\\sum (v_k/U_0)^2$.\n\nThe final loss weights read from \\texttt{2/train\\_hybrid.py} are:'
assert old_378 in text, 'old_378 not found'
text = text.replace(old_378, new_378)

# 3. Line 415
old_415 = 'Table~\\ref{tab:stage2_metrics} reports quantitative errors ordered monotonically across all Reynolds numbers ($\\mathrm{Re} \\in \\{50, 100, 200, 300, 500, 600\\}$).'
new_415 = 'Table~\\ref{tab:stage2_metrics} reports quantitative errors ordered monotonically by Reynolds number ($\\mathrm{Re} = 50$, $100$, $200$, $300$, $500$, and $600$).'
assert old_415 in text, 'old_415 not found'
text = text.replace(old_415, new_415)

# 4. Line 611
old_611 = 'Reference truth data was computed with an independent parabolized RANS marching solver (\\texttt{4/reference\\_}\\allowbreak\\texttt{solver.py}) using the SA-noft2 model with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
new_611 = 'Reference truth data was computed with an independent parabolized RANS marching solver (in file \\texttt{4/reference\\_solver.py}) using the SA-noft2 closure with the Allmaras--Johnson--Spalart limiter~\\cite{allmaras2012}.'
assert old_611 in text, 'old_611 not found'
text = text.replace(old_611, new_611)

# 5. Line 774
old_774 = 'Table~\\ref{tab:stage5_act} reports the deterministic forward pass evaluated from checkpoint \\texttt{5/outputs/models/}\\allowbreak\\texttt{dense/final\\_seed0.pt} at $(x=40R, r=0.5R, \\mathrm{Re}=40{,}000)$.'
new_774 = 'Table~\\ref{tab:stage5_act} reports the deterministic forward pass evaluated from checkpoint \\texttt{5/outputs/models/dense/} \\texttt{final\\_seed0.pt} at $(x=40R, r=0.5R, \\mathrm{Re}=40{,}000)$.'
assert old_774 in text, 'old_774 not found'
text = text.replace(old_774, new_774)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Updated 5 remaining spots.')

