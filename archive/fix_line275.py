path = r'.\report.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

old_s = 'Normalized coordinates evaluate to $\\hat{x} = 0.125$, $\\hat{y} = 0.400$, and $\\mathrm{Re}_n = -0.3979$ (from $2\\log_{10}(100/50) - 1$). The network outputs $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$.'
new_s = 'Normalized coordinates evaluate to $\\hat{x} = 0.125$, $\\hat{y} = 0.400$, and $\\mathrm{Re}_n = -0.3979$. The network outputs normalized vector $\\hat{\\mathbf{u}} = [1.2933, 0.0168, 16.6964]^T$.'

assert old_s in text, 'old_s not found'
text = text.replace(old_s, new_s)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Updated line 275')

