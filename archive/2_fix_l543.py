with open('../report.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines[542] = r"Unlike Stage~1 ($p_{\mathrm{scale}} = 0.5 U_0^2$) and Stage~2 ($p_{\mathrm{scale}} = 32\nu U_0 L / D^2$), the Stage~3 scale $p_{\mathrm{scale}} = 0.5 U_{\mathrm{ref}}^2 (L / D_h) = 0.10547\,\text{m}^2/\text{s}^2$ scales dynamic pressure by duct slenderness $L / D_h = 37.5$ to characterize cumulative frictional pressure drop. The normalized inputs evaluate to $\hat{x}=0.5$, $\hat{y}=0.5$, $\hat{z}=0.5$, and $\mathrm{Re}_n=-0.4421$. With $U_{\mathrm{ref}} = 0.075$ m/s and $p_{\mathrm{scale}} = 0.10547$ m$^2$/s$^2$, physical predictions are:" + "\n"

with open('../report.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed line 543 cleanly!")

