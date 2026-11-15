with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if "begin{tikzpicture}" in l or "Cross-Section" in l or "Uniform U0" in l or "3D Walls" in l:
        print(f"{i}: {l.strip()}")

