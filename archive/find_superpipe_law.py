with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if "1.930" in line or "Superpipe friction" in line or "friction-law" in line or "0.537" in line:
        print(f"{i}: {line.strip()}")

