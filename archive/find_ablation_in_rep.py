with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if "ablation" in l.lower():
        print(f"{i}: {l.strip()}")

