with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if "Dean" in line or "tsukahara" in line or "0.073" in line:
        print(f"{i}: {line.strip()}")

