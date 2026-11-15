with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if "table7_ablations" in l:
        print(f"{i}: {l.strip()}")
        for j in range(max(0, i-5), min(len(lines), i+10)):
            print(f"  {j+1}: {lines[j].strip()}")

