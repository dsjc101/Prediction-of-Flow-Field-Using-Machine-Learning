with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/generate_all_perfect_plots.py", encoding="utf-8") as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if "1.930" in l or "0.537" in l or "friction" in l.lower():
        print(f"{i}: {l.strip()[:100]}")

