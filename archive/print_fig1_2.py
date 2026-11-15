with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

print("=== FIG 1 ===")
for i in range(80, 102):
    print(f"{i+1}: {lines[i]}", end="")

print("\n=== FIG 2 ===")
for i in range(160, 205):
    print(f"{i+1}: {lines[i]}", end="")

