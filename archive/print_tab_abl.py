with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(914, 946):
    print(f"{i+1}: {lines[i]}", end="")

