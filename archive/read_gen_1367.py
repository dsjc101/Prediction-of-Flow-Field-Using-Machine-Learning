with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/generate_all_perfect_plots.py", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(1360, 1375):
    print(f"{i+1}: {lines[i]}", end="")

