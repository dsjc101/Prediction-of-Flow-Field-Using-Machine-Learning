import os

for f in os.listdir("c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs"):
    if f.startswith("abl_") or f.endswith(".csv") or f.endswith(".txt"):
        print(f"4/outputs/{f}")

