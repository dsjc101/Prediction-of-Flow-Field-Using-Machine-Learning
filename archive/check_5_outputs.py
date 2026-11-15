import os

files = os.listdir("c:/Users/divya/Downloads/openfoam_pinn_pipeline/5/outputs")
print([f for f in files if "ablation" in f or "dense" in f or "log" in f])

