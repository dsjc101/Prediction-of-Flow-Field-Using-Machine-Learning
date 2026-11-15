import os

for path in [
    "c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs/ablation_summary.txt",
    "c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs_sparse/ablation_summary.txt",
    "c:/Users/divya/Downloads/openfoam_pinn_pipeline/5/outputs/ablation_sparse.txt"
]:
    if os.path.exists(path):
        print("=== " + path + " ===")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            print(f.read())
    else:
        print("MISSING:", path)

