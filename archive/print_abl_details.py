import json

for name in ["abl_A_found.json", "abl_B_data_only.json"]:
    p = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs/" + name
    with open(p) as f:
        print(f"=== {name} ===")
        print(f.read()[:300])

for name in ["abl_A_found_summary.txt", "abl_B_data_only_summary.txt"]:
    p = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs/" + name
    with open(p) as f:
        print(f"=== {name} ===")
        print(f.read()[:500])

