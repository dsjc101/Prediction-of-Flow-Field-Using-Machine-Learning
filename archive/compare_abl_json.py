import json

for name in ["abl_A_found.json", "abl_B_data_only.json", "abl_C_physics_x100.json"]:
    p = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/4/outputs/" + name
    with open(p) as f:
        d = json.load(f)
        print(f"=== {name} ===")
        for k in ["seed", "w_data", "w_pde", "train_loss", "best_loss", "epochs"]:
            if k in d:
                print(f"  {k}: {d[k]}")
        print("  Keys:", list(d.keys())[:8])

