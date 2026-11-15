import os
for root, dirs, files in os.walk("c:/Users/divya/Downloads/openfoam_pinn_pipeline/5"):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            txt = open(p, encoding="utf-8", errors="ignore").read()
            if "friction" in txt.lower() or "cf" in txt.lower():
                print(os.path.relpath(p, "c:/Users/divya/Downloads/openfoam_pinn_pipeline"))

