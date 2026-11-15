import os

for root, dirs, files in os.walk("c:/Users/divya/Downloads/openfoam_pinn_pipeline/4"):
    for file in files:
        if file.endswith((".txt", ".json", ".csv", ".log")):
            p = os.path.join(root, file)
            try:
                txt = open(p, encoding="utf-8", errors="ignore").read()
                if "4.82" in txt or "4.75" in txt or "0.49" in txt:
                    print("Found in:", os.path.relpath(p, "c:/Users/divya/Downloads/openfoam_pinn_pipeline"))
            except Exception:
                pass

