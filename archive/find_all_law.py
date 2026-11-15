import os

for root, dirs, files in os.walk("c:/Users/divya/Downloads/openfoam_pinn_pipeline"):
    if "backup" in root or ".git" in root:
        continue
    for file in files:
        if file.endswith((".py", ".json", ".txt", ".csv")):
            p = os.path.join(root, file)
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "1.930" in content:
                    print("Found in:", os.path.relpath(p, "c:/Users/divya/Downloads/openfoam_pinn_pipeline"))
            except Exception:
                pass

