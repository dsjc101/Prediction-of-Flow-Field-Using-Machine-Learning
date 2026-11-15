with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/5/evaluate.py", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if "friction" in l.lower() or "cf" in l.lower() or "log10" in l.lower():
        print(f"5/evaluate.py:{i}: {l.strip()[:90]}")

with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/5/deploy_and_visualize.py", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
for i, l in enumerate(lines, 1):
    if "friction" in l.lower() or "cf" in l.lower() or "log10" in l.lower():
        print(f"5/deploy_and_visualize.py:{i}: {l.strip()[:90]}")

