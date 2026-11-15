import os

base = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/5"
found = []
for root, dirs, files in os.walk(base):
    for f in files:
        p = os.path.join(root, f)
        try:
            with open(p, encoding="utf-8", errors="ignore") as fp:
                for lno, line in enumerate(fp, 1):
                    if "1.93" in line or "0.537" in line:
                        found.append((os.path.relpath(p, "c:/Users/divya/Downloads/openfoam_pinn_pipeline"), lno, line.strip()))
        except Exception:
            pass

print("Matches in 5/:")
for path, lno, line in found:
    print(f"{path}:{lno}: {line}")

if not found:
    print("NO matches in 5/! Checking scripts that analyze 5/...")
    for root, dirs, files in os.walk("c:/Users/divya/Downloads/openfoam_pinn_pipeline"):
        if "backup" in root or ".git" in root or "sources" in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fp:
                    for lno, line in enumerate(fp, 1):
                        if "1.930" in line or "0.537" in line:
                            print(f"{os.path.relpath(p, 'c:/Users/divya/Downloads/openfoam_pinn_pipeline')}:{lno}: {line.strip()}")
            except Exception:
                pass

