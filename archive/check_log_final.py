with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.log", encoding="latin-1") as f:
    log = f.read()

import re
warnings = [l for l in log.splitlines() if "warning" in l.lower() and "infwarerr" not in l.lower()]
print(f"Non-infwarerr warnings: {len(warnings)}")
for w in warnings:
    print("  ", w)

undef = [l for l in log.splitlines() if "undefined" in l.lower()]
print(f"Undefined references: {len(undef)}")
for u in undef:
    print("  ", u)

overfulls = [l for l in log.splitlines() if "overfull" in l.lower()]
print(f"Overfull boxes: {len(overfulls)}")
for o in overfulls:
    print("  ", o)

