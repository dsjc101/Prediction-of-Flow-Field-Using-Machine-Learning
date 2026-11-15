with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'1\.930', text)]
for idx in matches:
    print(text[max(0, idx-100):min(len(text), idx+150)])
    print("="*60)

