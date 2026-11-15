with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8") as f:
    text = f.read()

import re
m = re.search(r'\\bibitem\{ray2017\}.*?(?=\\bibitem|\\end\{thebibliography\})', text, re.DOTALL)
if m:
    print(m.group(0))

