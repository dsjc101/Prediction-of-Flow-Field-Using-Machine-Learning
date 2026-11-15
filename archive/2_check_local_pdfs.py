import subprocess, os

pdfs = [
    ("glorot2010.pdf", 4, ["normalized", "initialization", "variance"]),
    ("raissi2019.pdf", 6, ["loss", "MSE", "Navier"]),
    ("wang2021.pdf", 10, ["Algorithm", "learning rate", "annealing"]),
    ("pirozzoli_real.pdf", 7, ["Superpipe", "friction", "Re_tau"]),
    ("allmaras2012.pdf", 1, ["Modifications", "Clarifications", "Spalart"]),
    ("hornbeck1973.pdf", 1, ["Numerical", "marching", "Hornbeck"]),
    ("spalart1994.pdf", 1, ["Spalart", "Allmaras", "one-equation"])
]

for fname, page, keywords in pdfs:
    cmd = ["pdftotext", "-f", str(page), "-l", str(page), fname, "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    text = " ".join([l.strip() for l in res.stdout.splitlines() if l.strip()])
    print(f"=== {fname} page {page} ===")
    print(text[:250])
    print("-" * 50)

