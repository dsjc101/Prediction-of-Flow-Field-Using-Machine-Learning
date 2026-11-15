import subprocess, re

def search_pdf(fname, patterns):
    print(f"=== {fname} ===")
    p = subprocess.run(["pdfinfo", fname], capture_output=True, text=True)
    pages = 10
    for line in p.stdout.splitlines():
        if "Pages:" in line:
            pages = int(line.split(":")[1].strip())
    print(f"Total pages: {pages}")
    
    for page in range(1, pages + 1):
        txt = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), fname, "-"], capture_output=True, text=True).stdout
        for pat in patterns:
            if re.search(pat, txt, re.IGNORECASE):
                print(f"Page {page} matches '{pat}': {txt[:120].strip()}...")

search_pdf("raissi.pdf", ["Navier-Stokes", "continuous time", "discrete time", "loss function", "mean squared error"])
search_pdf("wang.pdf", ["gradient pathologies", "learning rate annealing", "algorithm 1", "pathology"])
search_pdf("pirozzoli.pdf", ["Superpipe", "mean velocity", "friction", "Re_tau", "6000"])

