import subprocess

def get_page_snippet(pdf, page):
    cmd = ['pdftotext', '-f', str(page), '-l', str(page), pdf, '-']
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    return ' '.join(lines[:3])

print('Glorot p4:', get_page_snippet('c:/Users/divya/Downloads/openfoam_pinn_pipeline/glorot10a.pdf', 4))
print('Raissi p6:', get_page_snippet('c:/Users/divya/Downloads/openfoam_pinn_pipeline/raissi.pdf', 6))
print('Wang p10:', get_page_snippet('c:/Users/divya/Downloads/openfoam_pinn_pipeline/wang.pdf', 10))
print('Pirozzoli p7:', get_page_snippet('c:/Users/divya/Downloads/openfoam_pinn_pipeline/pirozzoli.pdf', 7))

