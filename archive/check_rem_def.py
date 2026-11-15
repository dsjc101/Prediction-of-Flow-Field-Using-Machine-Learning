import pypdf

reader = pypdf.PdfReader("c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/02_tsukahara_2014/1406.0248.pdf")
for i in range(6):
    txt = reader.pages[i].extract_text()
    for line in txt.splitlines():
        if "Rem" in line or "Re_m" in line or "bulk" in line or "hydraulic" in line or "Reynolds" in line:
            print(f"p.{i+1}: {line}")

