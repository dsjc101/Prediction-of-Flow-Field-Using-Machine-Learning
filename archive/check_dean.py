import pypdf

reader = pypdf.PdfReader("c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/02_tsukahara_2014/1406.0248.pdf")
print("Total pages:", len(reader.pages))

for idx, page in enumerate(reader.pages):
    txt = page.extract_text()
    if "0.073" in txt or "Dean" in txt:
        print(f"Page index {idx+1} (0-based {idx}):")
        for line in txt.splitlines():
            if "0.073" in line or "Dean" in line or "Cf" in line or "Rem" in line or "C_f" in line:
                print("   ", line)

