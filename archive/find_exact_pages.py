import pypdf

def find_pages(pdf_path, keywords):
    reader = pypdf.PdfReader(pdf_path)
    print(f"\n==========================================")
    print(f"FILE: {pdf_path} (Total Pages: {len(reader.pages)})")
    print(f"==========================================")
    for kw in keywords:
        hits = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if kw.lower() in txt.lower():
                hits.append(i + 1)
        print(f"Keyword '{kw}': found on PDF pages {hits[:8]}")

find_pages("raissi2019.pdf", ["Navier-Stokes", "continuous time", "discrete time", "equation (2.1)"])
find_pages("glorot2010.pdf", ["variance", "Equation (16)", "normalized", "249"])
find_pages("bergstra2012.pdf", ["random search", "Figure 1", "log-uniform", "effective dimension"])
find_pages("allmaras2012.pdf", ["negative", "sa-noft2", "destruction", "c_b1"])
find_pages("wang2021.pdf", ["gradient pathologies", "learning rate annealing", "algorithm 1", "table 1"])
find_pages("wang2022.pdf", ["neural tangent kernel", "eigenvalues", "trace", "algorithm 1"])
find_pages("jin2021.pdf", ["nsfnets", "pressure gap", "velocity-pressure", "table 1"])
find_pages("lu2021.pdf", ["deepxde", "residual", "geometry", "pde"])
find_pages("rao2020.pdf", ["incompressible laminar", "navier-stokes", "loss function", "boundary"])
find_pages("pirozzoli2021.pdf", ["superpipe", "mean velocity", "6000", "table 1", "law of the wall"])

