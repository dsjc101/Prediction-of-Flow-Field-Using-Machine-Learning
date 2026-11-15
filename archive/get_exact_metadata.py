import pypdf

def get_exact(fname, targets):
    reader = pypdf.PdfReader(fname)
    print(f"\n=== {fname} ({len(reader.pages)} pages) ===")
    for target in targets:
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if target.lower() in txt.lower():
                # print snippet
                idx = txt.lower().find(target.lower())
                snippet = txt[max(0, idx-50):min(len(txt), idx+150)].replace("\n", " ")
                print(f"  [PDF Page {i+1}] '{target}': {snippet[:120]}...")
                break

get_exact("raissi2019.pdf", ["continuous time models", "Navier-Stokes", "discrete time models"])
get_exact("glorot2010.pdf", ["normalized initialization", "variance of the states", "Equation (16)"])
get_exact("bergstra2012.pdf", ["random search", "Figure 1", "log-uniform"])
get_exact("allmaras2012.pdf", ["negative", "sa-noft2", "destruction"])
get_exact("wang2021.pdf", ["learning rate annealing", "gradient pathologies", "algorithm 1"])
get_exact("wang2022.pdf", ["neural tangent kernel", "trace", "algorithm 1"])
get_exact("jin2021.pdf", ["nsfnets", "pressure gap", "velocity-pressure"])
get_exact("lu2021.pdf", ["deepxde", "residual", "collocation"])
get_exact("rao2020.pdf", ["incompressible laminar", "navier-stokes", "loss function"])
get_exact("pirozzoli2021.pdf", ["superpipe", "6000", "mean velocity"])

