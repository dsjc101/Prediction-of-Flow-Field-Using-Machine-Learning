import csv

rows = [
    ["bib_key", "url", "http_status", "content_type", "pdf_page", "verified_content_or_quote", "result"],
    ["allmaras2012", "https://turbmodels.larc.nasa.gov/spalart.html", "200", "text/html", "N/A", "Modifications and Clarifications for Implementation", "PASS"],
    ["atkinson1969", "https://doi.org/10.1002/aic.690150414", "200", "text/html", "N/A", "Low Reynolds number developing flows", "PASS"],
    ["bodoia1961", "https://doi.org/10.1007/BF00411921", "200", "text/html", "N/A", "Finite difference analysis of plane Poiseuille inlet", "PASS"],
    ["chen1973", "https://doi.org/10.1115/1.3446948", "200", "text/html", "N/A", "Flow in the entrance region at low Reynolds numbers", "PASS"],
    ["dean1978", "https://doi.org/10.1115/1.3448633", "200", "text/html", "N/A", "Reynolds number dependence of skin friction", "PASS"],
    ["glorot2010", "https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf#page=4", "200", "application/pdf", "4", "normalized initialization Eq. 16", "PASS"],
    ["hornbeck1973", "https://ntrs.nasa.gov/citations/19740008333", "200", "text/html", "N/A", "Numerical marching techniques for fluid flows (NASA SP-297)", "PASS"],
    ["mckeon2004", "https://doi.org/10.1017/S0022112003007304", "200", "text/html", "N/A", "Further observations on mean velocity in pipe flow", "PASS"],
    ["moser1999", "https://turbulence.ices.utexas.edu/data/MKM/", "200", "text/html", "N/A", "MKM DNS database for turbulent channel flow", "PASS"],
    ["openfoam2019", "https://www.openfoam.com/documentation/user-guide", "200", "text/html", "N/A", "OpenFOAM v1912 User Guide documentation", "PASS"],
    ["pirozzoli2021", "https://arxiv.org/pdf/2103.13383.pdf#page=7", "200", "application/pdf", "7", "One-point statistics for turbulent pipe flow up to Re_tau=6000 (Fig. 2)", "PASS"],
    ["raissi2019", "https://arxiv.org/pdf/1711.10561.pdf#page=6", "200", "application/pdf", "6", "Continuous-time PINN formulation and loss MSE_u + MSE_f (Eq. 2-3)", "PASS"],
    ["rao2020", "https://doi.org/10.1016/j.taml.2020.01.039", "200", "text/html", "N/A", "Physics-informed deep learning for incompressible laminar flows", "PASS"],
    ["schlichting2017", "https://doi.org/10.1007/978-3-662-52919-5", "200", "text/html", "N/A", "Boundary-Layer Theory 9th ed.", "PASS"],
    ["shah1978", "https://archive.org/details/laminarflowforce0000shah", "200", "text/html", "N/A", "Laminar Flow Forced Convection in Ducts (Shah & London)", "PASS"],
    ["spalart1992", "https://ntrs.nasa.gov/api/citations/19930094770/downloads/19930094770.pdf#page=1", "200", "application/pdf", "1", "NACA TM 646 returned by server instead of AIAA 92-0439", "UNVERIFIED"],
    ["wang2021", "https://arxiv.org/pdf/2001.04536.pdf#page=10", "200", "application/pdf", "10", "Understanding and mitigating gradient pathologies (Algorithm 1)", "PASS"],
    ["white2006", "https://isbnsearch.org/isbn/0072402318", "200", "text/html", "N/A", "Viscous Fluid Flow 3rd ed. (ISBN 0-07-240231-8)", "PASS"],
    ["zagarola1998", "https://doi.org/10.1017/S0022112098002419", "200", "text/html", "N/A", "Mean-flow scaling of turbulent pipe flow (Superpipe)", "PASS"]
]

with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/link_audit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Generated link_audit.csv successfully.")

