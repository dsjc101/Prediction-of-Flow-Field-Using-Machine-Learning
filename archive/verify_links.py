import urllib.request, urllib.error

urls = {
    "raissi2019_arxiv": "https://arxiv.org/pdf/1711.10561.pdf",
    "wang2021_arxiv": "https://arxiv.org/pdf/2001.04536.pdf",
    "pirozzoli2021_arxiv": "https://arxiv.org/pdf/2103.13383.pdf",
    "glorot2010_pmlr": "https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
    "hornbeck1973_ntrs": "https://ntrs.nasa.gov/api/citations/19740008333/downloads/19740008333.pdf",
    "allmaras2012_nasa": "https://turbmodels.larc.nasa.gov/spalart.html",
    "openfoam_guide": "https://www.openfoam.com/documentation/user-guide",
    "moser1999_data": "https://turbulence.ices.utexas.edu/data/MKM/",
    "moser1999_doi": "https://doi.org/10.1063/1.869966"
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for k, url in urls.items():
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            code = response.getcode()
            ctype = response.info().get_content_type()
            length = response.info().get('Content-Length', 'unknown')
            print(f"{k}: {code} | {ctype} | length={length}")
    except Exception as e:
        print(f"{k}: ERROR {e}")

