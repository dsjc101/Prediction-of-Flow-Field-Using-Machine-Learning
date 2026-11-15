import urllib.request, ssl

urls = {
    'allmaras2012_pdf': 'https://turbmodels.larc.nasa.gov/spalart_iccfd7.pdf',
    'hornbeck_sp297_pdf': 'https://ntrs.nasa.gov/api/citations/19740008333/downloads/19740008333.pdf',
    'hornbeck_larc_sp297': 'https://ntrs.nasa.gov/api/citations/19740008333/downloads/19740008333_1974008333.pdf',
    'hornbeck_landing': 'https://ntrs.nasa.gov/citations/19740008333',
    'spalart_larc_html': 'https://turbmodels.larc.nasa.gov/spalart.html',
    'moser_data': 'https://turbulence.ices.utexas.edu/data/MKM/',
    'openfoam_guide': 'https://www.openfoam.com/documentation/user-guide',
    'shah1978_archive': 'https://archive.org/details/laminarflowforce0000shah',
    'glorot_pdf': 'https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf',
    'raissi_pdf': 'https://arxiv.org/pdf/1711.10561.pdf',
    'wang_pdf': 'https://arxiv.org/pdf/2001.04536.pdf',
    'piroz_pdf': 'https://arxiv.org/pdf/2103.13383.pdf'
}

ctx = ssl.create_default_context()
headers = {'User-Agent': 'Mozilla/5.0'}

for name, url in urls.items():
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            ct = resp.headers.get("Content-Type", "")
            cl = resp.headers.get("Content-Length", "")
            print(f"{name}: {resp.status} | {ct} | len={cl}")
    except Exception as e:
        print(f"{name}: ERROR {e}")

