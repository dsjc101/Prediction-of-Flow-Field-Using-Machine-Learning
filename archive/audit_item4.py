import urllib.request, pypdf, re, os, csv

targets = [
    {
        "key": "allmaras2012",
        "url": "https://iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf",
        "anchor_page": 2,
        "local_pdf": "sources/01_allmaras_2012/ICCFD7-1902_paper.pdf",
        "terms": ["fv1", "cb1"]
    },
    {
        "key": "glorot2010",
        "url": "https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
        "anchor_page": 4,
        "local_pdf": "sources/03_glorot_2010/glorot10a.pdf",
        "terms": ["normalized initialization", "variance", "Var"]
    },
    {
        "key": "moser1999",
        "url": "https://turbulence.oden.utexas.edu/data/MKM/",
        "anchor_page": "N/A",
        "local_html": "sources/05_moser_1999/mkm_database_index.html",
        "terms": ["chan590", "MKM"]
    },
    {
        "key": "pirozzoli2021",
        "url": "https://arxiv.org/pdf/2103.13383.pdf",
        "anchor_page": 7,
        "local_pdf": "sources/06_pirozzoli_2021/2103.13383.pdf",
        "terms": ["Superpipe", "Reynolds-number-dependent"]
    },
    {
        "key": "superpipe_data",
        "url": "https://smits.princeton.edu/superpipe-pitot-data",
        "anchor_page": "N/A",
        "local_html": "sources/08_superpipe_2026/superpipe_pitot_data.html",
        "terms": ["Pitot", "Superpipe"]
    },
    {
        "key": "raissi2019",
        "url": "https://arxiv.org/pdf/1711.10561.pdf",
        "anchor_page": 6,
        "local_pdf": "sources/09_raissi_2019/1711.10561.pdf",
        "terms": ["parameters", "Navier-Stokes"]
    },
    {
        "key": "rao2020",
        "url": "https://arxiv.org/pdf/2002.10558.pdf",
        "anchor_page": 2,
        "local_pdf": "sources/10_rao_2020/2002.10558.pdf",
        "terms": ["Navier", "Automatic Differentiation"]
    },
    {
        "key": "wang2021",
        "url": "https://arxiv.org/pdf/2001.04536.pdf",
        "anchor_page": 10,
        "local_pdf": "sources/11_wang_2021/2001.04536.pdf",
        "terms": ["Algorithm 1", "annealing"]
    }
]

base_dir = "c:/Users/divya/Downloads/openfoam_pinn_pipeline"
audit_rows = []

for t in targets:
    key = t["key"]
    url = t["url"]
    anchor = t["anchor_page"]
    
    # Network check
    status = "UNKNOWN"
    ctype = "UNKNOWN"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        status = str(resp.status)
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip()
    except Exception:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            status = str(resp.status)
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip()
        except Exception as e:
            status = f"ERR"
            ctype = "N/A"
            
    snippet = ""
    verdict = "PASS" if status == "200" else "FAIL"
    
    if "local_pdf" in t:
        pdf_path = os.path.join(base_dir, t["local_pdf"])
        reader = pypdf.PdfReader(pdf_path)
        p_idx = t["anchor_page"] - 1
        txt = reader.pages[p_idx].extract_text()
        
        found = False
        for term in t["terms"]:
            if term.lower() in txt.lower():
                m = re.search(r'([^.\n]*?' + re.escape(term) + r'[^.\n]*)', txt, re.IGNORECASE)
                if m:
                    words = m.group(0).split()
                    snippet = " ".join(words[:min(10, len(words))])
                    found = True
                    break
        if not found:
            verdict = "FAIL"
            snippet = "TERM_NOT_FOUND_ON_PAGE"
    elif "local_html" in t:
        html_path = os.path.join(base_dir, t["local_html"])
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            htxt = f.read()
        found = False
        for term in t["terms"]:
            if term.lower() in htxt.lower():
                m = re.search(r'([^.\n<]*?' + re.escape(term) + r'[^.\n<]*)', htxt, re.IGNORECASE)
                if m:
                    words = m.group(0).split()
                    snippet = " ".join(words[:min(10, len(words))])
                    found = True
                    break
        if not found:
            verdict = "FAIL"
            snippet = "TERM_NOT_FOUND_IN_HTML"
            
    audit_rows.append({
        "key": key,
        "url": url if anchor == "N/A" else f"{url}#page={anchor}",
        "status": status,
        "content_type": ctype,
        "page": str(anchor),
        "snippet": snippet,
        "verdict": verdict
    })
    print(f"[{verdict}] {key}: status={status}, ctype={ctype}, page={anchor}, snippet='{snippet}'")

out_csv = os.path.join(base_dir, "link_audit.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["key", "url", "status", "content_type", "page", "snippet", "verdict"])
    writer.writeheader()
    for r in audit_rows:
        writer.writerow(r)

print(f"\nWritten {len(audit_rows)} rows to {out_csv}")
