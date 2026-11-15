import urllib.request, json, re, pypdf, os

candidates = ["2003.06496", "2105.09506", "2107.10711", "2007.14527", "2109.01050"]
results = {}

for cid in candidates:
    print(f"\n=================== Checking arXiv:{cid} ===================")
    abs_url = f"https://arxiv.org/abs/{cid}"
    pdf_url = f"https://arxiv.org/pdf/{cid}.pdf"
    
    # 1. Check abstract page
    try:
        req = urllib.request.Request(abs_url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        
        m_title = re.search(r'title mathjax.*?>(.*?)</h1>', html, re.DOTALL)
        clean_title = m_title.group(1).replace('<span class="descriptor">Title:</span>', '').strip() if m_title else "N/A"
        
        m_auth = re.search(r'class="authors".*?>(.*?)</div>', html, re.DOTALL)
        clean_auth = re.sub(r'<[^>]+>', '', m_auth.group(1)).replace("Authors:", "").strip() if m_auth else "N/A"
        
        m_abs = re.search(r'<blockquote class="abstract mathjax">.*?<span class="descriptor">Abstract:</span>(.*?)</blockquote>', html, re.DOTALL)
        clean_abs = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', m_abs.group(1))).strip() if m_abs else "N/A"
        
        print(f"Title: {clean_title}")
        print(f"Authors: {clean_auth}")
        print(f"Abstract: {clean_abs[:200]}...")
        
        # 2. Download PDF
        os.makedirs("c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/new_refs", exist_ok=True)
        pdf_path = f"c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/new_refs/{cid}.pdf"
        if not os.path.exists(pdf_path):
            print(f"Downloading {pdf_url} ...")
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(pdf_path, "wb") as f_out:
                    f_out.write(resp.read())
            print(f"Saved to {pdf_path}")
            
        reader = pypdf.PdfReader(pdf_path)
        print(f"PDF Page count: {len(reader.pages)}")
        p1 = reader.pages[0].extract_text()
        print(f"PDF Page 1 text snippet:\n{p1[:250]}...")
        
        results[cid] = {
            "title": clean_title,
            "authors": clean_auth,
            "abstract": clean_abs,
            "pdf_url": pdf_url,
            "pdf_path": pdf_path,
            "valid": True
        }
    except Exception as e:
        print(f"Error checking {cid}: {e}")
        results[cid] = {"valid": False, "error": str(e)}

with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/new_refs_meta.json", "w") as f:
    json.dump(results, f, indent=2)

