import urllib.request, subprocess, os

pdfs = {
    "raissi": ("https://arxiv.org/pdf/1711.10561.pdf", "raissi.pdf"),
    "wang": ("https://arxiv.org/pdf/2001.04536.pdf", "wang.pdf"),
    "pirozzoli": ("https://arxiv.org/pdf/2103.13383.pdf", "pirozzoli.pdf"),
}

headers = {'User-Agent': 'Mozilla/5.0'}
for name, (url, fname) in pdfs.items():
    if not os.path.exists(fname):
        print(f"Downloading {name}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(fname, "wb") as out:
            out.write(resp.read())

print("Downloaded all.")

