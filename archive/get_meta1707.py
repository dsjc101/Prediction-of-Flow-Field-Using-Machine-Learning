import urllib.request
import re

url = "https://arxiv.org/abs/1707.04947"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")

m_title = re.search(r'title mathjax.*?>(.*?)</h1>', html, re.DOTALL)
if m_title:
    clean_title = m_title.group(1).replace('<span class="descriptor">Title:</span>', '').strip()
    print("ARXIV_TITLE:", clean_title)

m_auth = re.search(r'class="authors".*?>(.*?)</div>', html, re.DOTALL)
if m_auth:
    clean_auth = re.sub(r'<[^>]+>', '', m_auth.group(1)).replace("Authors:", "").strip()
    print("ARXIV_AUTHORS:", clean_auth)

