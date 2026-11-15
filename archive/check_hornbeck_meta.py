import urllib.request, json, re

url = "https://ntrs.nasa.gov/api/citations/19740004879"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode('utf-8'))
    print("API Data Title:", data.get("title"))
    print("Author:", [a.get("name") for a in data.get("authorAffiliations", [])])
    print("Publication Date:", data.get("publicationDate"))
    print("Report Number:", data.get("otherReportNumbers"))
    downloads = data.get("downloads", [])
    for d in downloads:
        print("Download link:", d.get("links", {}).get("pdf"))
except Exception as e:
    print("Error fetching API:", e)

# Also inspect local file
with open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/04_hornbeck_1973/download_log.txt") as f:
    print("\nDownload log:\n", f.read())

