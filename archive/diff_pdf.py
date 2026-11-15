import pypdf, difflib

r1 = pypdf.PdfReader("c:/Users/divya/Downloads/openfoam_pinn_pipeline/backup/v50.pdf")
r2 = pypdf.PdfReader("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.pdf")

t1 = [r1.pages[i].extract_text() for i in range(len(r1.pages))]
t2 = [r2.pages[i].extract_text() for i in range(len(r2.pages))]

lines1 = ("\n".join(t1)).splitlines(keepends=True)
lines2 = ("\n".join(t2)).splitlines(keepends=True)

diff = list(difflib.unified_diff(lines1, lines2, fromfile="v50.pdf", tofile="report.pdf", n=2))
print("Diff between v50.pdf and report.pdf:")
print("".join(diff))

