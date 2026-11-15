import difflib

f1 = open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/backup/v50.tex", encoding="utf-8").readlines()
f2 = open("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.tex", encoding="utf-8").readlines()

diff = list(difflib.unified_diff(f1, f2, fromfile="backup/v50.tex", tofile="report.tex"))

out_path = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/final_diff.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(diff)

print(f"Written final_diff.txt. Total diff lines: {len(diff)}")
hunk_headers = [l for l in diff if l.startswith("@@")]
print(f"Total hunks: {len(hunk_headers)}")
for h in hunk_headers:
    print("  ", h.strip())

