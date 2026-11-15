import pypdf

reader = pypdf.PdfReader("c:/Users/divya/Downloads/openfoam_pinn_pipeline/report.pdf")
print("Total pages:", len(reader.pages))

p1 = reader.pages[0].extract_text()
print("=== PAGE 1 ===")
print(p1)

print("\n=== PAGE 2 ===")
p2 = reader.pages[1].extract_text()
print(p2[:300])

