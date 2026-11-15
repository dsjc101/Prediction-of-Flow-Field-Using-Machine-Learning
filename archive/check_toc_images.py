import pypdf

pdf_path = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/04_hornbeck_1973/19740004879.pdf"
reader = pypdf.PdfReader(pdf_path)

# Let's inspect page 3 to 10 for images
for pno in range(3, 12):
    page = reader.pages[pno]
    print(f"Page {pno+1}: {len(page.images)} images")
    for img_idx, img in enumerate(page.images):
        print(f"  img {img_idx}: name={img.name}, size={len(img.data)}")

