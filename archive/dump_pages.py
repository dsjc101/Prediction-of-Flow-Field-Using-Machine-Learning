import pypdf
from PIL import Image
import io

pdf_path = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/04_hornbeck_1973/19740004879.pdf"
reader = pypdf.PdfReader(pdf_path)

for p in [4, 5, 6, 7, 8]:
    img = reader.pages[p-1].images[0]
    im = Image.open(io.BytesIO(img.data))
    im.save(f"c:/Users/divya/Downloads/openfoam_pinn_pipeline/page_{p}.png")
    print(f"Saved page_{p}.png, size={im.size}")

