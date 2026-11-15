import pypdf
import os

pdf_path = "c:/Users/divya/Downloads/openfoam_pinn_pipeline/sources/04_hornbeck_1973/19740004879.pdf"
reader = pypdf.PdfReader(pdf_path)
print("Total pages in Hornbeck PDF:", len(reader.pages))

# Check if pytesseract or fitz or PIL is available
try:
    import fitz
    print("PyMuPDF (fitz) is available!")
except ImportError:
    print("PyMuPDF not available")

try:
    import PIL.Image
    print("PIL is available!")
except ImportError:
    print("PIL not available")

try:
    import pytesseract
    print("pytesseract is available!")
except ImportError:
    print("pytesseract not available")

