"""
tools/verify_refs.py
Re-opens each cited source PDF in sources/, extracts the cited page using pdftotext,
and verifies that the supporting verbatim quote is present (whitespace-normalized).
Checks sources/INDEX.csv and reports summary.
"""

import os
import sys
import csv
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "sources", "INDEX.csv")

def verify():
    if not os.path.exists(INDEX_PATH):
        print(f"ERROR: {INDEX_PATH} does not exist!")
        return False

    with open(INDEX_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {os.path.relpath(INDEX_PATH, BASE_DIR)}:")
    print("=" * 80)
    
    all_pass = True
    pass_count = 0
    fail_count = 0

    for i, r in enumerate(rows, 1):
        rel_pdf = r["file"]
        pdf_path = os.path.join(BASE_DIR, rel_pdf)
        pdf_page = r["pdf_page"]
        quote = r["verbatim_quote"]
        loc = r["report_location"]

        if not os.path.exists(pdf_path):
            print(f"[{i:02d}] FAIL: PDF not found: {rel_pdf}")
            all_pass = False
            fail_count += 1
            continue

        cmd = ["pdftotext", "-f", str(pdf_page), "-l", str(pdf_page), pdf_path, "-"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[{i:02d}] FAIL: pdftotext error on {rel_pdf} page {pdf_page}")
            all_pass = False
            fail_count += 1
            continue

        txt_norm = " ".join(res.stdout.split())
        quote_norm = " ".join(quote.split())
        
        if quote_norm.lower() in txt_norm.lower():
            print(f"[{i:02d}] PASS: {rel_pdf} (p. {r['printed_page']}, PDF p. {pdf_page}) | {loc}")
            pass_count += 1
        else:
            print(f"[{i:02d}] FAIL: Quote not found in {rel_pdf} page {pdf_page}")
            print(f"       Target quote: {quote_norm}")
            print(f"       Extracted text: {txt_norm[:150]}...")
            all_pass = False
            fail_count += 1

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {pass_count} PASS, {fail_count} FAIL (Total: {len(rows)})")
    return all_pass

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
