import subprocess

pdfs = ['dean1406.pdf', 'dean1803.pdf', 'dean2501.pdf']

for p in pdfs:
    print(f"=== Testing {p} ===")
    res = subprocess.run(['pdftotext', p, '-'], capture_output=True, text=True)
    txt = res.stdout
    lines = txt.splitlines()
    found = False
    for i, line in enumerate(lines):
        if '0.073' in line or 'dean' in line.lower():
            print(f"  Line {i}: {line.strip()[:100]}")
            found = True
    if not found:
        print("  None found.")

