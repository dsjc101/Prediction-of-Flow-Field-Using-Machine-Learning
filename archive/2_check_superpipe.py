with open('superpipe.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re
links = re.findall(r'href=[\'"](.*?)[\'"]', html)
for l in links:
    if 'thesis' in l.lower() or 'zagarola' in l.lower() or 'mckeon' in l.lower() or '.pdf' in l.lower() or 'data' in l.lower():
        print('LINK:', l)

for line in html.splitlines():
    if '1.93' in line or '0.537' in line or 'friction' in line.lower() or 'thesis' in line.lower() or 'correction' in line.lower():
        print('LINE:', line.strip()[:140])

