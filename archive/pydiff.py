import difflib, sys

f1 = sys.argv[1]
f2 = sys.argv[2]

with open(f1, 'r', encoding='utf-8') as a, open(f2, 'r', encoding='utf-8') as b:
    lines1 = a.readlines()
    lines2 = b.readlines()

diff = list(difflib.unified_diff(lines1, lines2, fromfile=f1, tofile=f2))
sys.stdout.writelines(diff)

