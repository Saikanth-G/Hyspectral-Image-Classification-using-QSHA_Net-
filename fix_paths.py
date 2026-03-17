"""Fix DATA_PATH and GT_PATH in the notebook to use absolute paths."""
import json, pathlib, re

NB_PATH = pathlib.Path(r'c:\Users\saika\OneDrive\Desktop\test 6\QHSA_Net_Research_Notebook_2.ipynb')
BASE = r'c:/Users/saika/OneDrive/Desktop/test 6'

nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

fixed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    changed = False
    for line in cell['source']:
        if line.strip().startswith("DATA_PATH"):
            line = f"DATA_PATH = r'{BASE}/pavia u data/PaviaU.mat'\n"
            changed = True
        elif line.strip().startswith("GT_PATH"):
            line = f"GT_PATH   = r'{BASE}/pavia u data/PaviaU_gt.mat'\n"
            changed = True
        new_source.append(line)
    if changed:
        cell['source'] = new_source
        fixed += 1

print(f"Fixed {fixed} cell(s) with updated paths.")
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print("Saved.")

# Verify
nb2 = json.loads(NB_PATH.read_text(encoding='utf-8'))
for cell in nb2['cells']:
    src = ''.join(cell['source'])
    if 'DATA_PATH' in src and 'pavia u data' in src:
        print("Verified DATA_PATH:", [l for l in cell['source'] if 'DATA_PATH' in l or 'GT_PATH' in l][:4])
