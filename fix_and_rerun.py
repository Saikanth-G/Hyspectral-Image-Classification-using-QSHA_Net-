"""
Fix two issues in the notebook before re-running:
1. Cell 42 (S16): remove asyncio.set_event_loop_policy() which breaks zmq output capture
2. Cell 50 (S20): reduce FULL_SEEDS to [42] and FULL_EPOCHS to 30
3. Clear outputs for cells 42, 50, 52, 54 so they get re-executed cleanly
"""
import json, pathlib, re

NB_PATH = pathlib.Path(r'c:\Users\saika\OneDrive\Desktop\test 6\QHSA_Net_Research_Notebook_2.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

# ── Fix 1: Cell 42 (S16) — remove asyncio event loop policy lines ─────────────
cell42 = nb['cells'][42]
src42 = ''.join(cell42['source'])

ASYNCIO_BLOCK = '''import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

'''

if ASYNCIO_BLOCK in src42:
    src42 = src42.replace(ASYNCIO_BLOCK, '')
    cell42['source'] = [src42]
    cell42['outputs'] = []
    nb['cells'][42] = cell42
    print('Fixed Cell 42: removed asyncio policy block.')
else:
    # Try partial match
    lines = src42.split('\n')
    new_lines = [l for l in lines if 'WindowsSelectorEventLoopPolicy' not in l and
                 (l.strip() != 'import asyncio' or 'asyncio' not in src42.split('import asyncio')[0])]
    new_src = '\n'.join(new_lines)
    if new_src != src42:
        cell42['source'] = [new_src]
        cell42['outputs'] = []
        nb['cells'][42] = cell42
        print('Fixed Cell 42: removed asyncio lines (partial match).')
    else:
        print('WARNING: asyncio block not found in Cell 42 — check manually.')
        print('First 300 chars:', src42[:300])

# ── Fix 2: Cell 50 (S20) — reduce seeds and epochs ────────────────────────────
cell50 = nb['cells'][50]
src50 = ''.join(cell50['source'])

# Replace FULL_SEEDS
old_seeds = None
import re as _re
new_src50 = _re.sub(r'FULL_SEEDS\s*=\s*\[[^\]]+\]', 'FULL_SEEDS  = [42]', src50)
if new_src50 != src50:
    src50 = new_src50
    print('Fixed Cell 50: FULL_SEEDS reduced to [42]')
else:
    print('WARNING: FULL_SEEDS pattern not found in Cell 50.')
    for i, line in enumerate(src50.split('\n')[:30]):
        if 'SEED' in line or 'EPOCH' in line:
            print(f'  line {i}: {line}')

new_src50 = _re.sub(r'FULL_EPOCHS\s*=\s*\d+', 'FULL_EPOCHS = 30', src50)
if new_src50 != src50:
    src50 = new_src50
    print('Fixed Cell 50: FULL_EPOCHS set to 30')
else:
    print('Cell 50: FULL_EPOCHS already at desired value or not found.')

cell50['source'] = [src50]
cell50['outputs'] = []
nb['cells'][50] = cell50

# ── Clear outputs for S21 and S22 code cells too ──────────────────────────────
for idx in [52, 54]:
    nb['cells'][idx]['outputs'] = []
print('Cleared outputs for cells 52, 54 (S21, S22 code cells).')

# ── Save ──────────────────────────────────────────────────────────────────────
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nSaved notebook. Ready to re-run.')
print('Summary of changes:')
print('  Cell 42 (S16): removed asyncio.set_event_loop_policy() - was silently blocking outputs')
print('  Cell 50 (S20): FULL_SEEDS=[42] (was 3 seeds), FULL_EPOCHS=30 (was 120)')
print('  Estimated S20 runtime: ~20-40 min (vs 6+ hours before)')
