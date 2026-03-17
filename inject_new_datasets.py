"""Inject Sections 20 (Full Pavia), 21 (Indian Pines), 22 (Salinas) into the notebook.
Insert at position 49 (before current Conclusions cell).
Uses .replace() instead of f-strings to avoid {} escaping issues."""
import json, pathlib

NB_PATH = pathlib.Path(r'c:\Users\saika\OneDrive\Desktop\test 6\QHSA_Net_Research_Notebook_2.ipynb')
INSERT_AT = 49  # before Conclusions

nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
print(f'Cells before: {len(nb["cells"])}')

def md(source): return {'cell_type': 'markdown', 'metadata': {}, 'source': source.strip()}
def code(source): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.strip()}

BASE = r'c:/Users/saika/OneDrive/Desktop/test 6'

IP_CLASS_NAMES = [
    'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
    'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed',
    'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean',
    'Wheat', 'Woods', 'Buildings-Grass-Trees', 'Stone-Steel-Towers'
]
SAL_CLASS_NAMES = [
    'Broccolini-Grn-Wds-1', 'Broccolini-Grn-Wds-2', 'Fallow',
    'Fallow-Rough-Plow', 'Fallow-Smooth', 'Stubble', 'Celery',
    'Grapes-Untrained', 'Soil-Vinyard-Dev', 'Corn-Senesced-Grn-Wds',
    'Lettuce-Romaine-4wk', 'Lettuce-Romaine-5wk', 'Lettuce-Romaine-6wk',
    'Lettuce-Romaine-7wk', 'Vinyard-Untrained', 'Vinyard-Vertical'
]

# =============================================================================
# SECTION 20 MD
# =============================================================================
S20_MD = """## 20. Full Pavia University Run <a id="sec20"></a>

Repeats training with **full dataset** (no QUICK_RUN subsampling) for publication-quality results.

| Setting | QUICK_RUN | Full Run |
|---------|-----------|----------|
| Train samples | 3,000 | ~4,278 (10% of 42,776) |
| Epochs | 30 | 120 |
| Seeds | 1 | 3 |

Best configuration from ablations: **original QHSA-Net (8q PCA, Config A)** — validated as top performer.
Also reports improved variant (4q TruncatedSVD) for comparison."""

# =============================================================================
# SECTION 20 CODE — use placeholder for BASE path
# =============================================================================
S20_CODE_TMPL = r"""# =============================================================================
#  SECTION 20  Full Pavia University Run (publication-quality)
# =============================================================================
import time, numpy as np, pandas as pd, torch
import scipy.io as sio
from sklearn.decomposition import PCA as SKPCA, TruncatedSVD as TSVD
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

FULL_EPOCHS = 120
FULL_SEEDS  = [42, 0, 7]
PATCH_R20   = PATCH_SIZE // 2

print('=== Section 20: Full Pavia U Run ===')
_raw20 = sio.loadmat(DATA_PATH)
_gtm20 = sio.loadmat(GT_PATH)
_hsi20 = _raw20['paviaU'].astype(np.float32)
_gt20  = _gtm20['paviaU_gt'].astype(np.int32)
H20, W20, B20 = _hsi20.shape

_hsi20_n = (_hsi20 - _hsi20.min(0)) / (_hsi20.max(0) - _hsi20.min(0) + 1e-8)
_rows20, _cols20 = np.where(_gt20 > 0)
_labeled_lbls20  = _gt20[_rows20, _cols20] - 1

np.random.seed(42)
_idx20 = np.random.permutation(len(_labeled_lbls20))
_n_tr20 = int(0.1 * len(_idx20))
_tr_idx20, _te_idx20 = _idx20[:_n_tr20], _idx20[_n_tr20:]
print(f'  Full train: {_n_tr20:,}  test: {len(_te_idx20):,}')

def _extr20(hsi_n, rows, cols, ps=PATCH_SIZE):
    ph = ps // 2
    padded = np.pad(hsi_n, ((ph,ph),(ph,ph),(0,0)), mode='reflect')
    return np.stack([padded[r:r+ps, c:c+ps, :] for r,c in zip(rows,cols)]).astype(np.float32)

print('  Extracting patches (full dataset)...')
_Xtr20 = _extr20(_hsi20_n, _rows20[_tr_idx20], _cols20[_tr_idx20])
_Xte20 = _extr20(_hsi20_n, _rows20[_te_idx20], _cols20[_te_idx20])
_ytr20 = _labeled_lbls20[_tr_idx20]
_yte20 = _labeled_lbls20[_te_idx20]
_spec_tr20 = _Xtr20[:, PATCH_R20, PATCH_R20, :].astype(np.float32)
_spec_te20 = _Xte20[:, PATCH_R20, PATCH_R20, :].astype(np.float32)

s20_results = []

def _run_full20(seed, name, dr_tr, dr_te, n_q, n_l):
    torch.manual_seed(seed); np.random.seed(seed)
    _tr_l = DataLoader(HSIDataset(_Xtr20, dr_tr, _ytr20),
                       batch_size=64, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(_Xte20, dr_te, _yte20),
                       batch_size=512, shuffle=False, num_workers=0)
    _m = QHSANet(n_bands=B20, n_cls=N_CLASSES, n_qubits=n_q, n_q_layers=n_l).to(DEVICE)
    t0 = time.time()
    train_qhsa(_m, _tr_l, FULL_EPOCHS, name=f'{name}-s{seed}')
    _tt = time.time() - t0
    _yt, _yp, _, _, _ = eval_qhsa(_m, _te_l)
    _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
    del _m
    return dict(name=name, seed=seed, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt)

# Config 1: Original QHSA-Net (8q PCA)
print('--- QHSA-Net original (8q PCA, 120 epochs) ---')
for _seed in FULL_SEEDS:
    print(f'  Seed {_seed}...')
    _pca20 = SKPCA(n_components=N_QUBITS, random_state=_seed)
    _dtr = _pca20.fit_transform(_spec_tr20).astype(np.float32)
    _dte = _pca20.transform(_spec_te20).astype(np.float32)
    _r = _run_full20(_seed, 'QHSA-8q-PCA', _dtr, _dte, N_QUBITS, N_Q_LAYERS)
    s20_results.append(_r)
    print(f'    OA={_r["OA"]:.2f}%  AA={_r["AA"]:.2f}%  kappa={_r["kappa"]:.2f}  {_r["train_time_s"]/60:.1f}min')

# Config 2: Improved variant (4q TruncatedSVD)
print('--- QHSA-Net improved (4q TruncatedSVD, 120 epochs) ---')
for _seed in FULL_SEEDS:
    print(f'  Seed {_seed}...')
    _svd20 = TSVD(n_components=4, random_state=_seed)
    _dtr = _svd20.fit_transform(_spec_tr20).astype(np.float32)
    _dte = _svd20.transform(_spec_te20).astype(np.float32)
    _r = _run_full20(_seed, 'QHSA-4q-TruncSVD', _dtr, _dte, 4, 2)
    s20_results.append(_r)
    print(f'    OA={_r["OA"]:.2f}%  AA={_r["AA"]:.2f}%  kappa={_r["kappa"]:.2f}  {_r["train_time_s"]/60:.1f}min')

df20 = pd.DataFrame(s20_results)
print('\n=== Full Pavia U Results (mean +- std, 3 seeds) ===')
for _nm, _g in df20.groupby('name'):
    print(f'  {_nm}: OA={_g["OA"].mean():.2f}+-{_g["OA"].std():.2f}  '
          f'AA={_g["AA"].mean():.2f}+-{_g["AA"].std():.2f}  '
          f'kappa={_g["kappa"].mean():.2f}+-{_g["kappa"].std():.2f}')
df20.to_csv('full_pavia_results.csv', index=False)
print('Saved: full_pavia_results.csv')"""

S20_CODE = S20_CODE_TMPL  # no placeholders needed

# =============================================================================
# SECTION 21 — Indian Pines
# =============================================================================
S21_MD = """## 21. Indian Pines Dataset <a id="sec21"></a>

Validates QHSA-Net on the second canonical HSI benchmark.

| Property | Value |
|----------|-------|
| Sensor | AVIRIS |
| Size | 145 x 145 x 200 bands |
| Classes | 16 land-cover types |
| Labeled pixels | 10,249 |

Water absorption bands already removed (corrected, 200 bands).
Uses QUICK_RUN mode for rapid validation (30 epochs, max 3,000 train samples)."""

S21_CODE_TMPL = r"""# =============================================================================
#  SECTION 21  Indian Pines Dataset
# =============================================================================
import time, numpy as np, pandas as pd, torch
import scipy.io as sio
from sklearn.decomposition import PCA as SKPCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

IP_PATH    = r'IP_BASE/indian pines data/Indian_pines_corrected.mat'
IP_GT_PATH = r'IP_BASE/indian pines data/Indian_pines_gt.mat'
IP_CLASS_NAMES = IP_NAMES_LIST
IP_N_CLS = 16

print('=== Section 21: Indian Pines ===')
_ip = sio.loadmat(IP_PATH);  _ipg = sio.loadmat(IP_GT_PATH)
ip_hsi  = _ip['indian_pines_corrected'].astype(np.float32)
ip_gt   = _ipg['indian_pines_gt'].astype(np.int32)
H_ip, W_ip, B_ip = ip_hsi.shape
print(f'  Shape: {H_ip}x{W_ip}x{B_ip}  Classes: {IP_N_CLS}  Labeled: {int((ip_gt>0).sum()):,}')

ip_hsi_n = (ip_hsi - ip_hsi.min(0)) / (ip_hsi.max(0) - ip_hsi.min(0) + 1e-8)
ip_rows, ip_cols = np.where(ip_gt > 0)
ip_lbls = ip_gt[ip_rows, ip_cols] - 1
np.random.seed(SEED)
_idx_ip = np.random.permutation(len(ip_lbls))
_ntr_ip = int(0.1 * len(_idx_ip))
ip_tr_idx, ip_te_idx = _idx_ip[:_ntr_ip], _idx_ip[_ntr_ip:]
print(f'  Train: {_ntr_ip:,}  Test: {len(ip_te_idx):,}')

ph_ip = PATCH_SIZE // 2
def _extr_ip(hsi_n, rows, cols):
    padded = np.pad(hsi_n, ((ph_ip,ph_ip),(ph_ip,ph_ip),(0,0)), mode='reflect')
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :] for r,c in zip(rows,cols)]).astype(np.float32)

print('  Extracting patches...')
ip_Xtr = _extr_ip(ip_hsi_n, ip_rows[ip_tr_idx], ip_cols[ip_tr_idx])
ip_Xte = _extr_ip(ip_hsi_n, ip_rows[ip_te_idx], ip_cols[ip_te_idx])
ip_ytr = ip_lbls[ip_tr_idx]; ip_yte = ip_lbls[ip_te_idx]

if QUICK_RUN and len(ip_ytr) > MAX_TRAIN:
    _qi = np.random.permutation(len(ip_ytr))[:MAX_TRAIN]
    ip_Xtr, ip_ytr = ip_Xtr[_qi], ip_ytr[_qi]
    print(f'  QUICK_RUN: train subsampled to {len(ip_ytr):,}')

ip_spec_tr = ip_Xtr[:, ph_ip, ph_ip, :].astype(np.float32)
ip_spec_te = ip_Xte[:, ph_ip, ph_ip, :].astype(np.float32)
ip_pca = SKPCA(n_components=N_QUBITS, random_state=SEED)
ip_pca_tr = ip_pca.fit_transform(ip_spec_tr).astype(np.float32)
ip_pca_te = ip_pca.transform(ip_spec_te).astype(np.float32)
print(f'  PCA variance retained: {ip_pca.explained_variance_ratio_.sum()*100:.1f}%')

# SVM baseline
print('  SVM...')
t0 = time.time()
_svm_ip = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
_svm_ip.fit(ip_pca_tr, ip_ytr)
_ip_svm_pred = _svm_ip.predict(ip_pca_te)
ip_svm_oa, ip_svm_aa, ip_svm_kap, _, _ = compute_metrics(ip_yte, _ip_svm_pred)
print(f'    OA={ip_svm_oa:.2f}%  AA={ip_svm_aa:.2f}%  kappa={ip_svm_kap:.2f}  time={time.time()-t0:.1f}s')

# QHSA-Net
print('  QHSA-Net...')
ip_tr_dl = DataLoader(HSIDataset(ip_Xtr, ip_pca_tr, ip_ytr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
ip_te_dl = DataLoader(HSIDataset(ip_Xte, ip_pca_te, ip_yte), batch_size=256, shuffle=False, num_workers=0)
ip_qhsa  = QHSANet(n_bands=B_ip, n_cls=IP_N_CLS, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS).to(DEVICE)
t0 = time.time()
train_qhsa(ip_qhsa, ip_tr_dl, N_EPOCHS, name='QHSA-IP')
ip_train_t = time.time() - t0
ip_yt, ip_yp, _, _, _ = eval_qhsa(ip_qhsa, ip_te_dl)
ip_oa, ip_aa, ip_kap, ip_pc, ip_cm = compute_metrics(ip_yt, ip_yp)
print(f'  QHSA-Net: OA={ip_oa:.2f}%  AA={ip_aa:.2f}%  kappa={ip_kap:.2f}  time={ip_train_t/60:.1f}min')
print_metrics(ip_yt, ip_yp)

df21 = pd.DataFrame([
    dict(method='SVM (PCA-8)', OA=ip_svm_oa, AA=ip_svm_aa, kappa=ip_svm_kap),
    dict(method='QHSA-Net',    OA=ip_oa,     AA=ip_aa,     kappa=ip_kap),
])
print(df21.to_string(index=False))

# Plot
fig21, axes21 = plt.subplots(1, 2, figsize=(14, 5))
axes21[0].bar(range(IP_N_CLS), ip_pc, color='#3498db', alpha=0.85)
axes21[0].set_xticks(range(IP_N_CLS))
axes21[0].set_xticklabels([n[:12] for n in IP_CLASS_NAMES], rotation=45, ha='right', fontsize=8)
axes21[0].set_ylabel('Per-class Accuracy (%)')
axes21[0].set_title('Indian Pines: QHSA-Net Per-class Accuracy', fontweight='bold')
axes21[0].axhline(ip_oa, color='red', ls='--', label='OA='+str(round(ip_oa,1))+'%')
axes21[0].legend()
_ms21 = ['SVM', 'QHSA-Net']; _oas21 = [ip_svm_oa, ip_oa]; _aas21 = [ip_svm_aa, ip_aa]
_x21 = np.arange(len(_ms21)); _w21 = 0.35
axes21[1].bar(_x21,      _oas21, _w21, label='OA %', color='#3498db', alpha=0.85)
axes21[1].bar(_x21+_w21, _aas21, _w21, label='AA %', color='#e67e22', alpha=0.85)
axes21[1].set_xticks(_x21+_w21/2); axes21[1].set_xticklabels(_ms21)
axes21[1].set_ylabel('Accuracy (%)'); axes21[1].set_ylim(0, 105)
axes21[1].set_title('Indian Pines: OA & AA Comparison', fontweight='bold')
axes21[1].legend()
plt.tight_layout()
plt.savefig('fig_s21_indian_pines.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s21_indian_pines.png')"""

S21_CODE = S21_CODE_TMPL.replace('IP_BASE', BASE).replace('IP_NAMES_LIST', repr(IP_CLASS_NAMES))

# =============================================================================
# SECTION 22 — Salinas
# =============================================================================
S22_MD = """## 22. Salinas Dataset <a id="sec22"></a>

Third canonical HSI benchmark - agricultural land cover, high spatial resolution.

| Property | Value |
|----------|-------|
| Sensor | AVIRIS |
| Size | 512 x 217 x 204 bands |
| Classes | 16 agricultural types |
| Labeled pixels | 54,129 |

Uses QUICK_RUN mode (30 epochs, 3,000 train samples) for rapid validation."""

S22_CODE_TMPL = r"""# =============================================================================
#  SECTION 22  Salinas Dataset
# =============================================================================
import time, numpy as np, pandas as pd, torch
import scipy.io as sio
from sklearn.decomposition import PCA as SKPCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SAL_PATH    = r'SAL_BASE/salinas data/Salinas_corrected.mat'
SAL_GT_PATH = r'SAL_BASE/salinas data/Salinas_gt.mat'
SAL_CLASS_NAMES = SAL_NAMES_LIST
SAL_N_CLS = 16

print('=== Section 22: Salinas ===')
_sal = sio.loadmat(SAL_PATH);  _salg = sio.loadmat(SAL_GT_PATH)
sal_hsi = _sal['salinas_corrected'].astype(np.float32)
sal_gt  = _salg['salinas_gt'].astype(np.int32)
H_sal, W_sal, B_sal = sal_hsi.shape
print(f'  Shape: {H_sal}x{W_sal}x{B_sal}  Classes: {SAL_N_CLS}  Labeled: {int((sal_gt>0).sum()):,}')

sal_hsi_n = (sal_hsi - sal_hsi.min(0)) / (sal_hsi.max(0) - sal_hsi.min(0) + 1e-8)
sal_rows, sal_cols = np.where(sal_gt > 0)
sal_lbls = sal_gt[sal_rows, sal_cols] - 1
np.random.seed(SEED)
_idx_sal = np.random.permutation(len(sal_lbls))
_ntr_sal = int(0.1 * len(_idx_sal))
sal_tr_idx, sal_te_idx = _idx_sal[:_ntr_sal], _idx_sal[_ntr_sal:]
print(f'  Train: {_ntr_sal:,}  Test: {len(sal_te_idx):,}')

ph_sal = PATCH_SIZE // 2
def _extr_sal(hsi_n, rows, cols):
    padded = np.pad(hsi_n, ((ph_sal,ph_sal),(ph_sal,ph_sal),(0,0)), mode='reflect')
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :] for r,c in zip(rows,cols)]).astype(np.float32)

print('  Extracting patches...')
sal_Xtr = _extr_sal(sal_hsi_n, sal_rows[sal_tr_idx], sal_cols[sal_tr_idx])
sal_Xte = _extr_sal(sal_hsi_n, sal_rows[sal_te_idx], sal_cols[sal_te_idx])
sal_ytr = sal_lbls[sal_tr_idx]; sal_yte = sal_lbls[sal_te_idx]

if QUICK_RUN and len(sal_ytr) > MAX_TRAIN:
    _qi = np.random.permutation(len(sal_ytr))[:MAX_TRAIN]
    sal_Xtr, sal_ytr = sal_Xtr[_qi], sal_ytr[_qi]
    print(f'  QUICK_RUN: train subsampled to {len(sal_ytr):,}')

sal_spec_tr = sal_Xtr[:, ph_sal, ph_sal, :].astype(np.float32)
sal_spec_te = sal_Xte[:, ph_sal, ph_sal, :].astype(np.float32)
sal_pca = SKPCA(n_components=N_QUBITS, random_state=SEED)
sal_pca_tr = sal_pca.fit_transform(sal_spec_tr).astype(np.float32)
sal_pca_te = sal_pca.transform(sal_spec_te).astype(np.float32)
print(f'  PCA variance retained: {sal_pca.explained_variance_ratio_.sum()*100:.1f}%')

# SVM baseline
print('  SVM...')
t0 = time.time()
_svm_sal = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
_svm_sal.fit(sal_pca_tr, sal_ytr)
_sal_svm_pred = _svm_sal.predict(sal_pca_te)
sal_svm_oa, sal_svm_aa, sal_svm_kap, _, _ = compute_metrics(sal_yte, _sal_svm_pred)
print(f'    OA={sal_svm_oa:.2f}%  AA={sal_svm_aa:.2f}%  kappa={sal_svm_kap:.2f}  time={time.time()-t0:.1f}s')

# QHSA-Net
print('  QHSA-Net...')
sal_tr_dl = DataLoader(HSIDataset(sal_Xtr, sal_pca_tr, sal_ytr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
sal_te_dl = DataLoader(HSIDataset(sal_Xte, sal_pca_te, sal_yte), batch_size=256, shuffle=False, num_workers=0)
sal_qhsa  = QHSANet(n_bands=B_sal, n_cls=SAL_N_CLS, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS).to(DEVICE)
t0 = time.time()
train_qhsa(sal_qhsa, sal_tr_dl, N_EPOCHS, name='QHSA-Sal')
sal_train_t = time.time() - t0
sal_yt, sal_yp, _, _, _ = eval_qhsa(sal_qhsa, sal_te_dl)
sal_oa, sal_aa, sal_kap, sal_pc, sal_cm = compute_metrics(sal_yt, sal_yp)
print(f'  QHSA-Net: OA={sal_oa:.2f}%  AA={sal_aa:.2f}%  kappa={sal_kap:.2f}  time={sal_train_t/60:.1f}min')
print_metrics(sal_yt, sal_yp)

df22 = pd.DataFrame([
    dict(method='SVM (PCA-8)', OA=sal_svm_oa, AA=sal_svm_aa, kappa=sal_svm_kap),
    dict(method='QHSA-Net',    OA=sal_oa,     AA=sal_aa,     kappa=sal_kap),
])
print(df22.to_string(index=False))

# Cross-dataset summary
_pu_oa = results.get('QHSA-Net', {}).get('oa', float('nan'))
print('\n=== Cross-Dataset QHSA-Net Summary ===')
_cross = pd.DataFrame([
    dict(dataset='Pavia U (quick)',  bands=103, classes=9,  OA=_pu_oa,  AA=results.get('QHSA-Net',{}).get('aa',float('nan'))),
    dict(dataset='Indian Pines',     bands=200, classes=16, OA=ip_oa,   AA=ip_aa),
    dict(dataset='Salinas',          bands=204, classes=16, OA=sal_oa,  AA=sal_aa),
])
print(_cross.to_string(index=False))

# Plots
fig22, axes22 = plt.subplots(1, 2, figsize=(14, 5))
axes22[0].bar(range(SAL_N_CLS), sal_pc, color='#27ae60', alpha=0.85)
axes22[0].set_xticks(range(SAL_N_CLS))
axes22[0].set_xticklabels([n[:12] for n in SAL_CLASS_NAMES], rotation=45, ha='right', fontsize=8)
axes22[0].set_ylabel('Per-class Accuracy (%)')
axes22[0].set_title('Salinas: QHSA-Net Per-class Accuracy', fontweight='bold')
axes22[0].axhline(sal_oa, color='red', ls='--', label='OA='+str(round(sal_oa,1))+'%')
axes22[0].legend()
_datasets22 = ['Pavia U', 'Indian Pines', 'Salinas']
_oas22 = [_pu_oa, ip_oa, sal_oa]
_colors22 = ['#3498db', '#e67e22', '#27ae60']
axes22[1].bar(_datasets22, _oas22, color=_colors22, alpha=0.85)
axes22[1].set_ylabel('OA (%)'); axes22[1].set_ylim(0, 105)
axes22[1].set_title('QHSA-Net OA Across Datasets', fontweight='bold')
for _i, _v in enumerate(_oas22):
    if not np.isnan(_v):
        axes22[1].text(_i, _v+0.5, str(round(_v,1))+'%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_s22_salinas.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s22_salinas.png')"""

S22_CODE = S22_CODE_TMPL.replace('SAL_BASE', BASE).replace('SAL_NAMES_LIST', repr(SAL_CLASS_NAMES))

# Build new cells
new_cells = [
    md(S20_MD), code(S20_CODE),
    md(S21_MD), code(S21_CODE),
    md(S22_MD), code(S22_CODE),
]

nb['cells'][INSERT_AT:INSERT_AT] = new_cells
print(f'Cells after: {len(nb["cells"])}  (+{len(new_cells)} inserted at {INSERT_AT})')
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Saved notebook.')

# Quick syntax check
import ast
for i, c in enumerate(new_cells):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        try:
            ast.parse(src)
            print(f'  Syntax OK: new cell {i}')
        except SyntaxError as e:
            print(f'  SYNTAX ERROR in new cell {i}: {e}')
