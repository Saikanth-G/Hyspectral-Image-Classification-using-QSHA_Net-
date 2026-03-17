#!/usr/bin/env python3
"""
inject_experiments.py
Inserts Sections 13-19 into QHSA_Net_Research_Notebook_2.ipynb at position 35.
"""
import json, pathlib

NB_PATH   = pathlib.Path(r'c:/Users/saika/OneDrive/Desktop/test 6/QHSA_Net_Research_Notebook_2.ipynb')
INSERT_AT = 35

def md(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.strip()}

def code(source):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': source.strip()}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — Timing Analysis
# ─────────────────────────────────────────────────────────────────────────────
S13_MD = """## 13. Computational Timing Analysis <a id="sec13"></a>

Wall-clock timings for every major pipeline stage. Quantifies quantum simulation
overhead vs. classical baselines. Results: `timing_summary.csv`, `fig_s13_timings.png`."""

S13_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 13  Computational Timing Analysis               ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA as SKPCA
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

experiment_records = []   # master accumulator for all ablation sections
timings = {}
_ph = PATCH_SIZE // 2    # = 4, centre index of 9x9 patch

# 1) Data normalisation
t0 = time.time()
_flat = hsi.reshape(-1, B).astype(np.float32)
_mn, _mx = _flat.min(0), _flat.max(0)
_ = (_flat - _mn) / (_mx - _mn + 1e-8)
timings['1_data_normalisation'] = time.time() - t0

# 2) PCA fit on labeled pixels
t0 = time.time()
_pca_t = SKPCA(n_components=N_PCA_COMP, random_state=SEED)
_lm = gt.reshape(-1) > 0
_pca_t.fit(_flat[_lm])
timings['2_pca_fit'] = time.time() - t0

# 3) PCA transform (full image)
t0 = time.time()
_ = _pca_t.transform(_flat)
timings['3_pca_transform_full'] = time.time() - t0

# 4) SVM fit
t0 = time.time()
_svm_t = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
_svm_t.fit(Xpca_tr, y_tr)
timings['4_svm_fit'] = time.time() - t0

# 5) SVM inference
t0 = time.time()
_ = _svm_t.predict(Xpca_te)
timings['5_svm_inference'] = time.time() - t0
del _svm_t

# 6) QHSA-Net: 2-epoch proxy for per-epoch estimate
_q_tmp = QHSANet(n_bands=B, n_cls=N_CLASSES,
                 n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS).to(DEVICE)
_opt_t = torch.optim.Adam(_q_tmp.parameters(), lr=1e-3)
_crit_t = torch.nn.CrossEntropyLoss()
t0 = time.time()
for _ep in range(2):
    _q_tmp.train()
    for _pb, _qb, _lb in train_loader:
        _pb, _qb, _lb = _pb.to(DEVICE), _qb.to(DEVICE), _lb.to(DEVICE)
        _opt_t.zero_grad()
        _crit_t(_q_tmp(_pb, _qb), _lb).backward()
        _opt_t.step()
_t2ep = time.time() - t0
timings['6_qhsa_per_epoch_est'] = _t2ep / 2
timings['7_qhsa_full_train_est'] = timings['6_qhsa_per_epoch_est'] * N_EPOCHS

# 7) QHSA-Net inference
_q_tmp.eval()
t0 = time.time()
with torch.no_grad():
    for _pb, _qb, _lb in test_loader:
        _ = _q_tmp(_pb.to(DEVICE), _qb.to(DEVICE))
timings['8_qhsa_inference'] = time.time() - t0
del _q_tmp, _opt_t, _crit_t

# Print & save
print('=== Section 13: Timing Summary ===')
for k, v in timings.items():
    print(f'  {k:<42s}: {v:8.3f} s')

df_tim = pd.DataFrame(list(timings.items()), columns=['Stage', 'Time_s'])
df_tim.to_csv('timing_summary.csv', index=False)

_cols = ['#3498db' if 'qhsa' in k else '#e67e22' if 'svm' in k else '#2ecc71'
         for k in timings]
fig_t, ax_t = plt.subplots(figsize=(11, 6))
_bars = ax_t.barh(list(timings.keys()), list(timings.values()), color=_cols)
ax_t.set_xscale('log')
ax_t.set_xlabel('Wall-clock Time (s, log scale)')
ax_t.set_title('Section 13: Pipeline Stage Timing (QUICK_RUN mode)', fontweight='bold')
for _bar, _val in zip(_bars, timings.values()):
    ax_t.text(_val * 1.12, _bar.get_y() + _bar.get_height() / 2,
              f'{_val:.3f}s', va='center', fontsize=8.5)
from matplotlib.patches import Patch as _Patch
ax_t.legend(handles=[_Patch(color='#2ecc71', label='Preprocessing'),
                     _Patch(color='#e67e22', label='SVM'),
                     _Patch(color='#3498db', label='QHSA-Net')], loc='lower right')
plt.tight_layout()
plt.savefig('fig_s13_timings.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: timing_summary.csv  fig_s13_timings.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Dimension Reduction Comparison
# ─────────────────────────────────────────────────────────────────────────────
S14_MD = """## 14. Dimension Reduction Technique Comparison <a id="sec14"></a>

Systematically compares 9 DR methods (103 → 8 features) for the quantum encoding step.
Each method feeds the quantum branch; the classical 3D-CNN still receives full patches.

| Method | Category | Notes |
|--------|----------|-------|
| PCA | Linear | Baseline |
| Kernel-PCA (RBF) | Non-linear | Captures nonlinear spectral manifold |
| FastICA | Statistical | Maximises statistical independence |
| Factor Analysis | Probabilistic | Models latent Gaussian factors |
| Truncated SVD | Linear | LSA; mean-free PCA equivalent |
| Random Projection | Random | Johnson-Lindenstrauss; near-isometric |
| NMF | Parts-based | Requires non-negative input |
| UMAP | Manifold | Topology-preserving (if installed) |
| Learned Autoencoder | Deep | 103→32→8 MLP, MSE-trained |

**Hypothesis:** Kernel/manifold methods may capture nonlinear spectral structure better
than linear PCA for quantum angle encoding."""

S14_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 14  Dimension Reduction Comparison              ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import (PCA as SKPCA, KernelPCA, FastICA,
                                   FactorAnalysis, TruncatedSVD, NMF)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

EXP_EPOCHS_DR = max(8, N_EPOCHS // 4)   # short but comparative
K = N_PCA_COMP                           # = 8 features / qubits
_ph = PATCH_SIZE // 2                    # = 4

# Center-pixel spectra from patches (shape [N, 103])
spec_tr = X_tr[:, _ph, _ph, :].astype(np.float32)
spec_te = X_te[:, _ph, _ph, :].astype(np.float32)

# Min-max scaled version for NMF
_mm = MinMaxScaler()
spec_tr_mm = _mm.fit_transform(spec_tr)
spec_te_mm = _mm.transform(spec_te)

def run_dr_variant(name, dr_tr, dr_te, n_q=8):
    _tr_l = DataLoader(HSIDataset(X_tr, dr_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(X_te, dr_te, y_te),
                       batch_size=256, shuffle=False, num_workers=0)
    _model = QHSANet(n_bands=B, n_cls=N_CLASSES,
                     n_qubits=n_q, n_q_layers=N_Q_LAYERS).to(DEVICE)
    t0 = time.time()
    train_qhsa(_model, _tr_l, EXP_EPOCHS_DR, name=name)
    train_t = time.time() - t0
    t1 = time.time()
    _yt, _yp, _, _, _ = eval_qhsa(_model, _te_l)
    inf_t = time.time() - t1
    _oa, _aa, _kap, _pc, _ = compute_metrics(_yt, _yp)
    del _model
    return dict(method=name, OA=_oa, AA=_aa, kappa=_kap,
                train_time_s=train_t, inference_time_s=inf_t, n_qubits=n_q,
                n_epochs=EXP_EPOCHS_DR)

s14_results = []

# ── DR methods ────────────────────────────────────────────────────────────────
dr_configs = []

# 1. PCA (baseline)
t0 = time.time()
_pca14 = SKPCA(n_components=K, random_state=SEED)
_d_tr = _pca14.fit_transform(spec_tr).astype(np.float32)
_d_te = _pca14.transform(spec_te).astype(np.float32)
dr_configs.append(('PCA', _d_tr, _d_te, time.time()-t0))

# 2. Kernel-PCA (RBF)
try:
    t0 = time.time()
    _kpca = KernelPCA(n_components=K, kernel='rbf', random_state=SEED)
    _d_tr = _kpca.fit_transform(spec_tr).astype(np.float32)
    _d_te = _kpca.transform(spec_te).astype(np.float32)
    dr_configs.append(('Kernel-PCA', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'KernelPCA failed: {_e}')

# 3. FastICA
try:
    t0 = time.time()
    _ica = FastICA(n_components=K, random_state=SEED, max_iter=500, whiten='unit-variance')
    _d_tr = _ica.fit_transform(spec_tr).astype(np.float32)
    _d_te = _ica.transform(spec_te).astype(np.float32)
    dr_configs.append(('FastICA', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'FastICA failed: {_e}')

# 4. Factor Analysis
try:
    t0 = time.time()
    _fa = FactorAnalysis(n_components=K, random_state=SEED)
    _d_tr = _fa.fit_transform(spec_tr).astype(np.float32)
    _d_te = _fa.transform(spec_te).astype(np.float32)
    dr_configs.append(('FactorAnalysis', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'FactorAnalysis failed: {_e}')

# 5. Truncated SVD
try:
    t0 = time.time()
    _svd = TruncatedSVD(n_components=K, random_state=SEED)
    _d_tr = _svd.fit_transform(spec_tr).astype(np.float32)
    _d_te = _svd.transform(spec_te).astype(np.float32)
    dr_configs.append(('TruncatedSVD', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'TruncatedSVD failed: {_e}')

# 6. Gaussian Random Projection
try:
    t0 = time.time()
    _rp = GaussianRandomProjection(n_components=K, random_state=SEED)
    _d_tr = _rp.fit_transform(spec_tr).astype(np.float32)
    _d_te = _rp.transform(spec_te).astype(np.float32)
    dr_configs.append(('RandProjection', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'RandomProjection failed: {_e}')

# 7. NMF (non-negative input required)
try:
    t0 = time.time()
    _nmf = NMF(n_components=K, random_state=SEED, max_iter=500)
    _d_tr = _nmf.fit_transform(spec_tr_mm).astype(np.float32)
    _d_te = _nmf.transform(spec_te_mm).astype(np.float32)
    dr_configs.append(('NMF', _d_tr, _d_te, time.time()-t0))
except Exception as _e:
    print(f'NMF failed: {_e}')

# 8. UMAP (optional)
try:
    import umap as _umap_mod
    t0 = time.time()
    _um = _umap_mod.UMAP(n_components=K, random_state=SEED, n_jobs=1)
    _d_tr = _um.fit_transform(spec_tr).astype(np.float32)
    _d_te = _um.transform(spec_te).astype(np.float32)
    dr_configs.append(('UMAP', _d_tr, _d_te, time.time()-t0))
except ImportError:
    print('umap-learn not installed; skipping UMAP  (pip install umap-learn)')
except Exception as _e:
    print(f'UMAP failed: {_e}')

# 9. Learned Autoencoder (103->32->8->32->103, MSE-trained)
try:
    import torch.nn as _nn
    class _SpectralAE(_nn.Module):
        def __init__(self, in_d=103, z_d=8):
            super().__init__()
            self.enc = _nn.Sequential(_nn.Linear(in_d,32), _nn.ReLU(), _nn.Linear(32,z_d))
            self.dec = _nn.Sequential(_nn.Linear(z_d,32), _nn.ReLU(), _nn.Linear(32,in_d))
        def forward(self, x): return self.dec(self.enc(x))
        def encode(self, x): return self.enc(x)
    _ae = _SpectralAE(B, K).to(DEVICE)
    _ae_opt = torch.optim.Adam(_ae.parameters(), lr=1e-3)
    _ae_crit = torch.nn.MSELoss()
    _spec_t = torch.tensor(spec_tr, dtype=torch.float32).to(DEVICE)
    t0 = time.time()
    for _ep in range(80):
        _ae_opt.zero_grad()
        _ae_crit(_ae(_spec_t), _spec_t).backward()
        _ae_opt.step()
    _ae_t = time.time() - t0
    _ae.eval()
    with torch.no_grad():
        _d_tr = _ae.encode(_spec_t).cpu().numpy().astype(np.float32)
        _d_te = _ae.encode(torch.tensor(spec_te, dtype=torch.float32).to(DEVICE)).cpu().numpy().astype(np.float32)
    dr_configs.append(('AutoEncoder', _d_tr, _d_te, _ae_t))
    del _ae, _ae_opt, _ae_crit, _spec_t
    print(f'AutoEncoder trained in {_ae_t:.1f}s')
except Exception as _e:
    print(f'AutoEncoder failed: {_e}')

# ── Train QHSA-Net variant for each DR method ─────────────────────────────────
print(f'Running {len(dr_configs)} DR methods x {EXP_EPOCHS_DR} epochs each...')
for _name, _dtr, _dte, _fit_t in dr_configs:
    print(f'  [{_name}]')
    try:
        _r = run_dr_variant(_name, _dtr, _dte)
        _r['dr_fit_time_s'] = _fit_t
        s14_results.append(_r)
        experiment_records.append({**_r, 'section': 'S14_DR'})
        print(f'    OA={_r["OA"]:.2f}%  AA={_r["AA"]:.2f}%  kappa={_r["kappa"]:.2f}  '
              f'train={_r["train_time_s"]:.0f}s')
    except Exception as _ex:
        print(f'    FAILED: {_ex}')

df14 = pd.DataFrame(s14_results)
best_dr_method = df14.loc[df14['OA'].idxmax(), 'method'] if len(df14) > 0 else 'PCA'
print(f'Best DR method (OA): {best_dr_method}')
print(df14[['method','OA','AA','kappa','train_time_s','dr_fit_time_s']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig14, axes14 = plt.subplots(1, 2, figsize=(15, 6))
_methods = df14['method'].tolist()
_x = np.arange(len(_methods))
_w = 0.27
for _i, (_col, _lbl) in enumerate([('OA','OA %'),('AA','AA %'),('kappa','Kappa')]):
    if _col in df14.columns:
        axes14[0].bar(_x + _i*_w, df14[_col], _w, label=_lbl, alpha=0.85)
axes14[0].set_xticks(_x + _w)
axes14[0].set_xticklabels(_methods, rotation=30, ha='right', fontsize=9)
axes14[0].set_ylabel('Score (%)')
axes14[0].set_title('Section 14: DR Comparison (QHSA-Net, 8 qubits)', fontweight='bold')
axes14[0].legend()
axes14[0].set_ylim(0, 105)

if 'train_time_s' in df14.columns:
    axes14[1].bar(_methods, df14['train_time_s'], color='coral', alpha=0.85)
    for _tick in axes14[1].get_xticklabels():
        _tick.set_rotation(30); _tick.set_ha('right'); _tick.set_fontsize(9)
    axes14[1].set_ylabel('Training Time (s)')
    axes14[1].set_title('Training Time per DR Method', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_s14_dr_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s14_dr_comparison.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — Qubit Count & PCA Component Sweep
# ─────────────────────────────────────────────────────────────────────────────
S15_MD = """## 15. Qubit Count & PCA Components Joint Sweep <a id="sec15"></a>

Since angle encoding maps one feature per qubit, the number of qubits equals the
number of PCA components. This sweep jointly varies both to find the optimal operating point.

| n_qubits | Hilbert dim | VQC params (2 layers) | PCA variance |
|----------|-------------|----------------------|--------------|
| 2 | 4 | 12 | low |
| 4 | 16 | 24 | medium |
| 6 | 64 | 36 | good |
| **8** | **256** | **48** | **high (current)** |
| 10 | 1024 | 60 | very high |
| 12 | 4096 | 72 | ≈full |

**Barren plateau risk** increases with n_qubits; literature suggests 4-8 qubits with
2-4 layers is the practical NISQ sweet spot (Cerezo et al. 2021)."""

S15_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 15  Qubit Count & PCA Component Sweep           ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

QUBIT_SWEEP  = [2, 4, 6, 8, 10, 12]
EXP_EPOCHS_Q = max(8, N_EPOCHS // 4)
_ph = PATCH_SIZE // 2

spec_tr_all = X_tr[:, _ph, _ph, :].astype(np.float32)
spec_te_all = X_te[:, _ph, _ph, :].astype(np.float32)

s15_results = []
for _nq in QUBIT_SWEEP:
    print(f'  [S15] n_qubits={_nq}  (Hilbert dim=2^{_nq}={2**_nq})')
    try:
        _pca_q = SKPCA(n_components=_nq, random_state=SEED)
        _dtr_q = _pca_q.fit_transform(spec_tr_all).astype(np.float32)
        _dte_q = _pca_q.transform(spec_te_all).astype(np.float32)
        _var_q = float(np.sum(_pca_q.explained_variance_ratio_) * 100)

        _tr_l = DataLoader(HSIDataset(X_tr, _dtr_q, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        _te_l = DataLoader(HSIDataset(X_te, _dte_q, y_te),
                           batch_size=256, shuffle=False, num_workers=0)

        _model_q = QHSANet(n_bands=B, n_cls=N_CLASSES,
                           n_qubits=_nq, n_q_layers=N_Q_LAYERS).to(DEVICE)
        t0 = time.time()
        train_qhsa(_model_q, _tr_l, EXP_EPOCHS_Q, name=f'QHSA-{_nq}q')
        _tt = time.time() - t0
        _yt, _yp, _, _, _ = eval_qhsa(_model_q, _te_l)
        _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
        _n_params_q = N_Q_LAYERS * _nq * 3
        _r = dict(n_qubits=_nq, hilbert_dim=2**_nq, pca_variance_pct=_var_q,
                  vqc_params=_n_params_q, OA=_oa, AA=_aa, kappa=_kap,
                  train_time_s=_tt, n_layers=N_Q_LAYERS)
        s15_results.append(_r)
        experiment_records.append({**_r, 'method': f'QHSA-{_nq}q', 'section': 'S15_QubitSweep'})
        print(f'    OA={_oa:.2f}%  var={_var_q:.1f}%  time={_tt:.0f}s  params={_n_params_q}')
        del _model_q
    except Exception as _ex:
        print(f'    FAILED n_qubits={_nq}: {_ex}')

df15 = pd.DataFrame(s15_results)

# Secondary layer sweep for best n_qubits
best_nq = int(df15.loc[df15['OA'].idxmax(), 'n_qubits']) if len(df15) > 0 else N_QUBITS
print(f'Best n_qubits by OA: {best_nq}  -- running layer sweep [1,2,3]')
_pca_best = SKPCA(n_components=best_nq, random_state=SEED)
_dtr_b = _pca_best.fit_transform(spec_tr_all).astype(np.float32)
_dte_b = _pca_best.transform(spec_te_all).astype(np.float32)
_tr_lb = DataLoader(HSIDataset(X_tr, _dtr_b, y_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
_te_lb = DataLoader(HSIDataset(X_te, _dte_b, y_te), batch_size=256, shuffle=False, num_workers=0)

s15_layer = []
for _nl in [1, 2, 3]:
    try:
        _ml = QHSANet(n_bands=B, n_cls=N_CLASSES, n_qubits=best_nq, n_q_layers=_nl).to(DEVICE)
        t0 = time.time()
        train_qhsa(_ml, _tr_lb, EXP_EPOCHS_Q, name=f'QHSA-{best_nq}q-{_nl}L')
        _tt = time.time() - t0
        _yt, _yp, _, _, _ = eval_qhsa(_ml, _te_lb)
        _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
        _r2 = dict(n_qubits=best_nq, n_layers=_nl, OA=_oa, AA=_aa, kappa=_kap,
                   train_time_s=_tt, vqc_params=_nl*best_nq*3)
        s15_layer.append(_r2)
        experiment_records.append({**_r2, 'method': f'QHSA-{best_nq}q-{_nl}L', 'section': 'S15_LayerSweep'})
        print(f'  n_layers={_nl}: OA={_oa:.2f}%  params={_nl*best_nq*3}  time={_tt:.0f}s')
        del _ml
    except Exception as _ex:
        print(f'  Layer sweep n_layers={_nl} failed: {_ex}')

df15L = pd.DataFrame(s15_layer)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig15, axes15 = plt.subplots(1, 3, figsize=(17, 5))

if len(df15) > 0:
    ax = axes15[0]
    ax.plot(df15['n_qubits'], df15['OA'], 'o-', color='#3498db', lw=2, ms=8, label='OA %')
    ax.plot(df15['n_qubits'], df15['AA'], 's--', color='#e67e22', lw=1.5, ms=7, label='AA %')
    ax.axvline(best_nq, color='red', ls=':', alpha=0.6, label=f'Best n={best_nq}')
    ax.set_xlabel('n_qubits (= n_pca_components)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Qubit Count Sweep: Accuracy', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax2 = axes15[1]
    ax2.bar(df15['n_qubits'].astype(str), df15['train_time_s'], color='#9b59b6', alpha=0.8)
    ax2.set_xlabel('n_qubits')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs Qubit Count', fontweight='bold')
    for _tick in ax2.get_xticklabels(): _tick.set_fontsize(9)

if len(df15L) > 0:
    ax3 = axes15[2]
    ax3.bar(df15L['n_layers'].astype(str), df15L['OA'], color='#1abc9c', alpha=0.85)
    ax3.set_xlabel(f'n_layers (n_qubits={best_nq})')
    ax3.set_ylabel('OA (%)')
    ax3.set_title(f'Layer Depth Sweep (best {best_nq} qubits)', fontweight='bold')

plt.suptitle('Section 15: Qubit & Layer Sweep', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_s15_qubit_sweep.png', dpi=150, bbox_inches='tight')
plt.show()
print(df15[['n_qubits','pca_variance_pct','hilbert_dim','vqc_params','OA','AA','kappa']].to_string(index=False))
print(f'Saved: fig_s15_qubit_sweep.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — DR Placement in Architecture
# ─────────────────────────────────────────────────────────────────────────────
S16_MD = """## 16. DR Placement in Architecture Experiments <a id="sec16"></a>

Tests four configurations for *where* dimension reduction is applied:

| Config | Classical branch input | Quantum branch input |
|--------|----------------------|---------------------|
| **A (current)** | Full 103-band patches (3D-CNN) | 8 PCA features |
| **B (joint PCA)** | 8 PCA features (MLP) | 8 PCA features |
| **C (reversed)** | 8 PCA features (MLP) | 103 bands → learned Linear(103,8) → VQC |
| **D (shared encoder)** | Shared Linear(103→8) | Same shared encoding |

**Key question:** Is the classical 3D-CNN valuable *because* it sees all 103 spectral bands?"""

S16_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 16  DR Placement in Architecture                ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

EXP_EPOCHS_ARCH = max(8, N_EPOCHS // 4)

# ── Helper MLP classical branch (for configs that don't use 3D-CNN) ───────────
class ClassicalMLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, out_dim), nn.LayerNorm(out_dim)
        )
    def forward(self, x): return self.net(x)

# ── Config B: Both branches get 8 PCA features ────────────────────────────────
class QHSANet_ConfigB(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.classical = ClassicalMLP(n_qubits, 64)
        self.quantum   = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion    = GatedFusion(dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        fc = self.classical(pca_x)   # ignores patch_x
        fq = self.quantum(pca_x)
        return self.classifier(self.fusion(fc, fq))

# ── Config C: Classical=MLP(8 PCA), Quantum=VQC(103 bands → learned proj) ────
class QHSANet_ConfigC(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.proj_in   = nn.Linear(B, n_qubits)   # 103 -> 8 (learned)
        self.classical = ClassicalMLP(n_qubits, 64)
        self.quantum   = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion    = GatedFusion(dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        fc  = self.classical(pca_x)               # classical: PCA features -> MLP
        cen = patch_x[:, :, 4, 4]                 # [B, 103] centre pixel
        q_in = self.proj_in(cen)                  # [B, n_qubits] learned projection
        fq  = self.quantum(q_in)                  # quantum: raw bands -> VQC
        return self.classifier(self.fusion(fc, fq))

# ── Config D: Shared Linear(103->8) encoder, both branches share encoding ─────
class QHSANet_ConfigD(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.shared    = nn.Linear(B, n_qubits)   # 103 -> 8 shared
        self.classical = ClassicalMLP(n_qubits, 64)
        self.quantum   = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion    = GatedFusion(dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        cen    = patch_x[:, :, 4, 4]              # [B, 103]
        shared = self.shared(cen)                 # [B, n_qubits]
        fc = self.classical(shared)
        fq = self.quantum(shared)
        return self.classifier(self.fusion(fc, fq))

# ── Run each config ────────────────────────────────────────────────────────────
s16_results = []

def _run_arch(name, model_instance):
    _tr_l = DataLoader(HSIDataset(X_tr, Xpca_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(X_te, Xpca_te, y_te),
                       batch_size=256, shuffle=False, num_workers=0)
    t0 = time.time()
    train_qhsa(model_instance, _tr_l, EXP_EPOCHS_ARCH, name=name)
    _tt = time.time() - t0
    _yt, _yp, _, _, _ = eval_qhsa(model_instance, _te_l)
    _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
    return dict(config=name, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt)

# Config A: re-evaluate existing trained model (no retraining needed)
print('[S16] Config A: evaluating pre-trained QHSA-Net...')
_te_l_a = DataLoader(HSIDataset(X_te, Xpca_te, y_te), batch_size=256, shuffle=False, num_workers=0)
_yt_a, _yp_a, _, _, _ = eval_qhsa(qhsa, _te_l_a)
_oa_a, _aa_a, _kap_a, _, _ = compute_metrics(_yt_a, _yp_a)
_r_a = dict(config='A: 3D-CNN+PCA (current)', OA=_oa_a, AA=_aa_a, kappa=_kap_a, train_time_s=0.0)
s16_results.append(_r_a)
experiment_records.append({**_r_a, 'method': 'Config-A', 'section': 'S16_ArchPlacement'})
print(f'  OA={_oa_a:.2f}%')

for _cfg_name, _cfg_cls in [
    ('B: MLP+PCA (joint)', QHSANet_ConfigB),
    ('C: MLP-PCA+VQC-Raw (reversed)', QHSANet_ConfigC),
    ('D: Shared-Enc (joint)', QHSANet_ConfigD),
]:
    print(f'[S16] Training {_cfg_name}...')
    try:
        _m = _cfg_cls().to(DEVICE)
        _r = _run_arch(_cfg_name, _m)
        s16_results.append(_r)
        experiment_records.append({**_r, 'method': _cfg_name, 'section': 'S16_ArchPlacement'})
        print(f'  OA={_r["OA"]:.2f}%  AA={_r["AA"]:.2f}%  kappa={_r["kappa"]:.2f}  time={_r["train_time_s"]:.0f}s')
        del _m
    except Exception as _ex:
        print(f'  {_cfg_name} FAILED: {_ex}')

df16 = pd.DataFrame(s16_results)
print(df16[['config','OA','AA','kappa','train_time_s']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig16, ax16 = plt.subplots(figsize=(12, 5))
_x16 = np.arange(len(df16))
_w16 = 0.28
for _i, (_col, _lbl) in enumerate([('OA','OA %'), ('AA','AA %'), ('kappa','Kappa')]):
    if _col in df16.columns:
        ax16.bar(_x16 + _i*_w16, df16[_col], _w16, label=_lbl, alpha=0.85)
ax16.set_xticks(_x16 + _w16)
ax16.set_xticklabels(df16['config'], rotation=20, ha='right', fontsize=9)
ax16.set_ylabel('Score (%)')
ax16.set_title('Section 16: DR Placement in Architecture', fontweight='bold')
ax16.legend(); ax16.set_ylim(0, 105); ax16.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('fig_s16_arch_placement.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s16_arch_placement.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — Feature Selection Experiments
# ─────────────────────────────────────────────────────────────────────────────
S17_MD = """## 17. Spectral Band Selection Experiments <a id="sec17"></a>

Compares band *selection* (keeps original wavelengths, interpretable) against band
*extraction* (PCA baseline). Four selection strategies, each choosing 8 bands from 103.

| Method | Strategy |
|--------|----------|
| Variance | Top-8 highest per-band variance |
| ANOVA F-score | SelectKBest with f_classif |
| Mutual Information | SelectKBest with mutual_info_classif |
| DivMin Greedy | Maximises discriminability while minimising inter-band correlation |

Selected bands feed only the quantum branch; classical 3D-CNN still gets full patches."""

S17_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 17  Feature Selection Experiments               ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

EXP_EPOCHS_FS = max(8, N_EPOCHS // 4)
K_BANDS = N_PCA_COMP    # = 8
_ph = PATCH_SIZE // 2

spec_tr_fs = X_tr[:, _ph, _ph, :].astype(np.float32)
spec_te_fs = X_te[:, _ph, _ph, :].astype(np.float32)

# ── Band selection methods ─────────────────────────────────────────────────────
def select_variance(spec, k=8):
    _var = spec.var(0)
    return np.argsort(_var)[::-1][:k].tolist()

def select_anova(spec_tr, y, spec_te, k=8):
    _sel = SelectKBest(f_classif, k=k).fit(spec_tr, y)
    _idx = _sel.get_support(indices=True).tolist()
    return _idx, _sel.transform(spec_tr), _sel.transform(spec_te)

def select_mi(spec_tr, y, spec_te, k=8):
    _sel = SelectKBest(mutual_info_classif, k=k).fit(spec_tr, y)
    _idx = _sel.get_support(indices=True).tolist()
    return _idx, _sel.transform(spec_tr), _sel.transform(spec_te)

def select_divmin(spec, y, k=8):
    _n_b = spec.shape[1]
    _classes = np.unique(y)
    _grand = spec.mean(0)
    _between = np.zeros(_n_b)
    _within  = np.zeros(_n_b)
    for _c in _classes:
        _mask = y == _c
        _cm   = spec[_mask].mean(0)
        _nc   = _mask.sum()
        _between += _nc * (_cm - _grand) ** 2
        _within  += ((spec[_mask] - _cm) ** 2).sum(0)
    _disc = _between / (_within + 1e-8)
    _sel = [int(np.argmax(_disc))]
    for _ in range(k - 1):
        _best_s, _best_b = -np.inf, -1
        for _b in range(_n_b):
            if _b in _sel: continue
            _corr = max(abs(float(np.corrcoef(spec[:, _b], spec[:, _s])[0, 1]))
                        for _s in _sel)
            _score = _disc[_b] - _corr
            if _score > _best_s:
                _best_s = _score; _best_b = _b
        _sel.append(_best_b)
    return sorted(_sel)

s17_results = []
_all_band_indices = {}

def _run_fs(name, feat_tr, feat_te):
    _tr_l = DataLoader(HSIDataset(X_tr, feat_tr.astype(np.float32), y_tr),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(X_te, feat_te.astype(np.float32), y_te),
                       batch_size=256, shuffle=False, num_workers=0)
    _m = QHSANet(n_bands=B, n_cls=N_CLASSES,
                 n_qubits=K_BANDS, n_q_layers=N_Q_LAYERS).to(DEVICE)
    t0 = time.time()
    train_qhsa(_m, _tr_l, EXP_EPOCHS_FS, name=name)
    _tt = time.time() - t0
    _yt, _yp, _, _, _ = eval_qhsa(_m, _te_l)
    _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
    del _m
    return dict(method=name, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt)

# PCA baseline (already in experiment_records from S14, but include for direct comparison)
_pca_fs = __import__('sklearn.decomposition', fromlist=['PCA']).PCA(n_components=K_BANDS, random_state=SEED)
_pca_tr_fs = _pca_fs.fit_transform(spec_tr_fs).astype(np.float32)
_pca_te_fs = _pca_fs.transform(spec_te_fs).astype(np.float32)
print('[S17] PCA baseline...')
try:
    _r = _run_fs('PCA (baseline)', _pca_tr_fs, _pca_te_fs)
    s17_results.append({**_r, 'band_indices': 'N/A (extracted)'})
    experiment_records.append({**_r, 'section': 'S17_FeatSel'})
    print(f'  OA={_r["OA"]:.2f}%')
except Exception as _ex: print(f'  PCA baseline failed: {_ex}')

# 1. Variance
print('[S17] Variance selection...')
try:
    _idx_v = select_variance(spec_tr_fs, K_BANDS)
    _all_band_indices['Variance'] = _idx_v
    _r = _run_fs('Variance', spec_tr_fs[:, _idx_v], spec_te_fs[:, _idx_v])
    s17_results.append({**_r, 'band_indices': str(_idx_v)})
    experiment_records.append({**_r, 'section': 'S17_FeatSel'})
    print(f'  OA={_r["OA"]:.2f}%  bands={sorted(_idx_v)}')
except Exception as _ex: print(f'  Variance failed: {_ex}')

# 2. ANOVA F-score
print('[S17] ANOVA F-score selection...')
try:
    _idx_a, _f_tr, _f_te = select_anova(spec_tr_fs, y_tr, spec_te_fs, K_BANDS)
    _all_band_indices['ANOVA'] = _idx_a
    _r = _run_fs('ANOVA F-score', _f_tr, _f_te)
    s17_results.append({**_r, 'band_indices': str(_idx_a)})
    experiment_records.append({**_r, 'section': 'S17_FeatSel'})
    print(f'  OA={_r["OA"]:.2f}%  bands={sorted(_idx_a)}')
except Exception as _ex: print(f'  ANOVA failed: {_ex}')

# 3. Mutual Information
print('[S17] Mutual Information selection...')
try:
    _idx_m, _m_tr, _m_te = select_mi(spec_tr_fs, y_tr, spec_te_fs, K_BANDS)
    _all_band_indices['MI'] = _idx_m
    _r = _run_fs('Mutual Info', _m_tr, _m_te)
    s17_results.append({**_r, 'band_indices': str(_idx_m)})
    experiment_records.append({**_r, 'section': 'S17_FeatSel'})
    print(f'  OA={_r["OA"]:.2f}%  bands={sorted(_idx_m)}')
except Exception as _ex: print(f'  Mutual Info failed: {_ex}')

# 4. DivMin Greedy
print('[S17] DivMin greedy selection...')
try:
    _idx_d = select_divmin(spec_tr_fs, y_tr, K_BANDS)
    _all_band_indices['DivMin'] = _idx_d
    _r = _run_fs('DivMin Greedy', spec_tr_fs[:, _idx_d], spec_te_fs[:, _idx_d])
    s17_results.append({**_r, 'band_indices': str(_idx_d)})
    experiment_records.append({**_r, 'section': 'S17_FeatSel'})
    print(f'  OA={_r["OA"]:.2f}%  bands={sorted(_idx_d)}')
except Exception as _ex: print(f'  DivMin failed: {_ex}')

df17 = pd.DataFrame(s17_results)
print(df17[['method','OA','AA','kappa','train_time_s']].to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig17, axes17 = plt.subplots(1, 2, figsize=(14, 5))
_x17 = np.arange(len(df17))
_w17 = 0.27
for _i, (_col, _lbl) in enumerate([('OA','OA %'),('AA','AA %'),('kappa','Kappa')]):
    if _col in df17.columns:
        axes17[0].bar(_x17 + _i*_w17, df17[_col], _w17, label=_lbl, alpha=0.85)
axes17[0].set_xticks(_x17 + _w17)
axes17[0].set_xticklabels(df17['method'], rotation=20, ha='right', fontsize=9)
axes17[0].set_ylabel('Score (%)')
axes17[0].set_title('Section 17: Feature Selection vs PCA Baseline', fontweight='bold')
axes17[0].legend(); axes17[0].set_ylim(0, 105); axes17[0].grid(axis='y', alpha=0.2)

# Band index scatter
_colors17 = ['#e74c3c','#3498db','#2ecc71','#9b59b6']
for _ci, (_mname, _bidx) in enumerate(_all_band_indices.items()):
    axes17[1].scatter(_bidx, [_ci]*len(_bidx), s=80, label=_mname,
                      color=_colors17[_ci % len(_colors17)], zorder=3)
axes17[1].set_xlabel('Band Index (0-102)')
axes17[1].set_yticks(range(len(_all_band_indices)))
axes17[1].set_yticklabels(list(_all_band_indices.keys()))
axes17[1].set_title('Selected Band Indices per Method', fontweight='bold')
axes17[1].axvline(50, color='gray', ls='--', alpha=0.4, label='Band 50')
axes17[1].legend(fontsize=8); axes17[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('fig_s17_feature_selection.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s17_feature_selection.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — Attention Mechanism Analysis & Variants
# ─────────────────────────────────────────────────────────────────────────────
S18_MD = """## 18. Attention Mechanism Analysis & Variants <a id="sec18"></a>

### Part A — Analysis of existing QHSA-Net attention
- Alpha gate (α) distribution: per-sample and per-class
- ⟨PauliZ⟩ spectral attention scores per qubit and per class
- Correlation between PCA explained variance and learned qubit attention

### Part B — Attention mechanism variants
| Variant | Measurements | Output dim |
|---------|-------------|------------|
| Baseline (PauliZ) | Z per qubit | 8 |
| Multi-observable | X+Y+Z per qubit | 24 |
| Entangled (Z+ZZ) | Z single + ZZ pairs | 8+7=15 |
| Softmax-normalised | softmax(Z) | 8 |"""

S18_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 18  Attention Mechanism Analysis & Variants     ║
# ╚══════════════════════════════════════════════════════════╝
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
import pennylane as qml
from torch.utils.data import DataLoader
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

EXP_EPOCHS_ATT = max(8, N_EPOCHS // 4)

# ── Part A: Analyse trained qhsa model ────────────────────────────────────────
print('=== Part A: Attention Analysis of Trained QHSA-Net ===')

# Capture raw PauliZ outputs via forward hook
_raw_q_list = []
def _q_hook(mod, inp, out):
    _raw_q_list.append(out.detach().cpu())
_hook_handle = qhsa.quantum.qlayer.register_forward_hook(_q_hook)

qhsa.eval()
_yt_a, _yp_a, _fc_a, _fq_a, _alpha_a = eval_qhsa(qhsa, test_loader)
_hook_handle.remove()

raw_q_attn = torch.cat(_raw_q_list).numpy()   # [N_test, 8] PauliZ values
alpha_vals  = _alpha_a.squeeze()               # [N_test]

print(f'  Alpha gate: mean={alpha_vals.mean():.3f}  std={alpha_vals.std():.3f}')
print(f'  Alpha>0.5 (quantum dominant): {(alpha_vals>0.5).mean()*100:.1f}% of samples')
print(f'  Qubit attention range: [{raw_q_attn.min():.3f}, {raw_q_attn.max():.3f}]')

# Per-class alpha and PauliZ analysis
_class_names_a = [f'C{i+1}' for i in range(N_CLASSES)]
try:
    _class_names_a = CLASS_NAMES
except: pass
_alpha_per_class  = [alpha_vals[_yt_a == _c] for _c in range(N_CLASSES)]
_q_attn_per_class = np.array([raw_q_attn[_yt_a == _c].mean(0)
                               if (_yt_a == _c).any() else np.zeros(N_QUBITS)
                               for _c in range(N_CLASSES)])  # [N_cls, 8]

# PCA variance vs qubit attention correlation
_pca_var = pca.explained_variance_ratio_
_mean_q_attn = np.abs(raw_q_attn).mean(0)     # [8] mean |PauliZ| per qubit
_corr_pca_q  = float(np.corrcoef(_pca_var, _mean_q_attn)[0, 1])
print(f'  Pearson corr(PCA variance, |PauliZ| attention): {_corr_pca_q:.3f}')

# ── Part A plots ──────────────────────────────────────────────────────────────
fig18a, axes18a = plt.subplots(2, 2, figsize=(14, 10))

# 1. Alpha histogram
axes18a[0,0].hist(alpha_vals, bins=40, color='#3498db', alpha=0.8, edgecolor='white')
axes18a[0,0].axvline(0.5, color='red', ls='--', label='alpha=0.5')
axes18a[0,0].set_xlabel('Alpha gate value')
axes18a[0,0].set_ylabel('Count')
axes18a[0,0].set_title('Gate-alpha Distribution (0=classical, 1=quantum)', fontweight='bold')
axes18a[0,0].legend()

# 2. Per-class alpha boxplot
axes18a[0,1].boxplot(_alpha_per_class, labels=_class_names_a, showfliers=False)
axes18a[0,1].set_xlabel('Class')
axes18a[0,1].set_ylabel('Alpha value')
axes18a[0,1].axhline(0.5, color='red', ls='--', alpha=0.5)
axes18a[0,1].set_title('Per-class Alpha Gate Values', fontweight='bold')
for _lbl in axes18a[0,1].get_xticklabels():
    _lbl.set_rotation(30); _lbl.set_ha('right')

# 3. Qubit attention heatmap (per class)
_im = axes18a[1,0].imshow(_q_attn_per_class, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
axes18a[1,0].set_xticks(range(N_QUBITS))
axes18a[1,0].set_xticklabels([f'Q{i+1}' for i in range(N_QUBITS)])
axes18a[1,0].set_yticks(range(N_CLASSES))
axes18a[1,0].set_yticklabels(_class_names_a, fontsize=8)
axes18a[1,0].set_title('Mean PauliZ per Qubit per Class (Spectral Attention)', fontweight='bold')
plt.colorbar(_im, ax=axes18a[1,0], label='<PauliZ>')

# 4. PCA variance vs qubit attention
axes18a[1,1].bar(range(N_QUBITS), _pca_var * 100, alpha=0.6, label='PCA expl.var %', color='#e67e22')
_ax_r = axes18a[1,1].twinx()
_ax_r.plot(range(N_QUBITS), _mean_q_attn, 'bs-', lw=2, ms=8, label='|PauliZ| mean')
axes18a[1,1].set_xlabel('PCA Component / Qubit Index')
axes18a[1,1].set_ylabel('PCA Explained Variance (%)', color='#e67e22')
_ax_r.set_ylabel('Mean |PauliZ|', color='blue')
axes18a[1,1].set_title(f'PCA Variance vs Qubit Attention (r={_corr_pca_q:.2f})', fontweight='bold')
axes18a[1,1].legend(loc='upper right'); _ax_r.legend(loc='upper center')

plt.suptitle('Section 18A: QHSA-Net Attention Analysis', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_s18a_attention_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Part B: Attention mechanism variants ──────────────────────────────────────
print('=== Part B: Attention Variant Training ===')

class QuantumBranch_MultiObs(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, proj_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        _dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(_dev, interface='torch', diff_method='best')
        def _circ(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return ([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliY(i)) for i in range(n_qubits)])

        _ws = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(_circ, {'weights': _ws})
        self.proj   = nn.Sequential(nn.Linear(n_qubits*3, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('pi', torch.tensor(float(np.pi)))

    def forward(self, pca_x):
        angles = torch.sigmoid(pca_x) * self.pi
        q_out  = self.qlayer(angles)    # [B, 3*n_qubits]
        return self.proj(q_out)

class QuantumBranch_Entangled(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, proj_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        _dev = qml.device('default.qubit', wires=n_qubits)
        _out_dim = n_qubits + (n_qubits - 1)

        @qml.qnode(_dev, interface='torch', diff_method='best')
        def _circ(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            _z  = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            _zz = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1)) for i in range(n_qubits-1)]
            return _z + _zz

        _ws = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(_circ, {'weights': _ws})
        self.proj   = nn.Sequential(nn.Linear(_out_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('pi', torch.tensor(float(np.pi)))

    def forward(self, pca_x):
        angles = torch.sigmoid(pca_x) * self.pi
        return self.proj(self.qlayer(angles))

class QuantumBranch_Softmax(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, proj_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        _dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(_dev, interface='torch', diff_method='best')
        def _circ(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        _ws = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(_circ, {'weights': _ws})
        self.proj   = nn.Sequential(nn.Linear(n_qubits, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('pi', torch.tensor(float(np.pi)))

    def forward(self, pca_x):
        angles = torch.sigmoid(pca_x) * self.pi
        q_out  = self.qlayer(angles)
        q_att  = torch.softmax(q_out, dim=-1)   # softmax-normalised attention
        return self.proj(q_att)

class QHSANet_AltMeas(nn.Module):
    def __init__(self, q_branch, n_cls=N_CLASSES):
        super().__init__()
        self.classical  = ClassicalSpatialBranch(n_bands=B, out_dim=64)
        self.quantum    = q_branch
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        fc = self.classical(patch_x)
        fq = self.quantum(pca_x)
        return self.classifier(self.fusion(fc, fq))

s18b_results = []
s18b_results.append(dict(variant='Baseline (PauliZ)', OA=_oa_a, AA=_aa_a, kappa=_kap_a, train_time_s=0.0))

for _vname, _qbranch_cls in [
    ('Multi-obs (X+Y+Z)', QuantumBranch_MultiObs),
    ('Entangled (Z+ZZ)', QuantumBranch_Entangled),
    ('Softmax-Z', QuantumBranch_Softmax),
]:
    print(f'[S18B] Training {_vname}...')
    try:
        _qb = _qbranch_cls(N_QUBITS, N_Q_LAYERS, 64)
        _vm = QHSANet_AltMeas(_qb).to(DEVICE)
        _tr_l = DataLoader(HSIDataset(X_tr, Xpca_tr, y_tr),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        _te_l = DataLoader(HSIDataset(X_te, Xpca_te, y_te),
                           batch_size=256, shuffle=False, num_workers=0)
        t0 = time.time()
        train_qhsa(_vm, _tr_l, EXP_EPOCHS_ATT, name=_vname)
        _tt = time.time() - t0
        _yt, _yp, _, _, _ = eval_qhsa(_vm, _te_l)
        _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
        s18b_results.append(dict(variant=_vname, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt))
        experiment_records.append(dict(method=_vname, OA=_oa, AA=_aa, kappa=_kap,
                                       train_time_s=_tt, section='S18_AttentionVariant'))
        print(f'  OA={_oa:.2f}%  AA={_aa:.2f}%  kappa={_kap:.2f}  time={_tt:.0f}s')
        del _vm
    except Exception as _ex:
        print(f'  {_vname} FAILED: {_ex}')

df18b = pd.DataFrame(s18b_results)
print(df18b[['variant','OA','AA','kappa']].to_string(index=False))

fig18b, ax18b = plt.subplots(figsize=(10, 5))
_x18 = np.arange(len(df18b))
_w18 = 0.28
for _i, (_col, _lbl) in enumerate([('OA','OA %'), ('AA','AA %'), ('kappa','Kappa')]):
    if _col in df18b.columns:
        ax18b.bar(_x18 + _i*_w18, df18b[_col], _w18, label=_lbl, alpha=0.85)
ax18b.set_xticks(_x18 + _w18)
ax18b.set_xticklabels(df18b['variant'], rotation=15, ha='right', fontsize=9)
ax18b.set_ylabel('Score (%)')
ax18b.set_title('Section 18B: Quantum Attention Variant Comparison', fontweight='bold')
ax18b.legend(); ax18b.set_ylim(0, 105); ax18b.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('fig_s18b_attention_variants.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: fig_s18a_attention_analysis.png  fig_s18b_attention_variants.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — Master Experiment Dashboard
# ─────────────────────────────────────────────────────────────────────────────
S19_MD = """## 19. Master Experiment Dashboard <a id="sec19"></a>

Consolidates all ablation results from Sections 13–18 into a single overview.
Exports `all_experiments.csv` and `fig_s19_master_dashboard.png`.

This figure becomes **Fig 15** in the paper: comprehensive ablation study across
DR methods, qubit counts, architecture placements, feature selection, and attention variants."""

S19_CODE = """# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 19  Master Experiment Dashboard                 ║
# ╚══════════════════════════════════════════════════════════╝
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Add baseline models to experiment_records
for _bname, _bres in results.items():
    experiment_records.append(dict(
        method=_bname, section='Baseline',
        OA=_bres.get('OA', 0), AA=_bres.get('AA', 0),
        kappa=_bres.get('Kappa', 0)))

df_all = pd.DataFrame(experiment_records)
df_all.to_csv('all_experiments.csv', index=False)
print(f'Exported all_experiments.csv  ({len(df_all)} rows)')

# ── Summary table ─────────────────────────────────────────────────────────────
print('\\n=== Top-10 Configurations by OA ===')
_top = df_all.nlargest(10, 'OA')[['section','method','OA','AA','kappa']]
print(_top.to_string(index=False))

# ── Dashboard figure ──────────────────────────────────────────────────────────
_sections = df_all['section'].unique()
fig19, axes19 = plt.subplots(2, 3, figsize=(18, 11))
axes19_flat = axes19.flatten()

_section_labels = {
    'Baseline':        'Baseline Models',
    'S14_DR':          'S14: DR Method',
    'S15_QubitSweep':  'S15: Qubit Count',
    'S15_LayerSweep':  'S15: Layer Depth',
    'S16_ArchPlacement':'S16: Architecture',
    'S17_FeatSel':     'S17: Feature Selection',
    'S18_AttentionVariant': 'S18: Attention Variants',
}

_plot_order = ['Baseline','S14_DR','S15_QubitSweep','S16_ArchPlacement',
               'S17_FeatSel','S18_AttentionVariant']

for _ax_i, _sec in enumerate(_plot_order):
    _ax = axes19_flat[_ax_i]
    _df_s = df_all[df_all['section'] == _sec].copy()
    if len(_df_s) == 0:
        _ax.set_visible(False); continue
    _df_s = _df_s.sort_values('OA', ascending=True)
    _colors19 = ['#27ae60' if _v >= _df_s['OA'].max() - 0.5 else '#3498db'
                 for _v in _df_s['OA']]
    _ax.barh(_df_s['method'], _df_s['OA'], color=_colors19, alpha=0.85)
    _ax.set_xlabel('OA (%)')
    _ax.set_title(_section_labels.get(_sec, _sec), fontweight='bold', fontsize=10)
    _ax.axvline(_df_s['OA'].max(), color='green', ls='--', alpha=0.5, lw=1)
    for _tick in _ax.get_yticklabels(): _tick.set_fontsize(8)
    _ax.set_xlim(max(0, _df_s['OA'].min() - 5), 101)
    _ax.grid(axis='x', alpha=0.2)

plt.suptitle('Section 19: QHSA-Net Ablation Study Master Dashboard\\n(Quick-Run Mode)',
             fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig_s19_master_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Best configuration summary ────────────────────────────────────────────────
_best_row = df_all.loc[df_all['OA'].idxmax()]
print('\\n' + '='*60)
print('BEST CONFIGURATION ACROSS ALL ABLATIONS (Quick-Run)')
print('='*60)
print(f'  Section : {_best_row.get("section", "N/A")}')
print(f'  Method  : {_best_row.get("method", "N/A")}')
print(f'  OA      : {_best_row.get("OA", 0):.2f}%')
print(f'  AA      : {_best_row.get("AA", 0):.2f}%')
print(f'  Kappa   : {_best_row.get("kappa", 0):.2f}')
print('='*60)
print('Saved: all_experiments.csv  fig_s19_master_dashboard.png')"""

# ─────────────────────────────────────────────────────────────────────────────
# Inject all cells into the notebook
# ─────────────────────────────────────────────────────────────────────────────
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

n_before = len(nb['cells'])

new_cells = [
    md(S13_MD), code(S13_CODE),
    md(S14_MD), code(S14_CODE),
    md(S15_MD), code(S15_CODE),
    md(S16_MD), code(S16_CODE),
    md(S17_MD), code(S17_CODE),
    md(S18_MD), code(S18_CODE),
    md(S19_MD), code(S19_CODE),
]

nb['cells'][INSERT_AT:INSERT_AT] = new_cells

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_after = len(nb['cells'])
print(f'Done. Cells: {n_before} -> {n_after}  (+{n_after - n_before} inserted at position {INSERT_AT})')
