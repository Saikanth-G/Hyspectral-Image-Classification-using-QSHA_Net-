"""
Standalone script to run Section 21 (Indian Pines) and Section 22 (Salinas).
This avoids re-running the full 6-hour notebook — only S21+S22 need to be re-run.

The print_metrics bug in the injected cells caused the notebook to fail at the
reporting stage of S21 (after training completed successfully).
"""
import os, time, sys
os.chdir(r'c:\Users\saika\OneDrive\Desktop\test 6')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.decomposition import PCA as SKPCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pennylane as qml

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Constants (matching notebook Cell 1 / global settings) ───────────────────
QUICK_RUN   = True
SEED        = 42
N_QUBITS    = 8
N_Q_LAYERS  = 2
N_PCA_COMP  = N_QUBITS
N_EPOCHS    = 30
BATCH_SIZE  = 64
PATCH_SIZE  = 9
MAX_TRAIN   = 3000
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES   = 9   # Pavia U (used in some function defaults)
print(f'Device: {DEVICE}')

# Pavia U OA from previous successful run (for cross-dataset summary)
PAVIA_OA = 99.91
PAVIA_AA = 99.82

# ── Metric helpers (from notebook Cell 12) ───────────────────────────────────
def compute_metrics(y_true, y_pred):
    oa    = accuracy_score(y_true, y_pred) * 100
    cm    = confusion_matrix(y_true, y_pred)
    pc    = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1) * 100, 0.0)
    aa    = pc.mean()
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    return oa, aa, kappa, pc, cm

def print_metrics(name, oa, aa, kappa, pc, class_names=None):
    sep = '=' * 60
    print(f'\n{sep}\n  {name}\n{sep}')
    print(f'  OA: {oa:.2f}%    AA: {aa:.2f}%    kappa: {kappa:.2f}')
    if class_names is not None:
        print('  Per-class accuracy:')
        for nm, acc in zip(class_names, pc):
            flag = '  <-- below 90%' if acc < 90 else ''
            print(f'    {nm:<32}: {acc:6.2f}%{flag}')
    print(sep)

# ── HSIDataset (from notebook Cell 10) ───────────────────────────────────────
class HSIDataset(Dataset):
    def __init__(self, patches, pca_feats, labels):
        self.X    = torch.FloatTensor(patches.transpose(0, 3, 1, 2))
        self.Xpca = torch.FloatTensor(pca_feats)
        self.y    = torch.LongTensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xpca[i], self.y[i]

# ── QHSA-Net Model (from notebook Cell 17) ───────────────────────────────────
class QuantumSpectralBranch(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, proj_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        w_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': w_shape})
        self.proj = nn.Sequential(nn.Linear(n_qubits, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('pi', torch.tensor(float(np.pi)))

    def forward(self, pca_x):
        angles = torch.sigmoid(pca_x) * self.pi
        q_out  = self.qlayer(angles)
        return self.proj(q_out)


class ClassicalSpatialBranch(nn.Module):
    def __init__(self, n_bands=103, out_dim=64):
        super().__init__()
        d = n_bands - 12
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, 8,  (7, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(8, 16, (5, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(16,32, (3, 3, 3), padding=(0, 1, 1)), nn.GELU(),
        )
        self.enc2d = nn.Sequential(
            nn.Conv2d(32 * d, 128, 3, padding=1), nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, out_dim, 1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.enc3d(x)
        B_, C, D, h, w = x.shape
        x = x.view(B_, C * D, h, w)
        x = self.enc2d(x)
        return self.pool(x).flatten(1)


class GatedFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, fc, fq):
        alpha = self.gate(torch.cat([fc, fq], dim=-1))
        return self.norm(alpha * fq + (1 - alpha) * fc)


class QHSANet(nn.Module):
    def __init__(self, n_bands=103, n_cls=9, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS):
        super().__init__()
        self.classical  = ClassicalSpatialBranch(n_bands, out_dim=64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, patch_x, pca_x):
        fc = self.classical(patch_x)
        fq = self.quantum(pca_x)
        return self.classifier(self.fusion(fc, fq))

# ── Training pipeline (from notebook Cell 19) ─────────────────────────────────
def train_qhsa(model, loader, n_ep, name='QHSA-Net'):
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    q_params  = list(model.quantum.qlayer.parameters())
    cl_params = [p for n, p in model.named_parameters() if 'quantum.qlayer' not in n]
    opt = optim.Adam([{'params': cl_params, 'lr': 1e-3}, {'params': q_params, 'lr': 1e-2}], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    t_start = time.time()
    for ep in range(1, n_ep + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for patches_b, pca_b, labels_b in loader:
            patches_b = patches_b.to(DEVICE); pca_b = pca_b.to(DEVICE); labels_b = labels_b.to(DEVICE)
            opt.zero_grad()
            out  = model(patches_b, pca_b)
            loss = crit(out, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            run_loss    += loss.item()
            run_correct += (out.argmax(1) == labels_b).sum().item()
            run_total   += labels_b.size(0)
        sch.step()
        if ep % 5 == 0 or ep == 1:
            print(f'  [{name}] ep {ep:3d}/{n_ep}  loss={run_loss/len(loader):.4f}  '
                  f'train_acc={100*run_correct/run_total:.1f}%  ({time.time()-t_start:.0f}s)')
    total_time = time.time() - t_start
    print(f'\n  Training complete: {total_time:.1f}s ({total_time/60:.1f} min)')


@torch.no_grad()
def eval_qhsa(model, loader):
    model.eval()
    preds, truth = [], []
    for patches_b, pca_b, labels_b in loader:
        patches_b = patches_b.to(DEVICE); pca_b = pca_b.to(DEVICE)
        fc = model.classical(patches_b)
        fq = model.quantum(pca_b)
        out = model.classifier(model.fusion(fc, fq))
        preds.append(out.argmax(1).cpu().numpy())
        truth.append(labels_b.numpy())
    return np.concatenate(truth), np.concatenate(preds)


# =============================================================================
#  SECTION 21: Indian Pines
# =============================================================================
IP_PATH    = r'c:/Users/saika/OneDrive/Desktop/test 6/indian pines data/Indian_pines_corrected.mat'
IP_GT_PATH = r'c:/Users/saika/OneDrive/Desktop/test 6/indian pines data/Indian_pines_gt.mat'
IP_CLASS_NAMES = [
    'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
    'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
    'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
    'Woods', 'Buildings-Grass-Trees', 'Stone-Steel-Towers'
]
IP_N_CLS = 16

print('\n' + '='*60)
print('  SECTION 21: Indian Pines Dataset')
print('='*60)
_ip  = sio.loadmat(IP_PATH)
_ipg = sio.loadmat(IP_GT_PATH)
ip_hsi = _ip['indian_pines_corrected'].astype(np.float32)
ip_gt  = _ipg['indian_pines_gt'].astype(np.int32)
H_ip, W_ip, B_ip = ip_hsi.shape
print(f'  Shape: {H_ip}x{W_ip}x{B_ip}  Classes: {IP_N_CLS}  Labeled: {int((ip_gt>0).sum()):,}')

ip_hsi_n = (ip_hsi - ip_hsi.min(0)) / (ip_hsi.max(0) - ip_hsi.min(0) + 1e-8)
ip_rows, ip_cols = np.where(ip_gt > 0)
ip_lbls = ip_gt[ip_rows, ip_cols] - 1
np.random.seed(SEED)
_idx_ip  = np.random.permutation(len(ip_lbls))
_ntr_ip  = int(0.1 * len(_idx_ip))
ip_tr_idx, ip_te_idx = _idx_ip[:_ntr_ip], _idx_ip[_ntr_ip:]
print(f'  Train: {_ntr_ip:,}  Test: {len(ip_te_idx):,}')

ph_ip = PATCH_SIZE // 2
def _extr_ip(hsi_n, rows, cols):
    padded = np.pad(hsi_n, ((ph_ip,ph_ip),(ph_ip,ph_ip),(0,0)), mode='reflect')
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :] for r,c in zip(rows,cols)]).astype(np.float32)

print('  Extracting patches...')
ip_Xtr = _extr_ip(ip_hsi_n, ip_rows[ip_tr_idx], ip_cols[ip_tr_idx])
ip_Xte = _extr_ip(ip_hsi_n, ip_rows[ip_te_idx], ip_cols[ip_te_idx])
ip_ytr = ip_lbls[ip_tr_idx]
ip_yte = ip_lbls[ip_te_idx]

if QUICK_RUN and len(ip_ytr) > MAX_TRAIN:
    _qi = np.random.permutation(len(ip_ytr))[:MAX_TRAIN]
    ip_Xtr, ip_ytr = ip_Xtr[_qi], ip_ytr[_qi]
    print(f'  QUICK_RUN: train subsampled to {len(ip_ytr):,}')

ip_spec_tr = ip_Xtr[:, ph_ip, ph_ip, :].astype(np.float32)
ip_spec_te = ip_Xte[:, ph_ip, ph_ip, :].astype(np.float32)
ip_pca    = SKPCA(n_components=N_QUBITS, random_state=SEED)
ip_pca_tr = ip_pca.fit_transform(ip_spec_tr).astype(np.float32)
ip_pca_te = ip_pca.transform(ip_spec_te).astype(np.float32)
print(f'  PCA variance retained: {ip_pca.explained_variance_ratio_.sum()*100:.1f}%')

print('  SVM...')
t0 = time.time()
_svm_ip = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
_svm_ip.fit(ip_pca_tr, ip_ytr)
_ip_svm_pred = _svm_ip.predict(ip_pca_te)
ip_svm_oa, ip_svm_aa, ip_svm_kap, _, _ = compute_metrics(ip_yte, _ip_svm_pred)
print(f'    OA={ip_svm_oa:.2f}%  AA={ip_svm_aa:.2f}%  kappa={ip_svm_kap:.2f}  time={time.time()-t0:.1f}s')

print('  QHSA-Net...')
ip_tr_dl = DataLoader(HSIDataset(ip_Xtr, ip_pca_tr, ip_ytr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
ip_te_dl = DataLoader(HSIDataset(ip_Xte, ip_pca_te, ip_yte), batch_size=256,        shuffle=False, num_workers=0)
ip_qhsa  = QHSANet(n_bands=B_ip, n_cls=IP_N_CLS, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS).to(DEVICE)
t0 = time.time()
train_qhsa(ip_qhsa, ip_tr_dl, N_EPOCHS, name='QHSA-IP')
ip_train_t = time.time() - t0
ip_yt, ip_yp = eval_qhsa(ip_qhsa, ip_te_dl)
ip_oa, ip_aa, ip_kap, ip_pc, ip_cm = compute_metrics(ip_yt, ip_yp)
print(f'  QHSA-Net: OA={ip_oa:.2f}%  AA={ip_aa:.2f}%  kappa={ip_kap:.2f}  time={ip_train_t/60:.1f}min')
print_metrics('QHSA-Net (Indian Pines)', ip_oa, ip_aa, ip_kap, ip_pc, IP_CLASS_NAMES)

df21 = pd.DataFrame([
    dict(method='SVM (PCA-8)', OA=ip_svm_oa, AA=ip_svm_aa, kappa=ip_svm_kap),
    dict(method='QHSA-Net',    OA=ip_oa,     AA=ip_aa,     kappa=ip_kap),
])
print(df21.to_string(index=False))

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
print('Saved: fig_s21_indian_pines.png')
plt.close()

# =============================================================================
#  SECTION 22: Salinas
# =============================================================================
SAL_PATH    = r'c:/Users/saika/OneDrive/Desktop/test 6/salinas data/Salinas_corrected.mat'
SAL_GT_PATH = r'c:/Users/saika/OneDrive/Desktop/test 6/salinas data/Salinas_gt.mat'
SAL_CLASS_NAMES = [
    'Broccolini-Grn-Wds-1', 'Broccolini-Grn-Wds-2', 'Fallow', 'Fallow-Rough-Plow',
    'Fallow-Smooth', 'Stubble', 'Celery', 'Grapes-Untrained', 'Soil-Vinyard-Dev',
    'Corn-Senesced-Grn-Wds', 'Lettuce-Romaine-4wk', 'Lettuce-Romaine-5wk',
    'Lettuce-Romaine-6wk', 'Lettuce-Romaine-7wk', 'Vinyard-Untrained', 'Vinyard-Vertical'
]
SAL_N_CLS = 16

print('\n' + '='*60)
print('  SECTION 22: Salinas Dataset')
print('='*60)
_sal  = sio.loadmat(SAL_PATH)
_salg = sio.loadmat(SAL_GT_PATH)
sal_hsi = _sal['salinas_corrected'].astype(np.float32)
sal_gt  = _salg['salinas_gt'].astype(np.int32)
H_sal, W_sal, B_sal = sal_hsi.shape
print(f'  Shape: {H_sal}x{W_sal}x{B_sal}  Classes: {SAL_N_CLS}  Labeled: {int((sal_gt>0).sum()):,}')

sal_hsi_n = (sal_hsi - sal_hsi.min(0)) / (sal_hsi.max(0) - sal_hsi.min(0) + 1e-8)
sal_rows, sal_cols = np.where(sal_gt > 0)
sal_lbls = sal_gt[sal_rows, sal_cols] - 1
np.random.seed(SEED)
_idx_sal  = np.random.permutation(len(sal_lbls))
_ntr_sal  = int(0.1 * len(_idx_sal))
sal_tr_idx, sal_te_idx = _idx_sal[:_ntr_sal], _idx_sal[_ntr_sal:]
print(f'  Train: {_ntr_sal:,}  Test: {len(sal_te_idx):,}')

ph_sal = PATCH_SIZE // 2
def _extr_sal(hsi_n, rows, cols):
    padded = np.pad(hsi_n, ((ph_sal,ph_sal),(ph_sal,ph_sal),(0,0)), mode='reflect')
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :] for r,c in zip(rows,cols)]).astype(np.float32)

print('  Extracting patches...')
sal_Xtr = _extr_sal(sal_hsi_n, sal_rows[sal_tr_idx], sal_cols[sal_tr_idx])
sal_Xte = _extr_sal(sal_hsi_n, sal_rows[sal_te_idx], sal_cols[sal_te_idx])
sal_ytr = sal_lbls[sal_tr_idx]
sal_yte = sal_lbls[sal_te_idx]

if QUICK_RUN and len(sal_ytr) > MAX_TRAIN:
    _qi = np.random.permutation(len(sal_ytr))[:MAX_TRAIN]
    sal_Xtr, sal_ytr = sal_Xtr[_qi], sal_ytr[_qi]
    print(f'  QUICK_RUN: train subsampled to {len(sal_ytr):,}')

sal_spec_tr = sal_Xtr[:, ph_sal, ph_sal, :].astype(np.float32)
sal_spec_te = sal_Xte[:, ph_sal, ph_sal, :].astype(np.float32)
sal_pca    = SKPCA(n_components=N_QUBITS, random_state=SEED)
sal_pca_tr = sal_pca.fit_transform(sal_spec_tr).astype(np.float32)
sal_pca_te = sal_pca.transform(sal_spec_te).astype(np.float32)
print(f'  PCA variance retained: {sal_pca.explained_variance_ratio_.sum()*100:.1f}%')

print('  SVM...')
t0 = time.time()
_svm_sal = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
_svm_sal.fit(sal_pca_tr, sal_ytr)
_sal_svm_pred = _svm_sal.predict(sal_pca_te)
sal_svm_oa, sal_svm_aa, sal_svm_kap, _, _ = compute_metrics(sal_yte, _sal_svm_pred)
print(f'    OA={sal_svm_oa:.2f}%  AA={sal_svm_aa:.2f}%  kappa={sal_svm_kap:.2f}  time={time.time()-t0:.1f}s')

print('  QHSA-Net...')
sal_tr_dl = DataLoader(HSIDataset(sal_Xtr, sal_pca_tr, sal_ytr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
sal_te_dl = DataLoader(HSIDataset(sal_Xte, sal_pca_te, sal_yte), batch_size=256,        shuffle=False, num_workers=0)
sal_qhsa  = QHSANet(n_bands=B_sal, n_cls=SAL_N_CLS, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS).to(DEVICE)
t0 = time.time()
train_qhsa(sal_qhsa, sal_tr_dl, N_EPOCHS, name='QHSA-Sal')
sal_train_t = time.time() - t0
sal_yt, sal_yp = eval_qhsa(sal_qhsa, sal_te_dl)
sal_oa, sal_aa, sal_kap, sal_pc, sal_cm = compute_metrics(sal_yt, sal_yp)
print(f'  QHSA-Net: OA={sal_oa:.2f}%  AA={sal_aa:.2f}%  kappa={sal_kap:.2f}  time={sal_train_t/60:.1f}min')
print_metrics('QHSA-Net (Salinas)', sal_oa, sal_aa, sal_kap, sal_pc, SAL_CLASS_NAMES)

df22 = pd.DataFrame([
    dict(method='SVM (PCA-8)', OA=sal_svm_oa, AA=sal_svm_aa, kappa=sal_svm_kap),
    dict(method='QHSA-Net',    OA=sal_oa,     AA=sal_aa,     kappa=sal_kap),
])
print(df22.to_string(index=False))

# Cross-dataset summary
print('\n=== Cross-Dataset QHSA-Net Summary ===')
_cross = pd.DataFrame([
    dict(dataset='Pavia U (QUICK_RUN)', bands=103, classes=9,  OA=PAVIA_OA,  AA=PAVIA_AA),
    dict(dataset='Indian Pines',        bands=200, classes=16, OA=ip_oa,     AA=ip_aa),
    dict(dataset='Salinas',             bands=204, classes=16, OA=sal_oa,    AA=sal_aa),
])
print(_cross.to_string(index=False))
_cross.to_csv('cross_dataset_summary.csv', index=False)
print('Saved: cross_dataset_summary.csv')

fig22, axes22 = plt.subplots(1, 2, figsize=(14, 5))
axes22[0].bar(range(SAL_N_CLS), sal_pc, color='#27ae60', alpha=0.85)
axes22[0].set_xticks(range(SAL_N_CLS))
axes22[0].set_xticklabels([n[:12] for n in SAL_CLASS_NAMES], rotation=45, ha='right', fontsize=8)
axes22[0].set_ylabel('Per-class Accuracy (%)')
axes22[0].set_title('Salinas: QHSA-Net Per-class Accuracy', fontweight='bold')
axes22[0].axhline(sal_oa, color='red', ls='--', label='OA='+str(round(sal_oa,1))+'%')
axes22[0].legend()
_datasets22 = ['Pavia U', 'Indian Pines', 'Salinas']
_oas22 = [PAVIA_OA, ip_oa, sal_oa]
_colors22 = ['#3498db', '#e67e22', '#27ae60']
axes22[1].bar(_datasets22, _oas22, color=_colors22, alpha=0.85)
axes22[1].set_ylabel('OA (%)'); axes22[1].set_ylim(0, 105)
axes22[1].set_title('QHSA-Net OA Across Datasets', fontweight='bold')
for _i, _v in enumerate(_oas22):
    axes22[1].text(_i, _v+0.5, f'{_v:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_s22_salinas.png', dpi=150, bbox_inches='tight')
print('Saved: fig_s22_salinas.png')
plt.close()

print('\n' + '='*60)
print('  DONE — S21 and S22 complete.')
print('='*60)
print(f'  Indian Pines  QHSA-Net: OA={ip_oa:.2f}%  AA={ip_aa:.2f}%  kappa={ip_kap:.2f}')
print(f'  Salinas       QHSA-Net: OA={sal_oa:.2f}%  AA={sal_aa:.2f}%  kappa={sal_kap:.2f}')
