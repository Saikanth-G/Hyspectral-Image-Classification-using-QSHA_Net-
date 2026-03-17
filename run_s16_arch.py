"""
Standalone script for Section 16: DR Placement in Architecture.
Tests 4 architecture configs (A=baseline, B=MLP+PCA, C=reversed-DR, D=shared-enc)
on Pavia University, 8 epochs each (EXP_EPOCHS_ARCH = max(8, N_EPOCHS//4)).
Results are patched back into notebook cell 42.
"""
import os, time, sys, json, pathlib
os.chdir(r'c:\Users\saika\OneDrive\Desktop\test 6')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.decomposition import PCA as SKPCA
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
N_QUBITS    = 8
N_Q_LAYERS  = 2
N_EPOCHS    = 30
EXP_EPOCHS_ARCH = max(8, N_EPOCHS // 4)   # = 8
BATCH_SIZE  = 64
PATCH_SIZE  = 9
N_CLASSES   = 9   # Pavia U
B           = 103  # Pavia U bands
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}  EXP_EPOCHS_ARCH: {EXP_EPOCHS_ARCH}')

DATA_PATH = r'c:/Users/saika/OneDrive/Desktop/test 6/pavia u data/PaviaU.mat'
GT_PATH   = r'c:/Users/saika/OneDrive/Desktop/test 6/pavia u data/PaviaU_gt.mat'

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    oa    = accuracy_score(y_true, y_pred) * 100
    cm    = confusion_matrix(y_true, y_pred)
    pc    = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1) * 100, 0.0)
    aa    = pc.mean()
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    return oa, aa, kappa, pc, cm

# ── Dataset ───────────────────────────────────────────────────────────────────
class HSIDataset(Dataset):
    def __init__(self, patches, pca_feats, labels):
        # patches: [N, H, W, B] -> [N, B, H, W]
        self.X    = torch.FloatTensor(patches.transpose(0, 3, 1, 2))
        self.Xpca = torch.FloatTensor(pca_feats)
        self.y    = torch.LongTensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xpca[i], self.y[i]

# ── Model components ──────────────────────────────────────────────────────────
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
        self.proj   = nn.Sequential(nn.Linear(n_qubits, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('pi', torch.tensor(float(np.pi)))
    def forward(self, pca_x):
        angles = torch.sigmoid(pca_x) * self.pi
        return self.proj(self.qlayer(angles))

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
        return self.pool(self.enc2d(x)).flatten(1)

class GatedFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
    def forward(self, fc, fq):
        alpha = self.gate(torch.cat([fc, fq], dim=-1))
        return self.norm(alpha * fq + (1 - alpha) * fc)

# Config A: standard QHSA-Net (3D-CNN + PCA)
class QHSANet(nn.Module):
    def __init__(self, n_bands=103, n_cls=9, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS):
        super().__init__()
        self.classical  = ClassicalSpatialBranch(n_bands, out_dim=64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        return self.classifier(self.fusion(self.classical(patch_x), self.quantum(pca_x)))

# Config B: MLP classical branch (both branches get PCA features)
class ClassicalMLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, out_dim), nn.LayerNorm(out_dim))
    def forward(self, x): return self.net(x)

class QHSANet_ConfigB(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.classical  = ClassicalMLP(n_qubits, 64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        fc = self.classical(pca_x)   # ignores patch_x
        fq = self.quantum(pca_x)
        return self.classifier(self.fusion(fc, fq))

# Config C: classical=MLP(PCA), quantum=VQC(raw bands, learned projection)
class QHSANet_ConfigC(nn.Module):
    def __init__(self, n_bands=B, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.proj_in    = nn.Linear(n_bands, n_qubits)
        self.classical  = ClassicalMLP(n_qubits, 64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        fc   = self.classical(pca_x)
        cen  = patch_x[:, :, 4, 4]         # centre pixel [B, 103]
        q_in = self.proj_in(cen)
        fq   = self.quantum(q_in)
        return self.classifier(self.fusion(fc, fq))

# Config D: shared linear encoder for both branches
class QHSANet_ConfigD(nn.Module):
    def __init__(self, n_bands=B, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS, n_cls=N_CLASSES):
        super().__init__()
        self.shared     = nn.Linear(n_bands, n_qubits)
        self.classical  = ClassicalMLP(n_qubits, 64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        cen    = patch_x[:, :, 4, 4]       # centre pixel [B, 103]
        shared = self.shared(cen)
        fc = self.classical(shared)
        fq = self.quantum(shared)
        return self.classifier(self.fusion(fc, fq))

# ── Training ──────────────────────────────────────────────────────────────────
def train_qhsa(model, loader, n_ep, name='QHSA-Net'):
    model.to(DEVICE)
    crit     = nn.CrossEntropyLoss()
    q_params = list(model.quantum.qlayer.parameters()) if hasattr(model, 'quantum') else []
    cl_params = [p for n, p in model.named_parameters()
                 if 'quantum.qlayer' not in n]
    opt = optim.Adam([{'params': cl_params, 'lr': 1e-3},
                      {'params': q_params,  'lr': 1e-2}], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    t_start = time.time()
    for ep in range(1, n_ep + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for patches_b, pca_b, labels_b in loader:
            patches_b = patches_b.to(DEVICE); pca_b = pca_b.to(DEVICE); labels_b = labels_b.to(DEVICE)
            opt.zero_grad()
            out  = model(patches_b, pca_b)
            loss = nn.CrossEntropyLoss()(out, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            run_loss    += loss.item()
            run_correct += (out.argmax(1) == labels_b).sum().item()
            run_total   += labels_b.size(0)
        sch.step()
        if ep % 2 == 0 or ep == 1 or ep == n_ep:
            print(f'  [{name}] ep {ep:2d}/{n_ep}  loss={run_loss/len(loader):.4f}  '
                  f'acc={100*run_correct/run_total:.1f}%  ({time.time()-t_start:.0f}s)')
    print(f'  Done: {(time.time()-t_start)/60:.1f}min')
    return time.time() - t_start

@torch.no_grad()
def eval_simple(model, loader):
    model.eval()
    preds, truth = [], []
    for patches_b, pca_b, labels_b in loader:
        patches_b = patches_b.to(DEVICE); pca_b = pca_b.to(DEVICE)
        out = model(patches_b, pca_b)
        preds.append(out.argmax(1).cpu().numpy())
        truth.append(labels_b.numpy())
    return np.concatenate(truth), np.concatenate(preds)

# ── Load Pavia U ──────────────────────────────────────────────────────────────
print('\n=== Section 16: DR Placement in Architecture ===')
raw = sio.loadmat(DATA_PATH)
gt_m = sio.loadmat(GT_PATH)
hsi = raw['paviaU'].astype(np.float32)      # (610, 340, 103)
gt  = gt_m['paviaU_gt'].astype(np.int32)
H, W, B_loaded = hsi.shape
assert B_loaded == B, f"Expected {B} bands, got {B_loaded}"
print(f'  Shape: {H}x{W}x{B}  Classes: {N_CLASSES}  Labeled: {(gt>0).sum():,}')

# Normalize
hsi_n = (hsi - hsi.min(0)) / (hsi.max(0) - hsi.min(0) + 1e-8)

# Train/test split (10% train, matching notebook)
rows, cols = np.where(gt > 0)
labels = gt[rows, cols] - 1
np.random.seed(SEED)
idx = np.random.permutation(len(labels))
n_train = int(0.1 * len(idx))
tr_idx, te_idx = idx[:n_train], idx[n_train:]
print(f'  Train: {n_train:,}  Test: {len(te_idx):,}')

# Patch extraction (output: [N, PATCH_SIZE, PATCH_SIZE, B] = HWB format)
ph = PATCH_SIZE // 2
padded = np.pad(hsi_n, ((ph, ph), (ph, ph), (0, 0)), mode='reflect')

def extract_patches(ridx, cidx):
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :]
                     for r, c in zip(ridx, cidx)]).astype(np.float32)

print('  Extracting patches...')
X_tr = extract_patches(rows[tr_idx], cols[tr_idx])
X_te = extract_patches(rows[te_idx], cols[te_idx])
y_tr = labels[tr_idx]
y_te = labels[te_idx]

# PCA for quantum branch
spec_tr = X_tr[:, ph, ph, :]   # centre pixel [N, 103]
spec_te = X_te[:, ph, ph, :]
pca = SKPCA(n_components=N_QUBITS, random_state=SEED)
Xpca_tr = pca.fit_transform(spec_tr).astype(np.float32)
Xpca_te = pca.transform(spec_te).astype(np.float32)
print(f'  PCA variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%')

# ── Run arch configs ──────────────────────────────────────────────────────────
s16_results = []
output_lines = []

def log(line=''):
    print(line)
    output_lines.append(line + '\n')

log('\n' + '='*60)
log('  SECTION 16: DR Placement in Architecture')
log('='*60)
log(f'  EXP_EPOCHS_ARCH = {EXP_EPOCHS_ARCH}')

def _run_arch(name, model_instance):
    tr_l = DataLoader(HSIDataset(X_tr, Xpca_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te_l = DataLoader(HSIDataset(X_te, Xpca_te, y_te), batch_size=256, shuffle=False, num_workers=0)
    t0 = time.time()
    train_qhsa(model_instance, tr_l, EXP_EPOCHS_ARCH, name=name)
    tt = time.time() - t0
    yt, yp = eval_simple(model_instance, te_l)
    oa, aa, kap, pc, _ = compute_metrics(yt, yp)
    return dict(config=name, OA=oa, AA=aa, kappa=kap, train_time_s=tt)

for cfg_name, cfg_cls, cfg_kwargs in [
    ('A: 3D-CNN+PCA (baseline)',    QHSANet,        {'n_bands': B, 'n_cls': N_CLASSES}),
    ('B: MLP+PCA (joint)',          QHSANet_ConfigB, {}),
    ('C: MLP-PCA+VQC-Raw (reversed)',QHSANet_ConfigC,{'n_bands': B}),
    ('D: Shared-Enc (joint)',       QHSANet_ConfigD, {'n_bands': B}),
]:
    log(f'\n[S16] Training {cfg_name}...')
    torch.manual_seed(SEED); np.random.seed(SEED)
    try:
        m = cfg_cls(**cfg_kwargs).to(DEVICE)
        r = _run_arch(cfg_name, m)
        s16_results.append(r)
        log(f'  OA={r["OA"]:.2f}%  AA={r["AA"]:.2f}%  kappa={r["kappa"]:.2f}  time={r["train_time_s"]:.0f}s')
        del m
    except Exception as ex:
        log(f'  FAILED: {ex}')
        import traceback; traceback.print_exc()

df16 = pd.DataFrame(s16_results)
log('\n' + df16[['config','OA','AA','kappa','train_time_s']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig16, ax16 = plt.subplots(figsize=(12, 5))
x = np.arange(len(df16)); w = 0.28
for i, (col, lbl) in enumerate([('OA','OA %'), ('AA','AA %'), ('kappa','Kappa')]):
    ax16.bar(x + i*w, df16[col], w, label=lbl, alpha=0.85)
ax16.set_xticks(x + w)
ax16.set_xticklabels(df16['config'], rotation=20, ha='right', fontsize=9)
ax16.set_ylabel('Score (%)')
ax16.set_title('Section 16: DR Placement in Architecture', fontweight='bold')
ax16.legend(); ax16.set_ylim(0, 105); ax16.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig('fig_s16_arch_placement.png', dpi=150, bbox_inches='tight')
log('Saved: fig_s16_arch_placement.png')

df16.to_csv('s16_arch_results.csv', index=False)
log('Saved: s16_arch_results.csv')
print('\n'.join(output_lines))

# ── Patch notebook cell 42 ────────────────────────────────────────────────────
NB_PATH = pathlib.Path(r'c:/Users/saika/OneDrive/Desktop/test 6/QHSA_Net_Research_Notebook_2.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

nb['cells'][42]['outputs'] = [
    {"output_type": "stream", "name": "stdout", "text": line}
    for line in output_lines
    if line.strip()
]

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nNotebook cell 42 (S16) patched with outputs.')
