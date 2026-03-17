"""
Standalone script for Section 20: Full Pavia University Run (publication-quality).
- Config 1: QHSA-Net original (8q PCA, 30 epochs)
- Config 2: QHSA-Net improved (4q TruncatedSVD, 30 epochs)
- Seed: [42]
Results are patched back into notebook cell 50.
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
from sklearn.decomposition import PCA as SKPCA, TruncatedSVD as TSVD
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Constants ─────────────────────────────────────────────────────────────────
SEED         = 42
N_QUBITS     = 8
N_Q_LAYERS   = 2
FULL_EPOCHS  = 30
FULL_SEEDS   = [42]
BATCH_SIZE   = 64
PATCH_SIZE   = 9
N_CLASSES    = 9   # Pavia U
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}  FULL_EPOCHS: {FULL_EPOCHS}  Seeds: {FULL_SEEDS}')

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

# ── Model ─────────────────────────────────────────────────────────────────────
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

class QHSANet(nn.Module):
    def __init__(self, n_bands=103, n_cls=9, n_qubits=N_QUBITS, n_q_layers=N_Q_LAYERS):
        super().__init__()
        self.classical  = ClassicalSpatialBranch(n_bands, out_dim=64)
        self.quantum    = QuantumSpectralBranch(n_qubits, n_q_layers, proj_dim=64)
        self.fusion     = GatedFusion(dim=64)
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, patch_x, pca_x):
        return self.classifier(self.fusion(self.classical(patch_x), self.quantum(pca_x)))

# ── Training ──────────────────────────────────────────────────────────────────
def train_qhsa(model, loader, n_ep, name='QHSA-Net'):
    model.to(DEVICE)
    crit     = nn.CrossEntropyLoss()
    q_params = list(model.quantum.qlayer.parameters())
    cl_params = [p for n, p in model.named_parameters() if 'quantum.qlayer' not in n]
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
                  f'train_acc={100*run_correct/run_total:.1f}%  ({time.time()-t_start:.0f}s)',
                  flush=True)
    tt = time.time() - t_start
    print(f'\n  Training complete: {tt:.1f}s ({tt/60:.1f} min)')
    return tt

@torch.no_grad()
def eval_qhsa(model, loader):
    model.eval()
    preds, truth = [], []
    for patches_b, pca_b, labels_b in loader:
        patches_b = patches_b.to(DEVICE); pca_b = pca_b.to(DEVICE)
        out = model(patches_b, pca_b)
        preds.append(out.argmax(1).cpu().numpy())
        truth.append(labels_b.numpy())
    return np.concatenate(truth), np.concatenate(preds)

# ── Load Pavia U ──────────────────────────────────────────────────────────────
output_lines = []
def log(line=''):
    print(line, flush=True)
    output_lines.append(line + '\n')

log('\n' + '='*60)
log('  SECTION 20: Full Pavia University Run')
log('='*60)

raw  = sio.loadmat(DATA_PATH)
gt_m = sio.loadmat(GT_PATH)
hsi  = raw['paviaU'].astype(np.float32)
gt   = gt_m['paviaU_gt'].astype(np.int32)
H, W, B = hsi.shape
log(f'  Shape: {H}x{W}x{B}  Classes: {N_CLASSES}  Labeled: {(gt>0).sum():,}')

hsi_n = (hsi - hsi.min(0)) / (hsi.max(0) - hsi.min(0) + 1e-8)

rows, cols = np.where(gt > 0)
labels = gt[rows, cols] - 1
np.random.seed(42)
idx    = np.random.permutation(len(labels))
n_tr   = int(0.1 * len(idx))
tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
log(f'  Full train: {n_tr:,}  test: {len(te_idx):,}')

ph = PATCH_SIZE // 2
padded = np.pad(hsi_n, ((ph, ph), (ph, ph), (0, 0)), mode='reflect')

def extract_patches(ridx, cidx):
    return np.stack([padded[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :]
                     for r, c in zip(ridx, cidx)]).astype(np.float32)

log('  Extracting patches (full dataset)...')
Xtr = extract_patches(rows[tr_idx], cols[tr_idx])
Xte = extract_patches(rows[te_idx], cols[te_idx])
ytr = labels[tr_idx]
yte = labels[te_idx]
spec_tr = Xtr[:, ph, ph, :]
spec_te = Xte[:, ph, ph, :]

# ── Run configs ───────────────────────────────────────────────────────────────
s20_results = []

def _run_full20(seed, name, dr_tr, dr_te, n_q, n_l):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_l = DataLoader(HSIDataset(Xtr, dr_tr, ytr), batch_size=64, shuffle=True, num_workers=0)
    te_l = DataLoader(HSIDataset(Xte, dr_te, yte), batch_size=512, shuffle=False, num_workers=0)
    m = QHSANet(n_bands=B, n_cls=N_CLASSES, n_qubits=n_q, n_q_layers=n_l).to(DEVICE)
    tt = train_qhsa(m, tr_l, FULL_EPOCHS, name=f'{name}-s{seed}')
    yt, yp = eval_qhsa(m, te_l)
    oa, aa, kap, pc, _ = compute_metrics(yt, yp)
    del m
    return dict(name=name, seed=seed, OA=oa, AA=aa, kappa=kap, train_time_s=tt)

# Config 1: QHSA-8q-PCA
log('\n--- Config 1: QHSA-Net original (8q PCA, 30 epochs) ---')
for seed in FULL_SEEDS:
    log(f'  Seed {seed}...')
    pca20 = SKPCA(n_components=N_QUBITS, random_state=seed)
    dtr = pca20.fit_transform(spec_tr).astype(np.float32)
    dte = pca20.transform(spec_te).astype(np.float32)
    r = _run_full20(seed, 'QHSA-8q-PCA', dtr, dte, N_QUBITS, N_Q_LAYERS)
    s20_results.append(r)
    log(f'    OA={r["OA"]:.2f}%  AA={r["AA"]:.2f}%  kappa={r["kappa"]:.2f}  {r["train_time_s"]/60:.1f}min')

# Config 2: QHSA-4q-TruncSVD
log('\n--- Config 2: QHSA-Net improved (4q TruncSVD, 30 epochs) ---')
for seed in FULL_SEEDS:
    log(f'  Seed {seed}...')
    svd20 = TSVD(n_components=4, random_state=seed)
    dtr = svd20.fit_transform(spec_tr).astype(np.float32)
    dte = svd20.transform(spec_te).astype(np.float32)
    r = _run_full20(seed, 'QHSA-4q-TruncSVD', dtr, dte, 4, 2)
    s20_results.append(r)
    log(f'    OA={r["OA"]:.2f}%  AA={r["AA"]:.2f}%  kappa={r["kappa"]:.2f}  {r["train_time_s"]/60:.1f}min')

df20 = pd.DataFrame(s20_results)
log('\n=== Full Pavia U Results ===')
for nm, g in df20.groupby('name'):
    log(f'  {nm}: OA={g["OA"].mean():.2f}+-{g["OA"].std():.2f}  '
        f'AA={g["AA"].mean():.2f}+-{g["AA"].std():.2f}  '
        f'kappa={g["kappa"].mean():.2f}+-{g["kappa"].std():.2f}')
df20.to_csv('full_pavia_results.csv', index=False)
log('Saved: full_pavia_results.csv')

# ── Patch notebook cell 50 ────────────────────────────────────────────────────
NB_PATH = pathlib.Path(r'c:/Users/saika/OneDrive/Desktop/test 6/QHSA_Net_Research_Notebook_2.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

nb['cells'][50]['outputs'] = [
    {"output_type": "stream", "name": "stdout", "text": line}
    for line in output_lines
    if line.strip()
]

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nNotebook cell 50 (S20) patched with outputs.')
