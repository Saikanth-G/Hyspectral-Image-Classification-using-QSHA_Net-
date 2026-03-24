"""Runs t-SNE (Section 4) and convergence plots (Section 5) only."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import scipy.io as sio
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORKDIR = r'c:/Users/saika/OneDrive/Desktop/test 6'
PATCH_SIZE = 9; BATCH_SIZE = 64; DEVICE = torch.device('cpu'); SEED = 42; N_COMP = 4

def save(f):
    plt.savefig(os.path.join(WORKDIR, f), dpi=150, bbox_inches='tight')
    plt.close(); print(f'Saved: {f}')

# ── Load Pavia U ───────────────────────────────────────────────────────────────
raw = sio.loadmat(os.path.join(WORKDIR, 'pavia u data', 'PaviaU.mat'))
gt  = sio.loadmat(os.path.join(WORKDIR, 'pavia u data', 'PaviaU_gt.mat'))
HSI = raw['paviaU'].astype(np.float32); GT = gt['paviaU_gt'].astype(np.int32)
H, W, B = HSI.shape
HSI = (HSI - HSI.min((0,1), keepdims=True)) / (HSI.max((0,1), keepdims=True) - HSI.min((0,1), keepdims=True) + 1e-8)
rng = np.random.default_rng(SEED)
rows, cols = np.where(GT > 0); labels = GT[rows, cols] - 1
tr_idx, te_idx = [], []
for c in range(9):
    cidx = np.where(labels == c)[0]; n_tr = max(int(0.10 * len(cidx)), 3)
    perm = rng.permutation(len(cidx))
    tr_idx.extend(cidx[perm[:n_tr]].tolist()); te_idx.extend(cidx[perm[n_tr:]].tolist())
tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
PAD = PATCH_SIZE // 2
hpad = np.pad(HSI, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')
def extr(ridx):
    out = np.empty((len(ridx), B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows[ridx], cols[ridx])):
        out[i] = hpad[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :].transpose(2,0,1)
    return out
X_tr = extr(tr_idx); X_te = extr(te_idx)
y_tr = labels[tr_idx].astype(np.int64); y_te = labels[te_idx].astype(np.int64)
fa = FactorAnalysis(n_components=N_COMP, random_state=42)
fa_tr = fa.fit_transform(X_tr[:, :, PAD, PAD]).astype(np.float32)
fa_te = fa.transform(X_te[:, :, PAD, PAD]).astype(np.float32)
print(f'Loaded PaviaU | train={len(y_tr)} test={len(y_te)}')

class DS(Dataset):
    def __init__(self, p, f, l):
        self.p = torch.from_numpy(p); self.f = torch.from_numpy(f); self.l = torch.from_numpy(l).long()
    def __len__(self): return len(self.l)
    def __getitem__(self, i): return self.p[i], self.f[i], self.l[i]

tr_ld = DataLoader(DS(X_tr, fa_tr, y_tr), BATCH_SIZE, shuffle=True,  num_workers=0)
te_ld = DataLoader(DS(X_te, fa_te, y_te), 256,        shuffle=False, num_workers=0)

# ── Build QHSA-Net ────────────────────────────────────────────────────────────
dev_qml = qml.device('default.qubit', wires=4)
ws = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)

@qml.qnode(dev_qml, interface='torch', diff_method='backprop')
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QB(nn.Module):
    def __init__(self):
        super().__init__()
        self.ql   = qml.qnn.TorchLayer(circuit, {'weights': ws})
        self.proj = nn.Sequential(nn.Linear(4, 64), nn.LayerNorm(64))
    def forward(self, x):
        x = torch.tanh(x) * np.pi
        q = self.ql(x); q = torch.softmax(q, dim=-1)
        if isinstance(q, (list, tuple)): q = torch.stack(q, dim=-1)
        return self.proj(q)

class CB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,8,(7,3,3),padding=(3,1,1)),  nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8,16,(5,3,3),padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)),nn.BatchNorm3d(32), nn.ReLU())
        fl = self.conv(torch.zeros(1,1,B,PATCH_SIZE,PATCH_SIZE)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(fl, 64), nn.LayerNorm(64))
    def forward(self, x): return self.fc(self.conv(x.unsqueeze(1)).flatten(1))

class GF(nn.Module):
    def __init__(self): super().__init__(); self.g = nn.Sequential(nn.Linear(128,64), nn.Sigmoid())
    def forward(self, fc, fq): a = self.g(torch.cat([fc,fq], -1)); return a*fq + (1-a)*fc

class QHSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = CB(); self.q = QB(); self.f = GF()
        self.clf = nn.Sequential(nn.Linear(64,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,9))
    def forward(self, p, f, ret=False):
        fc = self.c(p); fq = self.q(f); ff = self.f(fc, fq)
        if ret: return self.clf(ff), fc, fq, ff
        return self.clf(ff)

# ── Train ─────────────────────────────────────────────────────────────────────
print('Training QHSA-Net for t-SNE...')
torch.manual_seed(SEED); np.random.seed(SEED)
model = QHSA().to(DEVICE)
qp = [p for n,p in model.named_parameters() if '.q.' in n or 'ql' in n]
cp = [p for n,p in model.named_parameters() if '.q.' not in n and 'ql' not in n]
opt = optim.Adam([{'params': cp, 'lr': 1e-3}, {'params': qp, 'lr': 1e-2}])
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
crit = nn.CrossEntropyLoss()
t0 = time.time(); model.train()
for ep in range(1, 31):
    tl, tc, tn = 0., 0, 0
    for pb, fb, lb in tr_ld:
        pb, fb, lb = pb.to(DEVICE), fb.to(DEVICE), lb.to(DEVICE)
        opt.zero_grad(); out = model(pb, fb); loss = crit(out, lb)
        loss.backward(); opt.step()
        tl += loss.item()*len(lb); tc += (out.argmax(1)==lb).sum().item(); tn += len(lb)
    sched.step()
    if ep % 5 == 0 or ep == 1:
        print(f'  ep {ep:2d}/30  loss={tl/tn:.4f}  acc={tc/tn*100:.1f}%  ({time.time()-t0:.0f}s)')

# ── Extract features ──────────────────────────────────────────────────────────
print('Extracting features...')
model.eval()
all_fc, all_fq, all_ff, all_y = [], [], [], []
with torch.no_grad():
    for pb, fb, lb in te_ld:
        pb, fb = pb.to(DEVICE), fb.to(DEVICE)
        _, fc, fq, ff = model(pb, fb, ret=True)
        all_fc.append(fc.cpu().numpy()); all_fq.append(fq.cpu().numpy())
        all_ff.append(ff.cpu().numpy()); all_y.append(lb.numpy())

fc_arr = np.concatenate(all_fc); fq_arr = np.concatenate(all_fq)
ff_arr = np.concatenate(all_ff); y_arr  = np.concatenate(all_y)

n_tsne = min(3000, len(y_arr))
rng2 = np.random.default_rng(42); idx = rng2.choice(len(y_arr), n_tsne, replace=False)
fc_s, fq_s, ff_s, y_s = fc_arr[idx], fq_arr[idx], ff_arr[idx], y_arr[idx]

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print('Running t-SNE (3 runs, ~15 min)...')
tsne_fc = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(fc_s)
print('  t-SNE 1/3 done')
tsne_fq = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(fq_s)
print('  t-SNE 2/3 done')
tsne_ff = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(ff_s)
print('  t-SNE 3/3 done')

colors9 = ['#808080','#00CC00','#FF6600','#006400','#FF0000','#CCCC00','#000080','#FF69B4','#00CCCC']
cnames  = ['Asphalt','Meadows','Gravel','Trees','Painted metal','Bare soil','Bitumen','Bricks','Shadows']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Classical Branch\n(3D-CNN output)', 'Quantum Branch\n(VQC output)', 'After Gated Fusion\n(Final features)']
for ax, data, title in zip(axes, [tsne_fc, tsne_fq, tsne_ff], titles):
    for c in range(9):
        mask = y_s == c
        ax.scatter(data[mask,0], data[mask,1], c=colors9[c], s=8, alpha=0.6, label=cnames[c])
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])

handles = [mpatches.Patch(color=colors9[i], label=cnames[i]) for i in range(9)]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5,-0.06))
plt.suptitle('t-SNE Feature Visualisation — QHSA-Net Internals (Pavia University)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_tsne.png')

# ── Section 5: Convergence ────────────────────────────────────────────────────
print('Generating convergence plots...')
df_curves = pd.read_csv(os.path.join(WORKDIR, 'paper_training_curves.csv'))
THRESHOLD = 95.0
ds_labels = {'PaviaU':'Pavia University','IndianPines':'Indian Pines','Salinas':'Salinas Valley'}
clr = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','3D-CNN-Only':'#9C27B0'}
sty = {'QHSA-Net':'-','SSRN':'--','3D-CNN-Only':'-.'}

conv_rows = []
for ds in ['PaviaU','IndianPines','Salinas']:
    for m in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub = df_curves[(df_curves['model']==m)&(df_curves['dataset']==ds)&(df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        reached = sub[sub['train_acc'] >= THRESHOLD]
        conv_rows.append(dict(model=m, dataset=ds,
                              epochs_to_95=reached['epoch'].min() if not reached.empty else None,
                              final_train_acc=sub['train_acc'].iloc[-1]))
df_conv = pd.DataFrame(conv_rows)
df_conv.to_csv(os.path.join(WORKDIR, 'paper_convergence.csv'), index=False)
print(df_conv.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ['PaviaU','IndianPines','Salinas']):
    for m in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub = df_curves[(df_curves['model']==m)&(df_curves['dataset']==ds)&(df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        ax.plot(sub['epoch'], sub['train_acc'], sty[m], color=clr[m], label=m, linewidth=2)
    ax.axhline(THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_title(ds_labels[ds], fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Accuracy (%)' if ds=='PaviaU' else '')
    ax.set_ylim(0, 105); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('Training Convergence Speed', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_convergence.png')

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, ds in zip(axes, ['PaviaU','IndianPines','Salinas']):
    sub = df_conv[df_conv['dataset']==ds].dropna(subset=['epochs_to_95'])
    if sub.empty:
        ax.text(0.5,0.5,'All models below\n95% threshold',ha='center',va='center',transform=ax.transAxes); continue
    bars = ax.bar(sub['model'], sub['epochs_to_95'], color=[clr[m] for m in sub['model']], alpha=0.85)
    ax.bar_label(bars, fmt='%d ep', fontsize=9, padding=2)
    ax.set_title(ds_labels[ds], fontsize=11, fontweight='bold')
    ax.set_ylabel('Epochs to 95% train acc' if ds=='PaviaU' else '')
    ax.set_ylim(0, 35); ax.grid(axis='y', alpha=0.3)
plt.suptitle('Epochs to Reach 95% Training Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_epochs_to_95.png')

print('ALL DONE — t-SNE and convergence complete.')
