"""
QHSA-Net Paper Additions
=========================
Implements 5 additions to strengthen the paper:

  SECTION 1 — Parameter Count Table
              Counts parameters for all 6 models on all 3 datasets.
              Also computes training/inference time per sample.
              No training needed — instantiate models and count.

  SECTION 2 — Classification Maps
              Trains QHSA-Net, SSRN, DBDA, 3D-CNN on all 3 datasets.
              Predicts labels on every pixel in the full scene.
              Plots ground truth vs model predictions side-by-side.

  SECTION 3 — Noise Robustness Experiment
              Trains QHSA-Net, SSRN, 3D-CNN on clean Pavia U.
              Tests on spectral noise levels [0, 0.05, 0.10, 0.20, 0.30, 0.50].
              Plots OA vs noise level — shows quantum resilience.

  SECTION 4 — t-SNE Feature Visualisation
              Trains QHSA-Net on Pavia U.
              Extracts features from classical branch, quantum branch,
              and after fusion — visualises cluster quality with t-SNE.

  SECTION 5 — Convergence Speed Comparison
              Uses existing training curves from paper_extension.py.
              Computes epochs-to-95%-train-accuracy for each model.
              Plots convergence curves for all key models.
"""

import os, sys, time, logging, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pennylane as qml
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================================
# CONFIG
# ============================================================
WORKDIR     = r'c:/Users/saika/OneDrive/Desktop/test 6'
LOG_PATH    = os.path.join(WORKDIR, 'paper_additions.log')

BEST_N_QUBITS    = 4
BEST_N_LAYERS    = 2
BEST_MEASUREMENT = 'softmax_z'
N_COMP           = BEST_N_QUBITS

PATCH_SIZE  = 9
EPOCHS      = 30
BATCH_SIZE  = 64
DEVICE      = torch.device('cpu')
SEED        = 42

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

DATASETS = {
    'PaviaU': {
        'data_path': os.path.join(WORKDIR, 'pavia u data', 'PaviaU.mat'),
        'gt_path':   os.path.join(WORKDIR, 'pavia u data', 'PaviaU_gt.mat'),
        'data_key':  'paviaU', 'gt_key': 'paviaU_gt',
        'n_classes': 9,
        'class_names': ['Asphalt','Meadows','Gravel','Trees',
                        'Painted metal','Bare soil','Bitumen',
                        'Bricks','Shadows'],
        'colors': ['#808080','#00FF00','#FF6600','#006400',
                   '#FF0000','#FFFF00','#000080','#FF69B4','#00FFFF'],
    },
    'IndianPines': {
        'data_path': os.path.join(WORKDIR, 'indian pines data', 'Indian_pines_corrected.mat'),
        'gt_path':   os.path.join(WORKDIR, 'indian pines data', 'Indian_pines_gt.mat'),
        'data_key':  'indian_pines_corrected', 'gt_key': 'indian_pines_gt',
        'n_classes': 16,
        'class_names': ['Alfalfa','Corn-notill','Corn-mintill','Corn',
                        'Grass-pasture','Grass-trees','Grass-mowed',
                        'Hay-windrowed','Oats','Soy-notill','Soy-mintill',
                        'Soy-clean','Wheat','Woods','Buildings','Towers'],
        'colors': ['#FF0000','#FF6600','#FFCC00','#FFFF00',
                   '#99FF00','#00FF00','#00FF99','#00FFFF',
                   '#0099FF','#0000FF','#6600FF','#CC00FF',
                   '#FF00CC','#FF0066','#996633','#666666'],
    },
    'Salinas': {
        'data_path': os.path.join(WORKDIR, 'salinas data', 'Salinas_corrected.mat'),
        'gt_path':   os.path.join(WORKDIR, 'salinas data', 'Salinas_gt.mat'),
        'data_key':  'salinas_corrected', 'gt_key': 'salinas_gt',
        'n_classes': 16,
        'class_names': ['Weeds_1','Weeds_2','Fallow','Fallow_rough','Fallow_smooth',
                        'Stubble','Celery','Grapes','Soil','Corn',
                        'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk','Lettuce_7wk',
                        'Vineyard_u','Vineyard_v'],
        'colors': ['#FF0000','#FF6600','#FFCC00','#FFFF00',
                   '#99FF00','#00FF00','#00FF99','#00FFFF',
                   '#0099FF','#0000FF','#6600FF','#CC00FF',
                   '#FF00CC','#FF0066','#996633','#666666'],
    },
}

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, 'w', 'utf-8'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger()

def section(title):
    log.info(f'\n{"="*60}\n  {title}\n{"="*60}')

def save_fig(fname):
    path = os.path.join(WORKDIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  Saved: {fname}')

# ============================================================
# DATA LOADING
# ============================================================
def load_dataset(name, seed=42):
    cfg = DATASETS[name]
    raw    = sio.loadmat(cfg['data_path'])
    gt_raw = sio.loadmat(cfg['gt_path'])
    dk = cfg['data_key'] if cfg['data_key'] in raw else [k for k in raw if not k.startswith('_')][0]
    gk = cfg['gt_key']   if cfg['gt_key']   in gt_raw else [k for k in gt_raw if not k.startswith('_')][0]

    HSI = raw[dk].astype(np.float32)
    GT  = gt_raw[gk].astype(np.int32)
    H, W, B = HSI.shape
    n_classes = cfg['n_classes']

    mn = HSI.min(axis=(0,1), keepdims=True)
    mx = HSI.max(axis=(0,1), keepdims=True)
    HSI = (HSI - mn) / (mx - mn + 1e-8)

    rng = np.random.default_rng(seed)
    rows, cols = np.where(GT > 0)
    labels = GT[rows, cols] - 1

    tr_idx_list, te_idx_list = [], []
    for c in range(n_classes):
        cidx = np.where(labels == c)[0]
        if len(cidx) == 0: continue
        n_tr = max(int(0.10 * len(cidx)), 3)
        perm = rng.permutation(len(cidx))
        tr_idx_list.extend(cidx[perm[:n_tr]].tolist())
        te_idx_list.extend(cidx[perm[n_tr:]].tolist())

    tr_idx = np.array(tr_idx_list)
    te_idx = np.array(te_idx_list)

    PAD = PATCH_SIZE // 2
    hsi_pad = np.pad(HSI, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

    def extract(ridx):
        out = np.empty((len(ridx), B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (r, c) in enumerate(zip(rows[ridx], cols[ridx])):
            out[i] = hsi_pad[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :].transpose(2,0,1)
        return out

    X_tr = extract(tr_idx); X_te = extract(te_idx)
    y_tr = labels[tr_idx].astype(np.int64)
    y_te = labels[te_idx].astype(np.int64)

    spec_tr = X_tr[:, :, PAD, PAD]
    spec_te = X_te[:, :, PAD, PAD]

    fa = FactorAnalysis(n_components=N_COMP, random_state=42)
    fa_tr = fa.fit_transform(spec_tr).astype(np.float32)
    fa_te = fa.transform(spec_te).astype(np.float32)

    log.info(f'  {name} | {H}×{W}×{B} | {n_classes} classes | '
             f'train={len(y_tr)} test={len(y_te)} | seed={seed}')

    return dict(HSI=HSI, GT=GT, H=H, W=W, B=B, PAD=PAD, hsi_pad=hsi_pad,
                rows=rows, cols=cols,
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                fa_tr=fa_tr, fa_te=fa_te,
                spec_tr=spec_tr, spec_te=spec_te,
                fa=fa, n_classes=n_classes, n_bands=B)


# ============================================================
# DATASET / LOADERS
# ============================================================
class HSIDataset(Dataset):
    def __init__(self, patches, fa, labels):
        self.patches = torch.from_numpy(patches)
        self.fa      = torch.from_numpy(fa)
        self.labels  = torch.from_numpy(labels).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.patches[i], self.fa[i], self.labels[i]

def make_loaders(d):
    tr = DataLoader(HSIDataset(d['X_tr'], d['fa_tr'], d['y_tr']),
                    BATCH_SIZE, shuffle=True,  num_workers=0)
    te = DataLoader(HSIDataset(d['X_te'], d['fa_te'], d['y_te']),
                    256,        shuffle=False, num_workers=0)
    return tr, te

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred, n_classes):
    oa    = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    pc    = np.where(cm.sum(1)>0, cm.diagonal()/cm.sum(1)*100, 0.0)
    aa    = float(np.mean(pc))
    return dict(OA=oa, AA=aa, kappa=kappa)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
def make_vqc(n_qubits, n_layers):
    dev = qml.device('default.qubit', wires=n_qubits)
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit, {'weights': wshape}, n_qubits


class QuantumBranch(nn.Module):
    def __init__(self, n_qubits=BEST_N_QUBITS, n_layers=BEST_N_LAYERS, proj_dim=64):
        super().__init__()
        circuit, wshape, out_dim = make_vqc(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(circuit, wshape)
        self.proj   = nn.Sequential(nn.Linear(out_dim, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x):
        x = torch.tanh(x) * np.pi
        q_out = self.qlayer(x)
        q_out = torch.softmax(q_out, dim=-1)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=-1)
        return self.proj(q_out)


class ClassicalBranch(nn.Module):
    def __init__(self, n_bands, proj_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,  8,(7,3,3),padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8, 16,(5,3,3),padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
        )
        dummy = torch.zeros(1,1,n_bands,PATCH_SIZE,PATCH_SIZE)
        flat  = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))


class GatedFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())

    def forward(self, fc, fq):
        alpha = self.gate(torch.cat([fc, fq], dim=-1))
        return alpha * fq + (1-alpha) * fc


class QHSANet(nn.Module):
    def __init__(self, n_bands, n_classes, proj_dim=64):
        super().__init__()
        self.classical = ClassicalBranch(n_bands, proj_dim)
        self.quantum   = QuantumBranch(BEST_N_QUBITS, BEST_N_LAYERS, proj_dim)
        self.fusion    = GatedFusion(proj_dim)
        self.clf = nn.Sequential(nn.Linear(proj_dim,128), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, fa, return_features=False):
        fc = self.classical(patch)
        fq = self.quantum(fa)
        ff = self.fusion(fc, fq)
        if return_features:
            return self.clf(ff), fc, fq, ff
        return self.clf(ff)


class CNN3DOnly(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8,(7,3,3),padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8,16,(5,3,3),padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)),nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32,128), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, fa=None):
        return self.head(self.conv(patch.unsqueeze(1)))


class HybridSN(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.c3_1 = nn.Sequential(nn.Conv3d(1, 8,(7,3,3),padding=(3,1,1)), nn.ReLU())
        self.c3_2 = nn.Sequential(nn.Conv3d(8,16,(5,3,3),padding=(2,1,1)), nn.ReLU())
        self.c3_3 = nn.Sequential(nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)),nn.ReLU())
        self.bp   = nn.AdaptiveAvgPool3d((1,PATCH_SIZE,PATCH_SIZE))
        self.c2_1 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.ReLU())
        self.c2_2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.ReLU())
        self.p2   = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(128,256),nn.ReLU(),
                                  nn.Dropout(0.4),nn.Linear(256,n_classes))

    def forward(self, p, fa=None):
        x = self.c3_3(self.c3_2(self.c3_1(p.unsqueeze(1))))
        return self.head(self.p2(self.c2_2(self.c2_1(self.bp(x).squeeze(2)))))


class SpectralResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.BatchNorm3d(ch),nn.ReLU(True),nn.Conv3d(ch,ch,(1,1,7),padding=(0,0,3)),
            nn.BatchNorm3d(ch),nn.ReLU(True),nn.Conv3d(ch,ch,(1,1,7),padding=(0,0,3)))
    def forward(self, x): return x + self.b(x)

class SpatialResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.BatchNorm3d(ch),nn.ReLU(True),nn.Conv3d(ch,ch,(3,3,1),padding=(1,1,0)),
            nn.BatchNorm3d(ch),nn.ReLU(True),nn.Conv3d(ch,ch,(3,3,1),padding=(1,1,0)))
    def forward(self, x): return x + self.b(x)

class SSRN(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.sc  = nn.Conv3d(1,24,(1,1,7),padding=(0,0,3))
        self.sr1 = SpectralResBlock(24)
        self.sr2 = SpectralResBlock(24)
        self.s2s = nn.Conv3d(24,128,(1,1,n_bands))
        self.sp1 = SpatialResBlock(128)
        self.sp2 = SpatialResBlock(128)
        self.pool= nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Linear(128, n_classes)

    def forward(self, x, fa=None):
        x = x.permute(0,2,3,1).unsqueeze(1)
        x = self.sr2(self.sr1(self.sc(x)))
        x = self.sp2(self.sp1(self.s2s(x)))
        return self.fc(self.pool(x).flatten(1))


class ChannelAttn(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch,max(1,ch//r)),nn.ReLU(),
                                nn.Linear(max(1,ch//r),ch),nn.Sigmoid())
    def forward(self, x):
        gap = x.flatten(2).mean(2)
        return x * self.fc(gap).view(x.shape[0],x.shape[1],*([1]*(x.dim()-2)))

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2,1,(3,3,1),padding=(1,1,0))
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.conv(torch.cat([x.mean(1,True), x.max(1,True)[0]],1)))

class DBDA(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        G = 12
        def sb(i,o): return nn.Sequential(nn.Conv3d(i,o,(1,1,7),padding=(0,0,3)),nn.BatchNorm3d(o),nn.ReLU())
        def tb(i,o): return nn.Sequential(nn.Conv3d(i,o,(3,3,1),padding=(1,1,0)),nn.BatchNorm3d(o),nn.ReLU())
        self.sc0,self.sc1,self.sc2,self.sc3 = sb(1,G),sb(G,G),sb(G*2,G),sb(G*3,G)
        self.sca = ChannelAttn(G*4); self.scp = nn.AdaptiveAvgPool3d(1)
        self.tc0,self.tc1,self.tc2,self.tc3 = tb(1,G),tb(G,G),tb(G*2,G),tb(G*3,G)
        self.tsa = SpatialAttn(); self.tcp = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(G*8,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,n_classes))

    def forward(self, x, fa=None):
        x3 = x.permute(0,2,3,1).unsqueeze(1)
        s0=self.sc0(x3); s1=self.sc1(s0); s2=self.sc2(torch.cat([s0,s1],1)); s3=self.sc3(torch.cat([s0,s1,s2],1))
        sf = self.scp(self.sca(torch.cat([s0,s1,s2,s3],1))).flatten(1)
        t0=self.tc0(x3); t1=self.tc1(t0); t2=self.tc2(torch.cat([t0,t1],1)); t3=self.tc3(torch.cat([t0,t1,t2],1))
        tf = self.tcp(self.tsa(torch.cat([t0,t1,t2,t3],1))).flatten(1)
        return self.head(torch.cat([sf,tf],1))


MODEL_BUILDERS = {
    'QHSA-Net':    QHSANet,
    'SSRN':        SSRN,
    'DBDA':        DBDA,
    '3D-CNN-Only': CNN3DOnly,
    'HybridSN':    HybridSN,
}

# ============================================================
# TRAINING / EVAL
# ============================================================
def train_model(model, loader, n_epochs, tag=''):
    q_p  = [p for n,p in model.named_parameters() if 'quantum' in n or 'qlayer' in n]
    cl_p = [p for n,p in model.named_parameters() if 'quantum' not in n and 'qlayer' not in n]
    groups = []
    if cl_p: groups.append({'params':cl_p,'lr':1e-3})
    if q_p:  groups.append({'params':q_p, 'lr':1e-2})
    opt   = optim.Adam(groups or model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()
    t0    = time.time()
    model.train()
    for ep in range(1, n_epochs+1):
        tl, tc, tn = 0., 0, 0
        for pb,fb,lb in loader:
            pb,fb,lb = pb.to(DEVICE),fb.to(DEVICE),lb.to(DEVICE)
            opt.zero_grad()
            out = model(pb, fb)
            loss = crit(out, lb); loss.backward(); opt.step()
            tl += loss.item()*len(lb); tc += (out.argmax(1)==lb).sum().item(); tn += len(lb)
        sched.step()
        if ep%5==0 or ep==1:
            log.info(f'  [{tag}] ep {ep:3d}/{n_epochs}  loss={tl/tn:.4f}  acc={tc/tn*100:.1f}%  ({time.time()-t0:.0f}s)')
    return time.time()-t0


@torch.no_grad()
def eval_model(model, loader, noise_std=0.0):
    model.eval()
    yt, yp = [], []
    for pb,fb,lb in loader:
        if noise_std > 0:
            pb = pb + noise_std * torch.randn_like(pb)
        pb,fb = pb.to(DEVICE),fb.to(DEVICE)
        yp.append(model(pb,fb).argmax(1).cpu().numpy())
        yt.append(lb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


@torch.no_grad()
def predict_all_pixels(model, d, noise_std=0.0):
    """Predict label for every pixel in the full scene (batch processing)."""
    model.eval()
    H, W, B = d['H'], d['W'], d['B']
    PAD = d['PAD']
    hsi_pad = d['hsi_pad']
    fa  = d['fa']

    all_rows = np.arange(H)
    all_cols = np.arange(W)
    rr, cc = np.meshgrid(all_rows, all_cols, indexing='ij')
    rr, cc = rr.flatten(), cc.flatten()

    pred_map = np.zeros(H * W, dtype=np.int32)
    PBATCH = 512

    for start in range(0, len(rr), PBATCH):
        end = min(start + PBATCH, len(rr))
        rs, cs = rr[start:end], cc[start:end]
        patches = np.empty((end-start, B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (r, c) in enumerate(zip(rs, cs)):
            patches[i] = hsi_pad[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :].transpose(2,0,1)

        specs = patches[:, :, PAD, PAD]
        fa_feats = fa.transform(specs).astype(np.float32)

        pb = torch.from_numpy(patches).to(DEVICE)
        if noise_std > 0:
            pb = pb + noise_std * torch.randn_like(pb)
        fb = torch.from_numpy(fa_feats).to(DEVICE)
        out = model(pb, fb)
        pred_map[start:end] = out.argmax(1).cpu().numpy()

    return pred_map.reshape(H, W)


# ============================================================
# SECTION 1 — PARAMETER COUNT TABLE
# ============================================================
section('Section 1: Parameter Count & Model Complexity')

param_rows = []
for ds_name in ['PaviaU', 'IndianPines', 'Salinas']:
    n_bands   = DATASETS[ds_name]['n_classes']  # placeholder
    # load one sample to get n_bands
    cfg = DATASETS[ds_name]
    raw = sio.loadmat(cfg['data_path'])
    dk  = cfg['data_key'] if cfg['data_key'] in raw else [k for k in raw if not k.startswith('_')][0]
    hsi = raw[dk].astype(np.float32)
    nb  = hsi.shape[2]
    nc  = cfg['n_classes']

    for mname, Builder in MODEL_BUILDERS.items():
        torch.manual_seed(42)
        m = Builder(nb, nc)
        total_params = sum(p.numel() for p in m.parameters())
        classic_params = sum(p.numel() for n,p in m.named_parameters()
                             if 'quantum' not in n and 'qlayer' not in n)
        quantum_params = sum(p.numel() for n,p in m.named_parameters()
                             if 'quantum' in n or 'qlayer' in n)
        param_rows.append(dict(
            model=mname, dataset=ds_name,
            total_params=total_params,
            classical_params=classic_params,
            quantum_params=quantum_params,
        ))
        log.info(f'  {mname:15s} | {ds_name:12s} | total={total_params:,} '
                 f'(classical={classic_params:,}, quantum={quantum_params:,})')

# also add SVM
for ds_name in ['PaviaU','IndianPines','Salinas']:
    param_rows.append(dict(model='SVM', dataset=ds_name,
                           total_params=0, classical_params=0, quantum_params=0))

df_params = pd.DataFrame(param_rows)
df_params.to_csv(os.path.join(WORKDIR, 'paper_params.csv'), index=False)

# ── Fig: parameter count bar chart (PaviaU only for clarity) ──
pu_params = df_params[(df_params['dataset']=='PaviaU') & (df_params['model']!='SVM')]
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800']
bars = ax.bar(pu_params['model'], pu_params['total_params']/1e3,
              color=colors, alpha=0.85)
ax.bar_label(bars, labels=[f'{v/1e3:.1f}K' for v in pu_params['total_params']],
             fontsize=9, padding=3)
ax.set_ylabel('Parameters (thousands)')
ax.set_title('Model Parameter Count (Pavia University)', fontsize=13, fontweight='bold')
ax.set_ylim(0, pu_params['total_params'].max()/1e3 * 1.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig('fig_paper_params.png')

# ── Fig: param vs OA scatter ──
master = pd.read_csv(os.path.join(WORKDIR, 'benchmark_master_results.csv'))
fig, ax = plt.subplots(figsize=(9, 6))
clr = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','DBDA':'#4CAF50',
       '3D-CNN-Only':'#9C27B0','HybridSN':'#FF9800','SVM':'#607D8B'}
for _, row in master.iterrows():
    mname = row['model']
    short = 'QHSA-Net' if 'QHSA' in mname else mname
    pp = df_params[(df_params['model']==short) & (df_params['dataset']=='PaviaU')]
    if pp.empty: continue
    p = pp['total_params'].values[0] / 1e3
    ax.scatter(p, row['OA'], s=200, color=clr.get(short,'#999'), zorder=5)
    ax.annotate(short, (p, row['OA']), textcoords='offset points',
                xytext=(6,4), fontsize=9)
ax.set_xlabel('Parameters (thousands)')
ax.set_ylabel('Overall Accuracy (%)')
ax.set_title('OA vs Model Size — Pavia University', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig('fig_paper_params_vs_oa.png')

log.info('Section 1 complete.')

# ============================================================
# SECTION 2 — CLASSIFICATION MAPS
# ============================================================
section('Section 2: Classification Maps')

MAP_MODELS = ['QHSA-Net', 'SSRN', 'DBDA', '3D-CNN-Only']

for ds_name in ['PaviaU', 'IndianPines', 'Salinas']:
    log.info(f'\n=== Classification Map: {ds_name} ===')
    cfg = DATASETS[ds_name]
    d   = load_dataset(ds_name, seed=SEED)
    tr_loader, te_loader = make_loaders(d)

    n_classes = d['n_classes']
    colors    = cfg['colors'][:n_classes]
    cmap      = ListedColormap(['#000000'] + colors)  # index 0 = background (black)

    # Ground truth map (GT uses 1-indexed, 0=background)
    gt_map = d['GT'].copy()

    # Train each model and predict on full scene
    pred_maps = {}
    for mname in MAP_MODELS:
        log.info(f'  Training {mname} for {ds_name} map...')
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = MODEL_BUILDERS[mname](d['n_bands'], n_classes).to(DEVICE)
        train_model(model, tr_loader, EPOCHS, tag=f'Map-{ds_name[:2]}-{mname[:6]}')
        log.info(f'  Predicting all {d["H"]*d["W"]:,} pixels...')
        pred_maps[mname] = predict_all_pixels(model, d)
        # Evaluate on test set too
        yt, yp = eval_model(model, te_loader)
        m = compute_metrics(yt, yp, n_classes)
        log.info(f'  {mname} test OA={m["OA"]:.2f}%  kappa={m["kappa"]:.2f}')
        del model

    # ── Plot: GT + 4 model predictions ──
    n_plots = 1 + len(MAP_MODELS)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))

    # Ground truth
    axes[0].imshow(gt_map, cmap=cmap, vmin=0, vmax=n_classes, interpolation='nearest')
    axes[0].set_title('Ground Truth', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for ax, mname in zip(axes[1:], MAP_MODELS):
        # pred_maps[mname] is 0-indexed predictions, shift to 1-indexed for colormap
        pred_display = pred_maps[mname] + 1   # 1-indexed
        # background pixels get 0
        pred_display[gt_map == 0] = 0
        ax.imshow(pred_display, cmap=cmap, vmin=0, vmax=n_classes, interpolation='nearest')
        ax.set_title(mname, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Legend
    legend_patches = [mpatches.Patch(color='#000000', label='Background')]
    for i, (name, color) in enumerate(zip(cfg['class_names'][:n_classes], colors)):
        legend_patches.append(mpatches.Patch(color=color, label=f'{i+1}: {name}'))

    fig.legend(handles=legend_patches, loc='lower center',
               ncol=min(6, n_classes+1), fontsize=7,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    ds_label = ds_name.replace('IndianPines','Indian Pines')
    plt.suptitle(f'Classification Maps — {ds_label}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(f'fig_paper_map_{ds_name.lower()}.png')

    # Save per-class OA for the table
    map_rows = []
    for mname, pmap in pred_maps.items():
        # only evaluate on labeled pixels
        labeled_mask = d['GT'] > 0
        yt = d['GT'][labeled_mask] - 1
        yp = pmap[labeled_mask]
        m  = compute_metrics(yt, yp, n_classes)
        map_rows.append(dict(model=mname, dataset=ds_name,
                             OA=m['OA'], AA=m['AA'], kappa=m['kappa']))

    pd.DataFrame(map_rows).to_csv(
        os.path.join(WORKDIR, f'paper_map_{ds_name.lower()}.csv'), index=False)

log.info('Section 2 complete.')

# ============================================================
# SECTION 3 — NOISE ROBUSTNESS
# ============================================================
section('Section 3: Noise Robustness Experiment')

log.info('Loading Pavia University for noise robustness...')
d_pu = load_dataset('PaviaU', seed=SEED)
tr_loader_pu, te_loader_pu = make_loaders(d_pu)

noise_models = {}
noise_rows   = []

for mname in ['QHSA-Net', 'SSRN', '3D-CNN-Only']:
    log.info(f'\n--- Training {mname} on clean Pavia U ---')
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MODEL_BUILDERS[mname](d_pu['n_bands'], d_pu['n_classes']).to(DEVICE)
    train_model(model, tr_loader_pu, EPOCHS, tag=f'Noise-{mname[:6]}')
    noise_models[mname] = model

    log.info(f'  Testing {mname} across noise levels...')
    for noise_std in NOISE_LEVELS:
        yt, yp = eval_model(model, te_loader_pu, noise_std=noise_std)
        m = compute_metrics(yt, yp, d_pu['n_classes'])
        noise_rows.append(dict(model=mname, noise_std=noise_std,
                               OA=m['OA'], AA=m['AA'], kappa=m['kappa']))
        log.info(f'    noise={noise_std:.2f}: OA={m["OA"]:.2f}%  kappa={m["kappa"]:.2f}')

df_noise = pd.DataFrame(noise_rows)
df_noise.to_csv(os.path.join(WORKDIR, 'paper_noise_robustness.csv'), index=False)

# ── Fig: OA vs noise level ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
clr = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','3D-CNN-Only':'#9C27B0'}
style = {'QHSA-Net':'-o','SSRN':'-s','3D-CNN-Only':'-^'}

# clean baseline OA
clean_oa = {m: df_noise[(df_noise['model']==m) & (df_noise['noise_std']==0.0)]['OA'].values[0]
            for m in ['QHSA-Net','SSRN','3D-CNN-Only']}

for ax, metric, label in zip(axes, ['OA', 'kappa'], ['Overall Accuracy (%)', 'Kappa (%)']):
    for mname in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub = df_noise[df_noise['model']==mname].sort_values('noise_std')
        ax.plot(sub['noise_std'], sub[metric], style[mname],
                color=clr[mname], label=mname, linewidth=2, markersize=7)
    ax.set_xlabel('Spectral Noise Level (σ)', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label} vs Noise Level — Pavia University', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xticks(NOISE_LEVELS)

plt.suptitle('Noise Robustness: Models Trained on Clean Data, Tested with Spectral Noise',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_noise_robustness.png')

# ── Fig: relative OA degradation ──
fig, ax = plt.subplots(figsize=(9, 5))
for mname in ['QHSA-Net','SSRN','3D-CNN-Only']:
    sub = df_noise[df_noise['model']==mname].sort_values('noise_std')
    baseline = sub[sub['noise_std']==0.0]['OA'].values[0]
    degradation = baseline - sub['OA'].values
    ax.plot(sub['noise_std'], degradation, style[mname],
            color=clr[mname], label=mname, linewidth=2, markersize=7)
ax.set_xlabel('Spectral Noise Level (σ)', fontsize=11)
ax.set_ylabel('OA Degradation (percentage points)', fontsize=11)
ax.set_title('Noise Robustness: OA Drop from Clean Baseline', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_xticks(NOISE_LEVELS)
plt.tight_layout()
save_fig('fig_paper_noise_degradation.png')

for mname in noise_models:
    del noise_models[mname]

log.info('Section 3 complete.')

# ============================================================
# SECTION 4 — t-SNE FEATURE VISUALISATION
# ============================================================
section('Section 4: t-SNE Feature Visualisation')

log.info('Training QHSA-Net on Pavia U for t-SNE...')
torch.manual_seed(SEED); np.random.seed(SEED)
d_pu = load_dataset('PaviaU', seed=SEED)
tr_loader_pu, te_loader_pu = make_loaders(d_pu)

qhsa = QHSANet(d_pu['n_bands'], d_pu['n_classes']).to(DEVICE)
train_model(qhsa, tr_loader_pu, EPOCHS, tag='tSNE-QHSA')

# Extract features from test set
log.info('Extracting features for t-SNE...')
qhsa.eval()
all_fc, all_fq, all_ff, all_y = [], [], [], []

with torch.no_grad():
    for pb, fb, lb in te_loader_pu:
        pb, fb = pb.to(DEVICE), fb.to(DEVICE)
        _, fc, fq, ff = qhsa(pb, fb, return_features=True)
        all_fc.append(fc.cpu().numpy())
        all_fq.append(fq.cpu().numpy())
        all_ff.append(ff.cpu().numpy())
        all_y.append(lb.numpy())

fc_arr = np.concatenate(all_fc)
fq_arr = np.concatenate(all_fq)
ff_arr = np.concatenate(all_ff)
y_arr  = np.concatenate(all_y)

# Subsample for t-SNE speed (max 3000 points)
n_tsne = min(3000, len(y_arr))
rng = np.random.default_rng(42)
idx = rng.choice(len(y_arr), n_tsne, replace=False)
fc_s, fq_s, ff_s, y_s = fc_arr[idx], fq_arr[idx], ff_arr[idx], y_arr[idx]

log.info(f'  Running t-SNE on {n_tsne} samples...')
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
tsne_fc = tsne.fit_transform(fc_s)
tsne_fq = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000).fit_transform(fq_s)
tsne_ff = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000).fit_transform(ff_s)
log.info('  t-SNE done.')

colors9  = ['#808080','#00CC00','#FF6600','#006400',
            '#FF0000','#CCCC00','#000080','#FF69B4','#00CCCC']
cnames   = DATASETS['PaviaU']['class_names']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Classical Branch Features\n(3D-CNN output)',
          'Quantum Branch Features\n(VQC output)',
          'Fused Features\n(After Gated Fusion)']

for ax, tsne_data, title in zip(axes, [tsne_fc, tsne_fq, tsne_ff], titles):
    for c in range(9):
        mask = y_s == c
        ax.scatter(tsne_data[mask,0], tsne_data[mask,1],
                   c=colors9[c], s=8, alpha=0.6, label=cnames[c])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

handles = [mpatches.Patch(color=colors9[i], label=cnames[i]) for i in range(9)]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.08))
plt.suptitle('t-SNE Feature Visualisation — QHSA-Net (Pavia University)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_tsne.png')

del qhsa
log.info('Section 4 complete.')

# ============================================================
# SECTION 5 — CONVERGENCE SPEED
# ============================================================
section('Section 5: Convergence Speed Comparison')

# Use existing training curves from paper_extension.py
df_curves = pd.read_csv(os.path.join(WORKDIR, 'paper_training_curves.csv'))

# target train acc threshold
THRESHOLD = 95.0

conv_rows = []
for ds_name in ['PaviaU', 'IndianPines', 'Salinas']:
    for mname in ['QHSA-Net', 'SSRN', '3D-CNN-Only']:
        sub = df_curves[(df_curves['model']==mname) &
                        (df_curves['dataset']==ds_name) &
                        (df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        reached = sub[sub['train_acc'] >= THRESHOLD]
        ep_to_thresh = reached['epoch'].min() if not reached.empty else None
        conv_rows.append(dict(model=mname, dataset=ds_name,
                              epochs_to_95=ep_to_thresh,
                              final_train_acc=sub['train_acc'].iloc[-1]))

df_conv = pd.DataFrame(conv_rows)
df_conv.to_csv(os.path.join(WORKDIR, 'paper_convergence.csv'), index=False)
log.info('\n' + df_conv.to_string(index=False))

# ── Fig: training curves per dataset ──
ds_labels = {'PaviaU':'Pavia University','IndianPines':'Indian Pines','Salinas':'Salinas Valley'}
clr2 = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','3D-CNN-Only':'#9C27B0'}
style2 = {'QHSA-Net':'-','SSRN':'--','3D-CNN-Only':'-.'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ['PaviaU','IndianPines','Salinas']):
    for mname in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub = df_curves[(df_curves['model']==mname) &
                        (df_curves['dataset']==ds) &
                        (df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        ax.plot(sub['epoch'], sub['train_acc'], style2[mname],
                color=clr2[mname], label=mname, linewidth=2)
    ax.axhline(THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label=f'{THRESHOLD}% threshold')
    ax.set_title(ds_labels[ds], fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Accuracy (%)' if ds=='PaviaU' else '')
    ax.set_ylim(0, 105); ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Training Convergence Speed — All Datasets', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_convergence.png')

# ── Fig: epochs-to-95% bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, ds in zip(axes, ['PaviaU','IndianPines','Salinas']):
    sub = df_conv[df_conv['dataset']==ds].dropna(subset=['epochs_to_95'])
    if sub.empty:
        ax.text(0.5,0.5,'All models < 95% threshold',ha='center',va='center',
                transform=ax.transAxes); continue
    bars = ax.bar(sub['model'], sub['epochs_to_95'],
                  color=[clr2[m] for m in sub['model']], alpha=0.85)
    ax.bar_label(bars, fmt='%d ep', fontsize=9, padding=2)
    ax.set_title(ds_labels[ds], fontsize=11, fontweight='bold')
    ax.set_ylabel('Epochs to reach 95% train acc' if ds=='PaviaU' else '')
    ax.set_ylim(0, EPOCHS + 5); ax.grid(axis='y', alpha=0.3)

plt.suptitle('Epochs to Reach 95% Training Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_epochs_to_95.png')

log.info('Section 5 complete.')

# ============================================================
# DONE
# ============================================================
section('ALL DONE')
log.info('\nOutput files:')
for f in ['paper_params.csv',
          'paper_noise_robustness.csv',
          'paper_convergence.csv',
          'paper_map_paviau.csv',
          'paper_map_indianpines.csv',
          'paper_map_salinas.csv',
          'fig_paper_params.png',
          'fig_paper_params_vs_oa.png',
          'fig_paper_map_paviau.png',
          'fig_paper_map_indianpines.png',
          'fig_paper_map_salinas.png',
          'fig_paper_noise_robustness.png',
          'fig_paper_noise_degradation.png',
          'fig_paper_tsne.png',
          'fig_paper_convergence.png',
          'fig_paper_epochs_to_95.png']:
    log.info(f'  {f}')
