"""
Recovery script: runs only Section 4 (t-SNE) and Section 5 (convergence).
Sections 1-3 already completed and their outputs are saved.
"""
import os, sys, time, logging, warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pennylane as qml
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORKDIR     = r'c:/Users/saika/OneDrive/Desktop/test 6'
LOG_PATH    = os.path.join(WORKDIR, 'paper_additions_s4s5.log')
BEST_N_QUBITS = 4; BEST_N_LAYERS = 2; N_COMP = 4
PATCH_SIZE = 9; EPOCHS = 30; BATCH_SIZE = 64
DEVICE = torch.device('cpu'); SEED = 42

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_PATH,'w','utf-8'),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger()

def section(t): log.info(f'\n{"="*60}\n  {t}\n{"="*60}')
def save_fig(f):
    plt.savefig(os.path.join(WORKDIR,f), dpi=150, bbox_inches='tight')
    plt.close(); log.info(f'  Saved: {f}')

# ── Data loading ───────────────────────────────────────────────────────────────
def load_pavia(seed=42):
    raw = sio.loadmat(os.path.join(WORKDIR,'pavia u data','PaviaU.mat'))
    gt  = sio.loadmat(os.path.join(WORKDIR,'pavia u data','PaviaU_gt.mat'))
    HSI = raw['paviaU'].astype(np.float32)
    GT  = gt['paviaU_gt'].astype(np.int32)
    H,W,B = HSI.shape
    mn = HSI.min(axis=(0,1),keepdims=True); mx = HSI.max(axis=(0,1),keepdims=True)
    HSI = (HSI-mn)/(mx-mn+1e-8)
    rng = np.random.default_rng(seed)
    rows,cols = np.where(GT>0); labels = GT[rows,cols]-1
    tr_idx,te_idx = [],[]
    for c in range(9):
        cidx = np.where(labels==c)[0]
        n_tr = max(int(0.10*len(cidx)),3)
        perm = rng.permutation(len(cidx))
        tr_idx.extend(cidx[perm[:n_tr]].tolist())
        te_idx.extend(cidx[perm[n_tr:]].tolist())
    tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
    PAD = PATCH_SIZE//2
    hpad = np.pad(HSI,((PAD,PAD),(PAD,PAD),(0,0)),mode='reflect')
    def extr(ridx):
        out = np.empty((len(ridx),B,PATCH_SIZE,PATCH_SIZE),dtype=np.float32)
        for i,(r,c) in enumerate(zip(rows[ridx],cols[ridx])):
            out[i] = hpad[r:r+PATCH_SIZE,c:c+PATCH_SIZE,:].transpose(2,0,1)
        return out
    X_tr=extr(tr_idx); X_te=extr(te_idx)
    y_tr=labels[tr_idx].astype(np.int64); y_te=labels[te_idx].astype(np.int64)
    spec_tr=X_tr[:,: ,PAD,PAD]; spec_te=X_te[:,: ,PAD,PAD]
    fa = FactorAnalysis(n_components=N_COMP,random_state=42)
    fa_tr=fa.fit_transform(spec_tr).astype(np.float32)
    fa_te=fa.transform(spec_te).astype(np.float32)
    log.info(f'  PaviaU | train={len(y_tr)} test={len(y_te)}')
    return dict(X_tr=X_tr,X_te=X_te,y_tr=y_tr,y_te=y_te,
                fa_tr=fa_tr,fa_te=fa_te,n_bands=B,n_classes=9)

class HSIDataset(Dataset):
    def __init__(self,p,f,l):
        self.p=torch.from_numpy(p); self.f=torch.from_numpy(f); self.l=torch.from_numpy(l).long()
    def __len__(self): return len(self.l)
    def __getitem__(self,i): return self.p[i],self.f[i],self.l[i]

def make_loaders(d):
    tr=DataLoader(HSIDataset(d['X_tr'],d['fa_tr'],d['y_tr']),BATCH_SIZE,shuffle=True,num_workers=0)
    te=DataLoader(HSIDataset(d['X_te'],d['fa_te'],d['y_te']),256,shuffle=False,num_workers=0)
    return tr,te

# ── Models ─────────────────────────────────────────────────────────────────────
def make_vqc(nq,nl):
    dev=qml.device('default.qubit',wires=nq)
    ws=qml.StronglyEntanglingLayers.shape(n_layers=nl,n_wires=nq)
    @qml.qnode(dev,interface='torch',diff_method='backprop')
    def circuit(inputs,weights):
        qml.AngleEmbedding(inputs,wires=range(nq))
        qml.StronglyEntanglingLayers(weights,wires=range(nq))
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)]
    return circuit,{'weights':ws},nq

class QuantumBranch(nn.Module):
    def __init__(self,nq=4,nl=2,pd=64):
        super().__init__()
        c,ws,od=make_vqc(nq,nl)
        self.ql=qml.qnn.TorchLayer(c,ws)
        self.proj=nn.Sequential(nn.Linear(od,pd),nn.LayerNorm(pd))
    def forward(self,x):
        x=torch.tanh(x)*np.pi; q=self.ql(x)
        q=torch.softmax(q,dim=-1)
        if isinstance(q,(list,tuple)): q=torch.stack(q,dim=-1)
        return self.proj(q)

class ClassicalBranch(nn.Module):
    def __init__(self,nb,pd=64):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(1,8,(7,3,3),padding=(3,1,1)),nn.BatchNorm3d(8),nn.ReLU(),
            nn.Conv3d(8,16,(5,3,3),padding=(2,1,1)),nn.BatchNorm3d(16),nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)),nn.BatchNorm3d(32),nn.ReLU())
        dummy=torch.zeros(1,1,nb,PATCH_SIZE,PATCH_SIZE)
        fl=self.conv(dummy).flatten(1).shape[1]
        self.fc=nn.Sequential(nn.Linear(fl,pd),nn.LayerNorm(pd))
    def forward(self,x): return self.fc(self.conv(x.unsqueeze(1)).flatten(1))

class GatedFusion(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
    def forward(self,fc,fq):
        a=self.gate(torch.cat([fc,fq],dim=-1)); return a*fq+(1-a)*fc

class QHSANet(nn.Module):
    def __init__(self,nb,nc,pd=64):
        super().__init__()
        self.classical=ClassicalBranch(nb,pd)
        self.quantum=QuantumBranch(4,2,pd)
        self.fusion=GatedFusion(pd)
        self.clf=nn.Sequential(nn.Linear(pd,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,nc))
    def forward(self,p,f,ret_feat=False):
        fc=self.classical(p); fq=self.quantum(f); ff=self.fusion(fc,fq)
        if ret_feat: return self.clf(ff),fc,fq,ff
        return self.clf(ff)

def train_model(model,loader,n_epochs,tag=''):
    qp=[p for n,p in model.named_parameters() if 'quantum' in n or 'qlayer' in n]
    cp=[p for n,p in model.named_parameters() if 'quantum' not in n and 'qlayer' not in n]
    groups=[]
    if cp: groups.append({'params':cp,'lr':1e-3})
    if qp: groups.append({'params':qp,'lr':1e-2})
    opt=optim.Adam(groups or model.parameters(),lr=1e-3)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_epochs)
    crit=nn.CrossEntropyLoss(); t0=time.time(); model.train()
    for ep in range(1,n_epochs+1):
        tl,tc,tn=0.,0,0
        for pb,fb,lb in loader:
            pb,fb,lb=pb.to(DEVICE),fb.to(DEVICE),lb.to(DEVICE)
            opt.zero_grad(); out=model(pb,fb); loss=crit(out,lb)
            loss.backward(); opt.step()
            tl+=loss.item()*len(lb); tc+=(out.argmax(1)==lb).sum().item(); tn+=len(lb)
        sched.step()
        if ep%5==0 or ep==1:
            log.info(f'  [{tag}] ep {ep:3d}/{n_epochs}  loss={tl/tn:.4f}  acc={tc/tn*100:.1f}%  ({time.time()-t0:.0f}s)')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
section('Section 4: t-SNE Feature Visualisation')

d = load_pavia(SEED)
tr_loader, te_loader = make_loaders(d)

log.info('Training QHSA-Net for t-SNE...')
torch.manual_seed(SEED); np.random.seed(SEED)
qhsa = QHSANet(d['n_bands'], d['n_classes']).to(DEVICE)
train_model(qhsa, tr_loader, EPOCHS, tag='tSNE-QHSA')

log.info('Extracting features...')
qhsa.eval()
all_fc,all_fq,all_ff,all_y = [],[],[],[]
with torch.no_grad():
    for pb,fb,lb in te_loader:
        pb,fb=pb.to(DEVICE),fb.to(DEVICE)
        _,fc,fq,ff=qhsa(pb,fb,ret_feat=True)
        all_fc.append(fc.cpu().numpy()); all_fq.append(fq.cpu().numpy())
        all_ff.append(ff.cpu().numpy()); all_y.append(lb.numpy())

fc_arr=np.concatenate(all_fc); fq_arr=np.concatenate(all_fq)
ff_arr=np.concatenate(all_ff); y_arr=np.concatenate(all_y)

n_tsne=min(3000,len(y_arr))
rng=np.random.default_rng(42); idx=rng.choice(len(y_arr),n_tsne,replace=False)
fc_s,fq_s,ff_s,y_s = fc_arr[idx],fq_arr[idx],ff_arr[idx],y_arr[idx]

log.info(f'  Running t-SNE on {n_tsne} points (3 runs)...')
tsne_fc = TSNE(n_components=2,random_state=42,perplexity=40,n_iter=1000).fit_transform(fc_s)
tsne_fq = TSNE(n_components=2,random_state=42,perplexity=40,n_iter=1000).fit_transform(fq_s)
tsne_ff = TSNE(n_components=2,random_state=42,perplexity=40,n_iter=1000).fit_transform(ff_s)
log.info('  t-SNE complete.')

colors9 = ['#808080','#00CC00','#FF6600','#006400',
           '#FF0000','#CCCC00','#000080','#FF69B4','#00CCCC']
cnames  = ['Asphalt','Meadows','Gravel','Trees','Painted metal',
           'Bare soil','Bitumen','Bricks','Shadows']

fig,axes = plt.subplots(1,3,figsize=(18,5))
titles = ['Classical Branch\n(3D-CNN output)',
          'Quantum Branch\n(VQC output)',
          'After Gated Fusion\n(Final features)']
for ax,data,title in zip(axes,[tsne_fc,tsne_fq,tsne_ff],titles):
    for c in range(9):
        mask=y_s==c
        ax.scatter(data[mask,0],data[mask,1],c=colors9[c],s=8,alpha=0.6,label=cnames[c])
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

handles=[mpatches.Patch(color=colors9[i],label=cnames[i]) for i in range(9)]
fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(0.5,-0.06))
plt.suptitle('t-SNE Feature Visualisation — QHSA-Net Internals (Pavia University)',
             fontsize=13,fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_tsne.png')
del qhsa
log.info('Section 4 complete.')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONVERGENCE SPEED
# ══════════════════════════════════════════════════════════════════════════════
section('Section 5: Convergence Speed')

df_curves = pd.read_csv(os.path.join(WORKDIR,'paper_training_curves.csv'))
THRESHOLD = 95.0
ds_labels = {'PaviaU':'Pavia University','IndianPines':'Indian Pines','Salinas':'Salinas Valley'}
clr = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','3D-CNN-Only':'#9C27B0'}
sty = {'QHSA-Net':'-','SSRN':'--','3D-CNN-Only':'-.'}

conv_rows=[]
for ds in ['PaviaU','IndianPines','Salinas']:
    for m in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub=df_curves[(df_curves['model']==m)&(df_curves['dataset']==ds)&
                      (df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        reached=sub[sub['train_acc']>=THRESHOLD]
        conv_rows.append(dict(model=m,dataset=ds,
                              epochs_to_95=reached['epoch'].min() if not reached.empty else None,
                              final_train_acc=sub['train_acc'].iloc[-1]))

df_conv=pd.DataFrame(conv_rows)
df_conv.to_csv(os.path.join(WORKDIR,'paper_convergence.csv'),index=False)
log.info('\n'+df_conv.to_string(index=False))

# training curves plot
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,ds in zip(axes,['PaviaU','IndianPines','Salinas']):
    for m in ['QHSA-Net','SSRN','3D-CNN-Only']:
        sub=df_curves[(df_curves['model']==m)&(df_curves['dataset']==ds)&
                      (df_curves['seed']==42)].sort_values('epoch')
        if sub.empty: continue
        ax.plot(sub['epoch'],sub['train_acc'],sty[m],color=clr[m],label=m,linewidth=2)
    ax.axhline(THRESHOLD,color='gray',linestyle=':',linewidth=1,alpha=0.7)
    ax.set_title(ds_labels[ds],fontsize=11,fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Accuracy (%)' if ds=='PaviaU' else '')
    ax.set_ylim(0,105); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('Training Convergence Speed',fontsize=13,fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_convergence.png')

# epochs-to-95 bar chart
fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,ds in zip(axes,['PaviaU','IndianPines','Salinas']):
    sub=df_conv[df_conv['dataset']==ds].dropna(subset=['epochs_to_95'])
    if sub.empty:
        ax.text(0.5,0.5,'All < 95%',ha='center',va='center',transform=ax.transAxes); continue
    bars=ax.bar(sub['model'],sub['epochs_to_95'],
                color=[clr[m] for m in sub['model']],alpha=0.85)
    ax.bar_label(bars,fmt='%d ep',fontsize=9,padding=2)
    ax.set_title(ds_labels[ds],fontsize=11,fontweight='bold')
    ax.set_ylabel('Epochs to 95% train acc' if ds=='PaviaU' else '')
    ax.set_ylim(0,35); ax.grid(axis='y',alpha=0.3)
plt.suptitle('Epochs to Reach 95% Training Accuracy',fontsize=13,fontweight='bold')
plt.tight_layout()
save_fig('fig_paper_epochs_to_95.png')

log.info('Section 5 complete.')
section('ALL DONE — Sections 4 and 5 complete.')
log.info('Saved: fig_paper_tsne.png, fig_paper_convergence.png, fig_paper_epochs_to_95.png, paper_convergence.csv')
