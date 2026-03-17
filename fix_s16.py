"""Fix Section 16 cell 42: replace eval_qhsa in _run_arch with a simple eval loop
that doesn't call model.classical(patches) directly."""
import json, pathlib

NB_PATH = pathlib.Path(r'c:\Users\saika\OneDrive\Desktop\test 6\QHSA_Net_Research_Notebook_2.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

OLD = '''def _run_arch(name, model_instance):
    _tr_l = DataLoader(HSIDataset(X_tr, Xpca_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(X_te, Xpca_te, y_te),
                       batch_size=256, shuffle=False, num_workers=0)
    t0 = time.time()
    train_qhsa(model_instance, _tr_l, EXP_EPOCHS_ARCH, name=name)
    _tt = time.time() - t0
    _yt, _yp, _, _, _ = eval_qhsa(model_instance, _te_l)
    _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
    return dict(config=name, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt)'''

NEW = '''def _eval_arch_simple(model, loader):
    """Simple eval that only calls model(patches, pca) without extracting intermediate features."""
    import torch, numpy as np
    model.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for _pb, _qb, _lb in loader:
            _pb = _pb.to(DEVICE); _qb = _qb.to(DEVICE)
            _out = model(_pb, _qb)
            yp_all.append(_out.argmax(1).cpu().numpy())
            yt_all.append(_lb.numpy())
    return np.concatenate(yt_all), np.concatenate(yp_all)

def _run_arch(name, model_instance):
    _tr_l = DataLoader(HSIDataset(X_tr, Xpca_tr, y_tr),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _te_l = DataLoader(HSIDataset(X_te, Xpca_te, y_te),
                       batch_size=256, shuffle=False, num_workers=0)
    t0 = time.time()
    train_qhsa(model_instance, _tr_l, EXP_EPOCHS_ARCH, name=name)
    _tt = time.time() - t0
    _yt, _yp = _eval_arch_simple(model_instance, _te_l)
    _oa, _aa, _kap, _, _ = compute_metrics(_yt, _yp)
    return dict(config=name, OA=_oa, AA=_aa, kappa=_kap, train_time_s=_tt)'''

cell = nb['cells'][42]
src = ''.join(cell['source'])
if OLD in src:
    new_src = src.replace(OLD, NEW)
    cell['source'] = [new_src]
    cell['outputs'] = []  # clear old outputs so it re-runs
    nb['cells'][42] = cell
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Fixed and saved Cell 42.')
else:
    print('ERROR: OLD string not found in Cell 42.')
    print('First 600 chars of cell source:')
    print(src[:600])
