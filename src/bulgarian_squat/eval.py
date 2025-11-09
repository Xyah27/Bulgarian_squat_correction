import time
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, average_precision_score
)

def collect_probs(model, loader, device='cuda'):
    model.eval()
    probs_all, y_all, times = [], [], []
    with torch.no_grad():
        for X,Y,M in loader:
            t0 = time.time()
            X,Y,M = X.to(device), Y.to(device), M.to(device)
            logits = model(X,M)
            p = torch.sigmoid(logits).cpu().numpy()
            dt = (time.time()-t0)*1000.0
            probs_all.append(p); y_all.append(Y.cpu().numpy())
            times.append(dt)
    P = np.concatenate(probs_all); Y = np.concatenate(y_all)
    return P, Y

def tune_thresholds(P_val, Y_val, steps=81):
    thrs = np.linspace(0.1,0.9,steps)
    best = []
    K = Y_val.shape[1]
    for k in range(K):
        best_f1, bt = -1, 0.5
        for t in thrs:
            f1 = f1_score(Y_val[:,k], (P_val[:,k]>=t).astype(int), zero_division=0)
            if f1>best_f1:
                best_f1, bt = f1, t
        best.append(bt)
    return np.array(best, dtype=float)

def evaluate_report(P, Y, thr):
    Yhat = (P >= thr).astype(int)
    report = classification_report(Y, Yhat, zero_division=0, target_names=[f"C{k}" for k in range(Y.shape[1])])
    try:
        auroc = roc_auc_score(Y, P, average='macro')
    except ValueError:
        auroc = float('nan')
    aupr  = average_precision_score(Y, P, average='macro')
    macroF1 = f1_score(Y, Yhat, average='macro', zero_division=0)
    lat_median, lat_p95 = 0.0, 0.0  # latencia medida en collect_probs si se adapta
    return report, auroc, aupr, lat_median, lat_p95, macroF1

def bootstrap_macro_f1(P, Y, thr, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    vals = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        Yb = Y[idx]; Pb = P[idx]
        Yhb = (Pb >= thr).astype(int)
        f1 = f1_score(Yb, Yhb, average='macro', zero_division=0)
        vals.append(f1)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(vals.mean()), (float(lo), float(hi))
