# src/train_hist.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

def _compute_pos_weight(loader, device):
    Ys = []
    for X, Y, M in loader:
        Ys.append(Y)
    Y = torch.cat(Ys, dim=0).numpy()
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    pw = (neg / (pos + 1e-6)).astype(np.float32)
    return torch.tensor(pw, dtype=torch.float32, device=device)

def train_with_history(model, loaders, epochs=100, lr=1e-3, device='cuda'):
    """
    Igual que train_bigru pero guarda historia de Macro-F1 val por época.
    Devuelve: model (mejor), history (lista de floats).
    """
    model.to(device)
    pos_weight = _compute_pos_weight(loaders['train'], device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state = -1, None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        for X, Y, M in loaders['train']:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            opt.zero_grad()
            logits = model(X, M)
            loss = criterion(logits, Y)
            loss.backward()
            opt.step()

        # validación
        model.eval(); preds = []; gts = []
        with torch.no_grad():
            for X, Y, M in loaders['val']:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                probs = torch.sigmoid(model(X, M)).cpu().numpy()
                preds.append(probs); gts.append(Y.cpu().numpy())
        P = np.concatenate(preds); Yv = np.concatenate(gts)
        f1 = f1_score(Yv, (P >= 0.5).astype(int), average='macro', zero_division=0)
        history.append(float(f1))
        print(f"[{ep:03d}] val Macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state = f1, model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
