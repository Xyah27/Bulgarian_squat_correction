import re
import numpy as np

# ---- Reglas (fallback) ----

def _angle(A, B, C, eps=1e-8):
    u = A - B; v = C - B
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    cosv = np.clip(np.dot(u, v) / (nu*nv + eps), -1, 1)
    return np.degrees(np.arccos(cosv))

def _trunk_angle(hip_mid, sh_mid):
    v = sh_mid - hip_mid
    v = v / (np.linalg.norm(v)+1e-8)
    cosv = np.clip(np.dot(v, np.array([0,-1])), -1, 1)
    return np.degrees(np.arccos(cosv))

def _idx2d(i): return 2*i, 2*i+1

def _per_frame_angles(seq_xy):
    T, F = seq_xy.shape
    out = {"knee_L": np.zeros(T), "knee_R": np.zeros(T), "trunk": np.zeros(T)}
    ids = {'lh':23,'rh':24,'lk':25,'rk':26,'la':27,'ra':28,'ls':11,'rs':12}
    for t in range(T):
        x = seq_xy[t]
        def p(i): a,b = _idx2d(i); return np.array([x[a], x[b]], dtype=float)
        hipL, hipR = p(ids['lh']), p(ids['rh'])
        kneeL, kneeR = p(ids['lk']), p(ids['rk'])
        anL, anR = p(ids['la']), p(ids['ra'])
        shL, shR = p(ids['ls']), p(ids['rs'])
        sh_mid = (shL + shR) / 2.0
        hip_mid = (hipL + hipR) / 2.0
        out["knee_L"][t] = _angle(hipL, kneeL, anL)
        out["knee_R"][t] = _angle(hipR, kneeR, anR)
        out["trunk"][t]  = _trunk_angle(hip_mid, sh_mid)
    return out

def _rule_based_labels(rep_matrix, rules=None):
    if rules is None:
        rules = {"trunk_incline_deg": 15.0, "knee_min_deg": 90.0}
    xy_dim = 33*2
    seq_xy = rep_matrix[:, :xy_dim]
    ang = _per_frame_angles(seq_xy)
    E1 = 1 if np.max(ang["trunk"]) > rules["trunk_incline_deg"] else 0
    E3 = 1 if min(np.min(ang["knee_L"]), np.min(ang["knee_R"])) > rules["knee_min_deg"] else 0
    E2 = 0
    correcta = 1 if (E1==0 and E2==0 and E3==0) else 0
    return np.array([correcta, E1, E2, E3], dtype=float)

# ---- Etiquetas CSV (si existieran) + fallback a reglas ----

_CANDIDATE_BIN = [
    "label","labels","class","clase","etiqueta","posture","pose","is_correct",
    "correct","correcta","incorrect","incorrecta","target","y"
]

def _find_label_col(df):
    cols = [c for c in df.columns]
    for c in cols:
        if c.lower() in _CANDIDATE_BIN:
            return c
    for c in cols:
        if re.search(r"(label|class|clase|etiquet|postur|pose|correct|incorrect|target|^y$)", c.lower()):
            return c
    return None

def _map_value_to_bin(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["incorrect","incorrecta","bad","mala","error","wrong","no","0","neg","negativo"]:
            return 1
        if s in ["correct","correcta","good","buena","ok","si","1","pos","positivo","right"]:
            return 0
        try:
            f = float(s); return 1 if f > 0.5 else 0
        except Exception:
            return None
    if isinstance(v, (int,float,np.integer,np.floating)):
        return 1 if v > 0.5 else 0
    return None

def build_labels_from_df_or_rules(df, rep_meta, video_col, reps_matrices, angle_rules=None):
    label_col = _find_label_col(df)
    if label_col is None:
        labels = np.array([_rule_based_labels(mat, rules=angle_rules) for mat in reps_matrices], dtype=float)
        return labels, ["correcta","E1_tronco","E2_valgo","E3_profundidad"], "rules"

    bin_labels = []
    for meta in rep_meta:
        vid = meta[video_col]; s = meta["start"]; e = meta["end"]
        sub = df[df[video_col] == vid].reset_index(drop=True)
        if len(sub)==0:
            bin_labels.append(None); continue
        e = min(e, len(sub)-1)
        vals = sub.loc[s:e, label_col].tolist()
        mapped = [ _map_value_to_bin(v) for v in vals if _map_value_to_bin(v) is not None ]
        if len(mapped)==0:
            bin_labels.append(None); continue
        ones = sum(mapped); zeros = len(mapped)-ones
        bin_labels.append(1 if ones >= zeros else 0)

    # Si demasiados None → reglas
    if sum(b is None for b in bin_labels) > 0.3*len(bin_labels):
        labels = np.array([_rule_based_labels(mat, rules=angle_rules) for mat in reps_matrices], dtype=float)
        return labels, ["correcta","E1_tronco","E2_valgo","E3_profundidad"], "rules"

    labels = []
    for i,b in enumerate(bin_labels):
        if b is None:
            vec = _rule_based_labels(reps_matrices[i], rules=angle_rules)
            corr = 1 if vec[0]==1 else 0; inc = 1-corr
        else:
            corr = 1 if b==0 else 0; inc = 1 if b==1 else 0
        labels.append([corr, inc])
    labels = np.array(labels, dtype=float)
    return labels, ["correcta","incorrecta"], "csv"
