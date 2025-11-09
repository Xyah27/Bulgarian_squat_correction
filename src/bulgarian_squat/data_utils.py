import os, re, glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

# ---------- Descarga o localización del dataset ----------

def download_or_locate_dataset():
    """
    Usa kagglehub si está disponible; si no, toma DATA_ROOT del entorno.
    """
    try:
        import kagglehub
        root = kagglehub.dataset_download("vivianpq/dataset-csv-aumentado")
        return root
    except Exception:
        env_root = os.environ.get("DATA_ROOT", None)
        if env_root is None:
            raise RuntimeError("No se pudo usar kagglehub y no hay DATA_ROOT. Define DATA_ROOT o instala kagglehub.")
        return env_root

# ---------- Elección de CSV maestro ----------

CANDIDATES_PRIOR = [
    "merged_dataset_etiquetado",   # preferido si existe
    "landmarks_dataset_balance",
    "landmarks_dataset_balanced",
    "merged_dataset"
]

def pick_master_csv(root):
    files = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*.csv"))]
    if not files:
        raise FileNotFoundError("No se encontraron CSVs en la carpeta del dataset.")
    # 1) Prioritarios
    for stem in CANDIDATES_PRIOR:
        for f in files:
            if stem in f.lower() and f.lower().endswith(".csv"):
                return os.path.join(root, f)
    # 2) Fallback: landmarks_original
    for f in files:
        if "landmarks_original" in f.lower() and f.lower().endswith(".csv"):
            return os.path.join(root, f)
    # 3) Último recurso: CSV más grande
    candidate = max(glob.glob(os.path.join(root,"*.csv")), key=os.path.getsize)
    return candidate

# ---------- Detección robusta de columnas ----------

def _extract_index(c):
    m = re.search(r'(\d+)$', c)
    if m: return int(m.group(1))
    m = re.search(r'_(\d+)_', c)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)', c)
    if m: return int(m.group(1))
    return None

def _group_landmark_cols(cols):
    x_map, y_map, z_map, v_map = {}, {}, {}, {}
    for c in cols:
        cl = c.lower()
        idx = _extract_index(cl)
        if idx is None: 
            continue
        if cl.startswith('x') or '_x' in cl or cl.endswith('_x'):
            x_map[idx] = c; continue
        if cl.startswith('y') or '_y' in cl or cl.endswith('_y'):
            y_map[idx] = c; continue
        if cl.startswith('z') or '_z' in cl or cl.endswith('_z'):
            z_map[idx] = c; continue
        if cl.startswith('vis') or cl.startswith('visibility') or '_vis' in cl or '_visibility' in cl:
            v_map[idx] = c; continue

    def _sorted_vals(m): return [m[k] for k in sorted(m.keys())]
    return _sorted_vals(x_map), _sorted_vals(y_map), _sorted_vals(z_map), _sorted_vals(v_map)

def load_and_clean(csv_path, min_vis_ratio=0.8):
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    x_cols, y_cols, z_cols, v_cols = _group_landmark_cols(cols)

    # filtrado por visibilidad si existe
    if v_cols:
        vis = df[v_cols].to_numpy(dtype=float)
        vis_ratio = (vis >= 0.5).sum(axis=1) / max(1, vis.shape[1])
        df = df[vis_ratio >= min_vis_ratio].reset_index(drop=True)

    df = df.replace([np.inf,-np.inf], np.nan).dropna(how='any')

    used = set(x_cols + y_cols + z_cols + v_cols)
    meta_cols = [c for c in cols if c not in used]
    return df, x_cols, y_cols, z_cols, v_cols, meta_cols

# ---------- Video ID ----------

def get_video_col(df):
    for c in ['video_id','video','source_video','file','filename','vid']:
        if c in df.columns:
            return c
    df['video_id'] = 'vid_0'
    return 'video_id'

# ---------- Segmentación de repeticiones ----------

def _find_y_index_col(y_cols, target_idx):
    for c in y_cols:
        idx = _extract_index(c.lower())
        if idx == target_idx:
            return c
    return None

def segment_reps(subdf, y_cols, fps=30):
    lhip_col = _find_y_index_col(y_cols, 23)
    rhip_col = _find_y_index_col(y_cols, 24)
    if lhip_col and rhip_col:
        y_l = subdf[lhip_col].to_numpy()
        y_r = subdf[rhip_col].to_numpy()
        y_signal = (y_l + y_r) / 2.0
    elif len(y_cols) > 0:
        y_signal = subdf[y_cols].mean(axis=1).to_numpy()
    else:
        raise RuntimeError("No se detectaron columnas Y de landmarks para segmentar repeticiones.")

    y_s = savgol_filter(y_signal, 9, 2, mode='interp')
    peaks, _ = find_peaks(-y_s, distance=int(0.5*fps))
    reps = []
    for i in range(len(peaks)-1):
        start = max(0, peaks[i] - int(0.4*fps))
        end   = min(len(subdf)-1, peaks[i+1] - int(0.4*fps))
        T = end - start + 1
        if int(1.0*fps) <= T <= int(6.0*fps):
            reps.append((start, end))
    return reps

def _sort_by_index(cols):
    pairs = []
    for c in cols:
        idx = _extract_index(c.lower())
        if idx is not None:
            pairs.append((idx, c))
    pairs.sort(key=lambda x: x[0])
    return [c for _,c in pairs]

def extract_xyvz_matrix(subdf, x_cols, y_cols, z_cols, v_cols):
    x_cols = _sort_by_index(x_cols)
    y_cols = _sort_by_index(y_cols)
    X = subdf[x_cols].to_numpy(dtype=float)
    Y = subdf[y_cols].to_numpy(dtype=float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Número de columnas X({X.shape[1]}) != Y({Y.shape[1]}).")
    T, N = X.shape
    XY = np.empty((T, 2*N), dtype=float)
    XY[:, 0::2] = X
    XY[:, 1::2] = Y

    parts = [XY]
    if z_cols:
        Z = subdf[_sort_by_index(z_cols)].to_numpy(dtype=float)
        parts.append(Z)
    if v_cols:
        V = subdf[_sort_by_index(v_cols)].to_numpy(dtype=float)
        parts.append(V)
    M = np.concatenate(parts, axis=1)
    return M

def build_repetitions_from_df(df, x_cols, y_cols, z_cols, v_cols, video_col='video_id', fps=30):
    reps, rep_meta = [], []
    for vid, sub in df.groupby(video_col, sort=False):
        sub = sub.reset_index(drop=True)
        segments = segment_reps(sub, y_cols, fps=fps)
        for (s,e) in segments:
            mat = extract_xyvz_matrix(sub.iloc[s:e+1], x_cols, y_cols, z_cols, v_cols)
            meta = {video_col: vid, "start": int(s), "end": int(e), "n_frames": int(e-s+1)}
            reps.append(mat)
            rep_meta.append(meta)
    return reps, rep_meta
