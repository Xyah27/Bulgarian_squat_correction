import numpy as np

# ---- Utilidades de ángulos ----

def angle(A, B, C, eps=1e-8):
    u = A - B; v = C - B
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    cosv = np.clip(np.dot(u, v) / (nu*nv + eps), -1, 1)
    return np.degrees(np.arccos(cosv))

def trunk_angle(hip_mid, sh_mid):
    v = sh_mid - hip_mid
    v = v / (np.linalg.norm(v)+1e-8)
    cosv = np.clip(np.dot(v, np.array([0,-1])), -1, 1)
    return np.degrees(np.arccos(cosv))

def idx2d(i):  # índice MP -> columnas (x,y) en matriz [T, 33*2]
    return 2*i, 2*i+1

# ---- Cálculo frame a frame con landmarks 2D (asumiendo orden de MediaPipe) ----
# Si tu CSV no sigue el orden 0..32 de MediaPipe, ajusta el mapeo abajo.

def per_frame_angles(seq_xy):
    """
    seq_xy: [T, 33*2] (x,y) concatenados en orden de landmarks MediaPipe.
    Retorna:
      dict con arrays [T]: angulo_rodilla_L, angulo_rodilla_R, angulo_cadera_L, angulo_cadera_R, angulo_tronco
    """
    T, F = seq_xy.shape
    out = {
        "knee_L": np.zeros(T), "knee_R": np.zeros(T),
        "hip_L": np.zeros(T), "hip_R": np.zeros(T),
        "trunk": np.zeros(T)
    }
    ids = {'lh':23,'rh':24,'lk':25,'rk':26,'la':27,'ra':28,'ls':11,'rs':12}
    for t in range(T):
        x = seq_xy[t]
        def p(i): a,b = idx2d(i); return np.array([x[a], x[b]], dtype=float)
        hipL, hipR = p(ids['lh']), p(ids['rh'])
        kneeL, kneeR = p(ids['lk']), p(ids['rk'])
        anL, anR = p(ids['la']), p(ids['ra'])
        shL, shR = p(ids['ls']), p(ids['rs'])
        sh_mid = (shL + shR) / 2.0
        hip_mid = (hipL + hipR) / 2.0
        out["knee_L"][t] = angle(hipL, kneeL, anL)
        out["knee_R"][t] = angle(hipR, kneeR, anR)
        out["hip_L"][t]  = angle(shL, hipL, kneeL)
        out["hip_R"][t]  = angle(shR, hipR, kneeR)
        out["trunk"][t]  = trunk_angle(hip_mid, sh_mid)
    return out

def aggregate_rep_features(ang):
    """
    ang: dict de per_frame_angles
    Retorna vector de features agregadas por repetición.
    """
    rom_kneeL = np.min(ang["knee_L"])
    rom_kneeR = np.min(ang["knee_R"])
    trunk_mean = np.mean(ang["trunk"])
    trunk_max  = np.max(ang["trunk"])
    # suavidad: jerk medio de rodilla izquierda
    vel = np.gradient(ang["knee_L"])
    jerk = np.mean(np.abs(np.gradient(vel)))
    return np.array([rom_kneeL, rom_kneeR, trunk_mean, trunk_max, jerk], dtype=float)

def assemble_representation(rep_matrix):
    """
    Construye la representación final por repetición:
      X_seq: secuencia T×F (landmarks crudos 2D + opcionales),
      feats: vector agregado por repetición (angulares/ROM/suavidad)
    """
    # Asumimos rep_matrix = [T, 33*2 (+Z) (+vis)] pero como mínimo 33*2
    # Extraemos sólo XY para ángulos
    T, F = rep_matrix.shape
    # inferir si incluye z/vis: nos quedamos con primera parte 33*2
    xy_dim = 33*2
    if F < xy_dim:
        raise ValueError("La matriz de entrada no contiene 33*2 columnas de (x,y).")
    seq_xy = rep_matrix[:, :xy_dim]
    ang = per_frame_angles(seq_xy)
    feats = aggregate_rep_features(ang)
    return rep_matrix.astype(np.float32), feats.astype(np.float32)
