"""
src/rt_infer.py
Utilidades de inferencia en tiempo real:
- Captura de cámara + MediaPipe Pose
- Detector de repeticiones (FSM por ángulo de rodilla)
- Ensamblado de vector por frame (x,y,(z)) en orden 0..32
- Dibujado de overlay y esqueleto
- Inferencia de secuencia con el modelo Bi-GRU
"""

from collections import deque
import time
import numpy as np
import cv2
import torch

# === MediaPipe Pose ===
try:
    import mediapipe as mp
except ImportError:
    mp = None
    # El usuario debe instalar mediapipe.


# ---------- Captura y Pose ----------

class PoseStreamer:
    def __init__(self, cam_index=0, width=960, height=540):
        if mp is None:
            raise ImportError("Falta mediapipe. Instala con: pip install mediapipe")

        self.cap = cv2.VideoCapture(cam_index)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.pose_conn = mp.solutions.pose.POSE_CONNECTIONS

    def __iter__(self):
        return self._generator()

    def _generator(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espejo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            landmarks_norm = None
            if res.pose_landmarks and len(res.pose_landmarks.landmark) == 33:
                # Normalizados a [0..1] en x,y y z relativo
                lm = res.pose_landmarks.landmark
                landmarks_norm = np.array(
                    [[p.x, p.y, p.z] for p in lm],
                    dtype=np.float32
                )  # (33,3)

            yield frame, landmarks_norm

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass


# ---------- Geometría / Ángulos ----------

# Índices MediaPipe (oficiales)
LHIP, LKNEE, LANK = 23, 25, 27
RHIP, RKNEE, RANK = 24, 26, 28
LSHO, RSHO = 11, 12

def _angle_at(B, A, C):
    """
    Ángulo en A (en grados) dado triángulo ABC en 2D (x,y).
    """
    BA = B - A
    CA = C - A
    # producto punto y norma
    num = np.dot(BA, CA)
    den = np.linalg.norm(BA) * np.linalg.norm(CA) + 1e-8
    cosang = np.clip(num / den, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang

def knee_angles_deg(lm):
    """
    Devuelve (ang_left, ang_right) ángulo de rodilla en grados, usando (x,y).
    """
    if lm is None or lm.shape[0] < 29:
        return None, None
    L = lm[[LHIP, LKNEE, LANK], :2]
    R = lm[[RHIP, RKNEE, RANK], :2]
    angL = _angle_at(L[0], L[1], L[2])  # ang en rodilla izq
    angR = _angle_at(R[0], R[1], R[2])  # ang en rodilla der
    return angL, angR

def trunk_angle_forward_deg(lm):
    """
    Ángulo de tronco respecto a vertical (promedio hombros- caderas).
    """
    if lm is None or lm.shape[0] < 25:
        return None
    hip_mid = (lm[LHIP, :2] + lm[RHIP, :2]) / 2.0
    sho_mid = (lm[LSHO, :2] + lm[RSHO, :2]) / 2.0
    vec = sho_mid - hip_mid  # arriba
    # vertical "ideal": (0,-1). Usamos Y hacia abajo en imagen, por lo que el ang es con eje vertical.
    vec_norm = np.linalg.norm(vec) + 1e-8
    cos = (-vec[1]) / vec_norm  # componente vertical
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))  # 0° = tronco vertical; mayor = inclinado
    return ang


# ---------- Detector de Repeticiones ----------

class RepDetector:
    """
    FSM basada en ángulo de rodilla:
      - UP: rodilla extendida (>= up_thr)
      - GOING_DOWN: entre up_thr y down_thr
      - DOWN: flexión marcada (<= down_thr)
    Se cierra repetición al volver a UP tras pasar por DOWN.
    """
    def __init__(self, up_thr=160.0, down_thr=100.0, min_frames=24, max_frames=96):
        self.up_thr = up_thr
        self.down_thr = down_thr
        self.min_frames = min_frames
        self.max_frames = max_frames

        self.state = "UP"
        self.buffer = []  # guarda landmarks por frame
        self._was_down = False

        self.last_knee = (None, None)
        self.last_trunk = None

    def update(self, lm_norm):
        """
        Alimenta un frame. Devuelve:
          (rep_cerrada: bool, seq: list[landmarks] o None, angles_info: dict)
        """
        angL, angR = knee_angles_deg(lm_norm)
        trunk = trunk_angle_forward_deg(lm_norm)
        self.last_knee = (angL, angR)
        self.last_trunk = trunk

        # elegir la rodilla "activa" (la más flexionada)
        if angL is None or angR is None:
            active_knee = None
        else:
            active_knee = min(angL, angR)

        rep_closed = False
        seq = None

        # alimentar buffer
        if lm_norm is not None:
            self.buffer.append(lm_norm)
            if len(self.buffer) > self.max_frames:
                self.buffer = self.buffer[-self.max_frames:]

        # FSM
        if active_knee is None:
            return False, None, {"kneeL": angL, "kneeR": angR, "trunk": trunk}

        if self.state == "UP":
            if active_knee < self.up_thr:
                self.state = "GOING_DOWN"

        elif self.state == "GOING_DOWN":
            if active_knee <= self.down_thr:
                self.state = "DOWN"
                self._was_down = True
            elif active_knee >= self.up_thr:
                # no llegó a down: reset parcial pero mantenemos buffer
                self.state = "UP"
                self._was_down = False

        elif self.state == "DOWN":
            # subir cierra repetición
            if active_knee >= self.up_thr:
                if self._was_down and len(self.buffer) >= self.min_frames:
                    rep_closed = True
                    seq = self.buffer[:]  # copiar
                self.buffer = []  # reinicia para siguiente repetición
                self.state = "UP"
                self._was_down = False

        return rep_closed, seq, {"kneeL": angL, "kneeR": angR, "trunk": trunk}


# ---------- Ensamble de features y dibujo ----------

def assemble_frame_vector(lm_norm, use_z=False):
    """
    Convierte landmarks (33,3) normalizados en vector (F,)
    orden: [0_x..32_x, 0_y..32_y, (0_z..32_z si use_z)]
    """
    if lm_norm is None or lm_norm.shape != (33, 3):
        # vector nulo si no hay landmarks
        F = 33 * (2 + (1 if use_z else 0))
        return np.zeros((F,), dtype=np.float32)

    xs = lm_norm[:, 0]  # (33,)
    ys = lm_norm[:, 1]
    if use_z:
        zs = lm_norm[:, 2]
        vec = np.concatenate([xs, ys, zs], axis=0)
    else:
        vec = np.concatenate([xs, ys], axis=0)
    return vec.astype(np.float32)


def draw_pose_and_text(frame_bgr, lm_norm, state, fps, feedback_text, feedback_color):
    h, w = frame_bgr.shape[:2]

    # Pose simple (líneas básicas) si hay landmarks
    if lm_norm is not None and lm_norm.shape[0] == 33:
        # dibujar puntos
        for (x, y, z) in lm_norm:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 255), -1)

        # opcional: líneas principales del tren inferior (hip-knee-ankle)
        pairs = [(LHIP, LKNEE), (LKNEE, LANK), (RHIP, RKNEE), (RKNEE, RANK)]
        for a, b in pairs:
            ax, ay = int(lm_norm[a,0]*w), int(lm_norm[a,1]*h)
            bx, by = int(lm_norm[b,0]*w), int(lm_norm[b,1]*h)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 215, 255), 2)

    # Panel superior
    cv2.rectangle(frame_bgr, (0,0), (w, 58), (0,0,0), -1)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"State: {state}", (10,42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1, cv2.LINE_AA)

    # Feedback
    if feedback_text:
        cv2.putText(frame_bgr, feedback_text, (220, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, feedback_color, 2, cv2.LINE_AA)


# ---------- Inferencia ----------

def infer_rep_sequence(model, X_seq, thr, device=None):
    """
    X_seq: np.ndarray (T,F)
    thr: np.ndarray (K,)
    Devuelve (probs[K], preds[K]).

    Soporta forward(x, mask)  o  forward(x, lengths, mask)  o  forward(x).
    En live-stream no hay padding, así que 'mask' es todo unos.
    """
    import torch
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_seq).float().unsqueeze(0).to(device)   # (1,T,F)
        T = X_seq.shape[0]
        # >>> IMPORTANTE: máscara entera (no float) <<<
        mask = torch.ones((1, T), dtype=torch.long, device=device)    # (1,T)
        lengths = torch.tensor([T], dtype=torch.long, device=device)  # (1,)

        # Intentar firmas de forward comunes
        try:
            logits = model(x, mask)                    # forward(x, mask)
        except TypeError:
            try:
                logits = model(x, lengths, mask)       # forward(x, lengths, mask)
            except TypeError:
                logits = model(x)                      # forward(x)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    preds = (probs >= thr).astype(np.uint8)
    return probs, preds



