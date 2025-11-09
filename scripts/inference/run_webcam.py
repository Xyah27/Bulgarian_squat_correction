"""
Script de inferencia en tiempo real con webcam
==============================================

Detecta y clasifica repeticiones de Bulgarian Split Squat en tiempo real.

Uso:
    python scripts/inference/run_webcam.py --model models/best/best_model_bigru.pt --cam 0
    
Opciones:
    --model: Ruta al modelo entrenado
    --cam: Índice de la cámara (default: 0)
    --minlen: Mínimo de frames por repetición (default: 20)
    --maxlen: Máximo de frames por repetición (default: 90)
"""

import sys
from pathlib import Path

# Añadir src al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import cv2
import torch
import numpy as np
import mediapipe as mp
import json
import argparse

from src.bulgarian_squat.model_improved import BiGRUClassifierImproved


def load_model(model_dir):
    """Cargar modelo y metadatos"""
    model_dir = Path(model_dir)
    
    # Cargar metadatos
    with open(model_dir / "run_meta.json", "r") as f:
        meta = json.load(f)
    
    # Cargar nombres de clases
    with open(model_dir / "class_names.json", "r") as f:
        class_names_dict = json.load(f)
    class_names = [class_names_dict[str(i)] for i in range(len(class_names_dict))]
    
    # Cargar umbrales
    thresholds = np.load(model_dir / "thr_per_class.npy")
    
    # Crear modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiGRUClassifierImproved(
        in_dim=meta["in_dim"],
        num_classes=meta["num_classes"],
        hidden1=meta.get("hidden_dim1", 128),
        hidden2=meta.get("hidden_dim2", 64),
        dropout=meta.get("dropout", 0.3),
        use_batch_norm=True,
        use_attention=meta.get("use_attention", True)
    ).to(device)
    
    # Cargar pesos
    state = torch.load(model_dir / "best_model_bigru.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    print(f"✓ Modelo cargado: {meta.get('model_type', 'BiGRU+Attention')}")
    print(f"✓ Dispositivo: {device}")
    print(f"✓ Clases: {class_names}")
    print(f"✓ Umbrales: {thresholds}")
    
    return model, device, class_names, thresholds, meta


def extract_landmarks(results, include_z=False):
    """Extraer landmarks normalizados de MediaPipe"""
    if not results.pose_landmarks or len(results.pose_landmarks.landmark) != 33:
        return None
    
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        if include_z:
            landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.append([lm.x, lm.y])
    
    return np.array(landmarks, dtype=np.float32).flatten()


def calculate_knee_angle(landmarks_2d, side='left'):
    """Calcular ángulo de rodilla"""
    if landmarks_2d is None:
        return None
    
    try:
        lm = landmarks_2d.reshape(33, 2)
        
        if side == 'left':
            hip, knee, ankle = lm[23], lm[25], lm[27]
        else:
            hip, knee, ankle = lm[24], lm[26], lm[28]
        
        v1 = hip - knee
        v2 = ankle - knee
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle
    except:
        return None


def detect_view_and_leg(results, debug=False):
    """Detectar vista (frontal/lateral) y pierna activa"""
    if not results.pose_landmarks:
        return 'lateral', 'left', {}
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder, right_shoulder = landmarks[11], landmarks[12]
        left_hip, right_hip = landmarks[23], landmarks[24]
        
        shoulder_width_x = abs(left_shoulder.x - right_shoulder.x)
        hip_width_x = abs(left_hip.x - right_hip.x)
        
        shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
        hip_depth_diff = abs(left_hip.z - right_hip.z)
        
        avg_width = (shoulder_width_x + hip_width_x) / 2
        avg_depth_diff = (shoulder_depth_diff + hip_depth_diff) / 2
        
        is_frontal = (avg_width > 0.20) and (avg_depth_diff < 0.15)
        
        debug_info = {
            'avg_width': avg_width,
            'avg_depth_diff': avg_depth_diff,
            'is_frontal': is_frontal
        } if debug else {}
        
        if is_frontal:
            landmarks_2d = []
            for lm in landmarks:
                landmarks_2d.append([lm.x, lm.y])
            landmarks_2d = np.array(landmarks_2d, dtype=np.float32).flatten()
            
            left_angle = calculate_knee_angle(landmarks_2d, 'left')
            right_angle = calculate_knee_angle(landmarks_2d, 'right')
            
            if left_angle and right_angle:
                active_leg = 'left' if left_angle < right_angle else 'right'
            else:
                active_leg = 'left'
            
            return 'frontal', active_leg, debug_info
        else:
            left_avg_z = (left_shoulder.z + left_hip.z) / 2
            right_avg_z = (right_shoulder.z + right_hip.z) / 2
            active_leg = 'left' if left_avg_z < right_avg_z else 'right'
            
            return 'lateral', active_leg, debug_info
            
    except Exception as e:
        return 'lateral', 'left', {}


def infer_sequence(model, sequence, thresholds, device):
    """Inferir clasificación de una secuencia"""
    if len(sequence) < 20:
        return None, None
    
    X = np.array(sequence)
    x_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
    
    T = X.shape[0]
    mask = torch.ones((1, T), dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(x_tensor, mask)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    
    predictions = (probs >= thresholds).astype(int)
    
    return probs, predictions


def main():
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real con webcam")
    parser.add_argument("--model", type=str, default="models/best",
                       help="Directorio del modelo")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--minlen", type=int, default=20, help="Mínimo frames")
    parser.add_argument("--maxlen", type=int, default=90, help="Máximo frames")
    
    args = parser.parse_args()
    
    print("\n=== Bulgarian Split Squat - Análisis en Tiempo Real ===\n")
    
    # Cargar modelo
    model, device, class_names, thresholds, meta = load_model(args.model)
    
    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Inicializar cámara
    print(f"Iniciando cámara {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la cámara {args.cam}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"✓ Cámara {args.cam} iniciada")
    print("\n📋 Instrucciones:")
    print("  • Colócate en vista LATERAL o FRONTAL")
    print("  • Realiza movimientos LENTOS Y CONTROLADOS")
    print(f"  • Mínimo {args.minlen} frames por repetición")
    print("\n🎮 Controles:")
    print("  D: Modo debug on/off")
    print("  ESPACIO: Captura manual")
    print("  Q o ESC: Salir")
    print("\n" + "="*50)
    
    # Estado
    capturing = False
    sequence = []
    last_result = ""
    state = "ESPERANDO"
    knee_angle = 0
    view_type = "lateral"
    active_leg = "left"
    debug_mode = False
    debug_info = {}
    
    # FSM
    fsm_state = "WAIT"
    threshold_down = 130
    threshold_up = 155
    min_down_angle = 180
    frame_counter = 0
    min_frames_down = 8
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                landmarks = extract_landmarks(results, include_z=False)
                
                if landmarks is not None:
                    view_type, active_leg, debug_info = detect_view_and_leg(results, debug=debug_mode)
                    knee_angle = calculate_knee_angle(landmarks, active_leg) or 0
                    
                    # FSM
                    if fsm_state == "WAIT":
                        min_down_angle = 180
                        frame_counter = 0
                        if knee_angle < threshold_down:
                            fsm_state = "DOWN"
                            capturing = True
                            sequence = []
                            state = "⬇️ BAJANDO"
                            print(f"\n▶ Repetición iniciada (vista: {view_type}, pierna: {active_leg})")
                    
                    elif fsm_state == "DOWN":
                        frame_counter += 1
                        min_down_angle = min(min_down_angle, knee_angle)
                        
                        if knee_angle > min_down_angle + 8 and frame_counter > min_frames_down:
                            fsm_state = "UP"
                            state = "⬆️ SUBIENDO"
                            frame_counter = 0
                    
                    elif fsm_state == "UP":
                        frame_counter += 1
                        
                        if knee_angle > threshold_up or frame_counter > 20:
                            fsm_state = "WAIT"
                            capturing = False
                            state = "🔍 ANALIZANDO..."
                            
                            if len(sequence) >= args.minlen:
                                probs, preds = infer_sequence(model, sequence, thresholds, device)
                                
                                if probs is not None:
                                    results_text = []
                                    for i, (name, prob, pred) in enumerate(zip(class_names, probs, preds)):
                                        if pred == 1:
                                            results_text.append(f"{name}: {prob:.2f}")
                                    
                                    if results_text:
                                        last_result = " | ".join(results_text)
                                    else:
                                        max_idx = np.argmax(probs)
                                        last_result = f"{class_names[max_idx]}: {probs[max_idx]:.2f}"
                                    
                                    print(f"✓ Analizada ({len(sequence)} frames, ángulo mín: {min_down_angle:.1f}°)")
                                    print(f"  Resultado: {last_result}")
                            else:
                                print(f"⚠ Muy pocos frames: {len(sequence)} (mín: {args.minlen})")
                            
                            state = "✅ COMPLETO"
                    
                    if capturing and landmarks is not None:
                        sequence.append(landmarks)
            
            # Overlay
            h, w = frame.shape[:2]
            panel_height = 160 if debug_mode else 120
            cv2.rectangle(frame, (0, 0), (w, panel_height), (0, 0, 0), -1)
            
            cv2.putText(frame, f"Estado: {state}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            view_text = f"Vista: {view_type.upper()} | Pierna: {active_leg.upper()}"
            if debug_mode and debug_info:
                view_text += f" [W:{debug_info.get('avg_width', 0):.3f} D:{debug_info.get('avg_depth_diff', 0):.3f}]"
            cv2.putText(frame, view_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            angle_color = (0, 255, 0) if knee_angle > threshold_down else (0, 165, 255)
            cv2.putText(frame, f"Angulo: {knee_angle:.1f}° (min: {min_down_angle:.1f}°)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
            
            if capturing:
                cv2.putText(frame, f"Frames: {len(sequence)}", (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if debug_mode:
                cv2.putText(frame, f"Umbrales: W>0.20 y D<0.15 = FRONTAL", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            if last_result:
                cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, f"Resultado: {last_result}", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Bulgarian Split Squat - Tiempo Real", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"\n{'✓' if debug_mode else '✗'} Debug: {'ON' if debug_mode else 'OFF'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n✓ Cámara cerrada")


if __name__ == "__main__":
    main()
