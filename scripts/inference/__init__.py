"""
scripts.inference
=================

Módulo de scripts de inferencia para el proyecto Bulgarian Split Squat.

Este módulo contiene scripts para realizar inferencia en tiempo real y
batch usando modelos entrenados de clasificación de ejercicios.

Módulos disponibles:
-------------------
- run_webcam: Inferencia en tiempo real desde webcam
- batch_inference: Inferencia por lotes en videos
- realtime_feedback: Sistema de retroalimentación en tiempo real

Funcionalidades:
---------------
- Captura de video desde webcam o archivo
- Estimación de pose con MediaPipe
- Clasificación de errores en tiempo real
- Visualización de landmarks y predicciones
- Generación de reportes de sesión

Ejemplo de uso:
--------------
    # Desde línea de comandos:
    python scripts/inference/run_webcam.py --model models/best/bigru_best.pth
    
    # O importar en código:
    from scripts.inference import run_webcam
    
Requisitos:
----------
- MediaPipe instalado
- OpenCV (cv2)
- Modelo entrenado (.pth)
- Webcam o archivo de video

Salidas:
-------
- Visualización en tiempo real
- Estadísticas de sesión
- Videos procesados (opcional)
- Reportes JSON con métricas

Autor: Juan Jose Núñez, Juan Jose Castro
Institución: Universidad San Buenaventura
"""

__version__ = "1.0.0"
__author__ = "Juan Jose Núñez, Juan Jose Castro"

# Importar funciones principales
__all__ = [
    'run_webcam',
    'batch_inference',
    'realtime_feedback',
]

# Configuración por defecto
DEFAULT_INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'visibility_threshold': 0.5,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'fps': 30,
    'display_width': 1280,
    'display_height': 720,
}

# Nombres de clases
CLASS_NAMES = [
    "E0_correcta",
    "E1_inclinacion_tronco", 
    "E2_valgo_rodilla",
    "E3_profundidad_insuficiente"
]

# Colores para visualización (BGR)
CLASS_COLORS = {
    "E0_correcta": (0, 255, 0),           # Verde
    "E1_inclinacion_tronco": (0, 165, 255),  # Naranja
    "E2_valgo_rodilla": (0, 0, 255),      # Rojo
    "E3_profundidad_insuficiente": (255, 0, 255),  # Magenta
}

# Mensajes de retroalimentación
FEEDBACK_MESSAGES = {
    "E0_correcta": "¡Excelente técnica! Mantén la forma.",
    "E1_inclinacion_tronco": "⚠️ Mantén el tronco más vertical",
    "E2_valgo_rodilla": "⚠️ Evita que la rodilla colapse hacia adentro",
    "E3_profundidad_insuficiente": "⚠️ Baja más en la sentadilla",
}
