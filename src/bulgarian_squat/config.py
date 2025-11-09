# Configuración global

SEED = 42

# Visibilidad mínima de landmarks por frame (proporción de 33 puntos con vis>=0.5)
MIN_VIS_RATIO = 0.80

# Usar z o visibilidad en la entrada (si están en el CSV)
USE_Z = False
USE_VIS = True

# Splits por video
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Reglas por defecto (solo si no hay etiquetas finas E1/E2/E3)
ANGLE_RULES = {
    "trunk_incline_deg": 15.0,   # E1 si tronco desvía > 15° de la vertical en la fase inferior
    "knee_min_deg": 90.0         # E3 si ángulo de rodilla mínima > 90° => profundidad insuficiente
    # E2 (valgo) depende de vista frontal/oblicua; aquí no se activa por defecto
}

# Entrenamiento
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# Dispositivo
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Salidas
OUTPUT_DIR = "./outputs"
