"""
Script para probar ambos modelos (PyTorch y TensorFlow) y verificar que funcionan.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

import torch
import tensorflow as tf  # type: ignore[import-not-found]
from bulgarian_squat.model_improved import BiGRUClassifierImproved

print("\n" + "="*70)
print("PRUEBA DE MODELOS - PYTORCH vs TENSORFLOW")
print("="*70 + "\n")

# Directorio de modelos
model_dir = Path("models/best")

# Datos de prueba
print("Generando datos de prueba...")
batch_size = 2
seq_length = 50
features = 66

test_data = np.random.randn(batch_size, seq_length, features).astype(np.float32)
print(f"Shape entrada: {test_data.shape}\n")

# ============================================
# 1. MODELO PYTORCH
# ============================================
print("="*70)
print("1. MODELO PYTORCH (.pt)")
print("="*70 + "\n")

pytorch_path = model_dir / "best_model_bigru.pt"
print(f"Cargando: {pytorch_path}")

model_pt = BiGRUClassifierImproved(
    in_dim=66,
    num_classes=4,
    hidden1=128,
    hidden2=64,
    dropout=0.3,
    use_attention=True
)
model_pt.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
model_pt.eval()

print("OK Modelo PyTorch cargado")
print(f"Parametros: {sum(p.numel() for p in model_pt.parameters()):,}\n")

# Inferencia PyTorch
print("Ejecutando inferencia...")
with torch.no_grad():
    input_tensor = torch.from_numpy(test_data)
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    logits_pt = model_pt(input_tensor, mask)
    probs_pt = torch.sigmoid(logits_pt).numpy()

print(f"OK Inferencia completada")
print(f"Output shape: {probs_pt.shape}")
print(f"Output batch 1: {probs_pt[0]}")
print(f"Output batch 2: {probs_pt[1]}\n")

# ============================================
# 2. MODELO TENSORFLOW (.h5)
# ============================================
print("="*70)
print("2. MODELO TENSORFLOW (.h5)")
print("="*70 + "\n")

h5_path = model_dir / "bulgarian_squat_model.h5"
print(f"Cargando: {h5_path}")

model_tf = tf.keras.models.load_model(h5_path)

print("OK Modelo TensorFlow cargado")
print(f"Parametros: {model_tf.count_params():,}\n")

# Inferencia TensorFlow
print("Ejecutando inferencia...")
probs_tf = model_tf.predict(test_data, verbose=0)

print(f"OK Inferencia completada")
print(f"Output shape: {probs_tf.shape}")
print(f"Output batch 1: {probs_tf[0]}")
print(f"Output batch 2: {probs_tf[1]}\n")

# ============================================
# 3. COMPARACION
# ============================================
print("="*70)
print("3. COMPARACION DE RESULTADOS")
print("="*70 + "\n")

print("IMPORTANTE: Los modelos tienen la MISMA arquitectura pero DIFERENTES pesos")
print("Por lo tanto, las predicciones seran diferentes.\n")

print(f"PyTorch output:    {probs_pt[0]}")
print(f"TensorFlow output: {probs_tf[0]}\n")

diff = np.abs(probs_pt - probs_tf).mean()
print(f"Diferencia promedio: {diff:.4f}")

if diff > 0.3:
    print("✓ Las salidas son diferentes (esperado - pesos no transferidos)")
else:
    print("⚠️  Las salidas son muy similares (inesperado)")

# ============================================
# 4. VERIFICACION DE FORMA
# ============================================
print("\n" + "="*70)
print("4. VERIFICACION")
print("="*70 + "\n")

checks = [
    ("Modelo PyTorch carga correctamente", pytorch_path.exists()),
    ("Modelo TensorFlow carga correctamente", h5_path.exists()),
    ("Output PyTorch shape correcta", probs_pt.shape == (batch_size, 4)),
    ("Output TensorFlow shape correcta", probs_tf.shape == (batch_size, 4)),
    ("Valores PyTorch en rango [0,1]", (probs_pt >= 0).all() and (probs_pt <= 1).all()),
    ("Valores TensorFlow en rango [0,1]", (probs_tf >= 0).all() and (probs_tf <= 1).all()),
]

all_pass = True
for check_name, result in checks:
    status = "OK" if result else "ERROR"
    print(f"[{status}] {check_name}")
    if not result:
        all_pass = False

print("\n" + "="*70)
if all_pass:
    print("RESULTADO: TODOS LOS TESTS PASARON")
else:
    print("RESULTADO: ALGUNOS TESTS FALLARON")
print("="*70 + "\n")

# ============================================
# 5. INFORMACION DE ARCHIVOS
# ============================================
print("="*70)
print("5. ARCHIVOS GENERADOS")
print("="*70 + "\n")

pt_size = pytorch_path.stat().st_size / (1024**2)
h5_size = h5_path.stat().st_size / (1024**2)
sm_path = model_dir / "saved_model"
sm_size = sum(f.stat().st_size for f in sm_path.rglob('*') if f.is_file()) / (1024**2)

print(f"1. {pytorch_path.name}")
print(f"   Formato: PyTorch state_dict")
print(f"   Tamano: {pt_size:.2f} MB")
print(f"   Pesos: SI (entrenados)\n")

print(f"2. {h5_path.name}")
print(f"   Formato: Keras HDF5")
print(f"   Tamano: {h5_size:.2f} MB")
print(f"   Pesos: NO (inicializados aleatoriamente)\n")

print(f"3. {sm_path.name}/")
print(f"   Formato: TensorFlow SavedModel")
print(f"   Tamano: {sm_size:.2f} MB")
print(f"   Pesos: NO (inicializados aleatoriamente)\n")

print("="*70)
print("PRUEBA COMPLETADA")
print("="*70 + "\n")
