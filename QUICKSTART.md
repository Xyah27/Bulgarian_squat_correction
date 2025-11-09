# Guía de Inicio Rápido

## ⚡ Quick Start en 3 Pasos

### 1️⃣ Instalar Dependencias (1 minuto)

```bash
pip install -r requirements.txt
```

### 2️⃣ Probar con Webcam (inmediato)

```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
```

**¿No funciona tu cámara?** Prueba con otro índice:
```bash
python scripts/inference/run_webcam.py --cam 1
# o --cam 2, etc.
```

### 3️⃣ ¡Listo! 🎉

Colócate frente a la cámara en **vista lateral** y realiza una Bulgarian Split Squat.

---

## 📋 Controles Durante Ejecución

- **D**: Ver métricas de detección (debug mode)
- **ESPACIO**: Captura manual on/off
- **Q o ESC**: Salir

---

## 🔧 Troubleshooting Rápido

### ❌ Error: "No se pudo abrir la cámara"
```bash
# Listar cámaras disponibles y probar cada una
python scripts/inference/run_webcam.py --cam 0  # Prueba 0, 1, 2...
```

### ❌ Error: "Import 'bulgarian_squat' could not be resolved"
```bash
# Opción 1: Instalar como paquete
pip install -e .

# Opción 2: Usar Python desde la raíz del proyecto
cd "C:\Users\JUAN JOSE\Desktop\Workspace\Electiva IA\Proyecto VISION BULGARA"
python scripts/inference/run_webcam.py --model models/best --cam 0
```

### ❌ Error: "ModuleNotFoundError: No module named 'torch'"
```bash
# Instalar PyTorch (con GPU si tienes CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# O sin GPU
pip install torch torchvision
```

### ❌ Error: "No module named 'mediapipe'"
```bash
pip install mediapipe opencv-python
```

---

## 🎓 Para Entrenar tu Propio Modelo

### Dataset Mínimo Requerido

Necesitas un CSV con:
- 33 landmarks × 2 coords (x, y) = 66 columnas de features
- Columnas de etiquetas: `correcta`, `E1_tronco`, `E2_valgo`, `E3_profundidad`
- Valores binarios (0 o 1)

### Comando de Entrenamiento

```bash
python scripts/training/train_bigru.py \
    --dataset data/raw/tu_dataset.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 15
```

El modelo se guardará en `models/checkpoints/bigru_[timestamp]/`

### Usar tu Modelo Entrenado

```bash
# 1. Preparar artifacts
python scripts/utils/prepare_artifacts.py \
    --model models/checkpoints/bigru_20241106_123456/best_model.pt \
    --output models/my_model

# 2. Usar en inferencia
python scripts/inference/run_webcam.py --model models/my_model --cam 0
```

---

## 📚 Documentación Completa

Para información detallada, ver:
- **README.md**: Documentación completa
- **PROJECT_STRUCTURE.md**: Estructura del proyecto
- **CONTRIBUTING.md**: Guía para contribuir
- **docs/papers/**: Artículo científico

---

## 💡 Consejos para Mejores Resultados

### Configuración de Cámara
✅ **Distancia**: 2-3 metros de la cámara  
✅ **Vista**: Lateral completa (perfil) o frontal  
✅ **Iluminación**: Buena luz, fondo contrastante  
✅ **Posición**: Todo el cuerpo visible  

### Ejecución del Ejercicio
✅ **Velocidad**: Movimientos LENTOS (2-3 seg por rep)  
✅ **Amplitud**: Rango completo de movimiento  
✅ **Control**: Evitar movimientos bruscos  

### Interpretación de Resultados

El sistema detecta 4 tipos de postura:

- **✅ correcta** (0.82 F1): Técnica correcta
- **⚠️ E1_tronco** (0.38 F1): Tronco muy inclinado
- **⚠️ E2_valgo** (0.15 F1): Rodilla hacia dentro
- **⚠️ E3_profundidad** (0.81 F1): Bajada insuficiente

**Puede detectar múltiples errores** en una misma repetición.

---

## 🆘 ¿Necesitas Ayuda?

1. **Issues**: [GitHub Issues](https://github.com/tu-usuario/bulgarian-split-squat/issues)
2. **Email**: tu.email@example.com
3. **Docs**: Lee README.md completo

---

## ✨ Ejemplos de Uso

### Ejemplo 1: Análisis Básico
```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
```

### Ejemplo 2: Ajustar Sensibilidad
```bash
python scripts/inference/run_webcam.py \
    --model models/best \
    --cam 0 \
    --minlen 15 \
    --maxlen 120
```

### Ejemplo 3: Modo Debug
```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
# Presiona 'D' durante ejecución para ver métricas
```

---

**⏱️ Tiempo total de setup: ~2-3 minutos**

¡Disfruta analizando tu técnica! 🏋️‍♂️
