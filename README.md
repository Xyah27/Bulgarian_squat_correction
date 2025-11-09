# Bulgarian Split Squat - Sistema de Análisis de Postura

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de análisis automático de postura para el ejercicio Bulgarian Split Squat utilizando **MediaPipe Pose** y modelos **BiGRU con mecanismo de atención**.

## 📋 Características

- ✅ **Detección de postura en tiempo real** con MediaPipe Pose (33 landmarks)
- ✅ **Clasificación multi-etiqueta** de 4 tipos de postura:
  - `correcta`: Técnica correcta
  - `E1_tronco`: Inclinación excesiva del tronco
  - `E2_valgo`: Valgo de rodilla (rodilla hacia dentro)
  - `E3_profundidad`: Profundidad insuficiente
- ✅ **Arquitectura BiGRU+Attention** (51.98% F1-score macro)
- ✅ **Inferencia en tiempo real** con webcam (~30 FPS)
- ✅ **Detección automática de vista** (frontal/lateral)
- ✅ **FSM para detección de repeticiones** automática

## 🎯 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **F1-Score (Macro)** | 51.98% |
| **Accuracy** | 65.74% |
| **Parámetros** | 119,812 |
| **Tamaño del modelo** | ~500 KB |

## 📁 Estructura del Proyecto

```
bulgarian-split-squat/
├── src/
│   └── bulgarian_squat/          # Paquete principal
│       ├── __init__.py
│       ├── model_improved.py     # Arquitectura BiGRU+Attention
│       ├── datamodule.py         # Carga y procesamiento de datos
│       ├── train.py              # Funciones de entrenamiento
│       ├── eval.py               # Evaluación y métricas
│       ├── rt_infer.py           # Inferencia en tiempo real
│       ├── config.py             # Configuración global
│       ├── features.py           # Extracción de características
│       ├── labels.py             # Procesamiento de etiquetas
│       ├── splits.py             # Divisiones train/val/test
│       ├── viz.py                # Visualizaciones
│       └── data_utils.py         # Utilidades de datos
│
├── scripts/
│   ├── training/
│   │   └── train_bigru.py        # Script de entrenamiento
│   ├── inference/
│   │   └── run_webcam.py         # Inferencia con webcam
│   └── utils/
│       └── prepare_artifacts.py  # Preparar artifacts de modelo
│
├── data/
│   ├── raw/                      # Datos originales
│   │   └── landmarks_dataset_BALANCEADO_v2.csv
│   └── processed/                # Datos procesados
│
├── models/
│   ├── best/                     # Mejor modelo entrenado
│   │   ├── best_model_bigru.pt
│   │   ├── run_meta.json
│   │   ├── class_names.json
│   │   └── thr_per_class.npy
│   └── checkpoints/              # Checkpoints de entrenamiento
│
├── docs/
│   ├── papers/                   # Artículos y documentación
│   │   └── paper_bulgarian_squat_es.pdf
│   └── figures/                  # Figuras y gráficos
│       ├── architecture_diagram.pdf
│       ├── bigru_architecture.pdf
│       ├── confusion_matrix_normalized.pdf
│       └── attention_weights_visualization.pdf
│
├── configs/                      # Archivos de configuración
├── notebooks/                    # Jupyter notebooks de análisis
├── logs/                         # Logs de entrenamiento
├── requirements.txt              # Dependencias del proyecto
├── setup.py                      # Instalación del paquete
├── .gitignore                    # Archivos ignorados por git
└── README.md                     # Este archivo
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip
- (Opcional) CUDA para aceleración GPU

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/bulgarian-split-squat.git
cd bulgarian-split-squat
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

O instalar el paquete en modo desarrollo:

```bash
pip install -e .
```

## 📦 Dependencias Principales

- **PyTorch** (>= 2.0.0): Framework de deep learning
- **MediaPipe** (>= 0.10.0): Detección de pose
- **OpenCV** (>= 4.8.0): Procesamiento de video
- **NumPy** (>= 1.24.0): Operaciones numéricas
- **pandas** (>= 2.0.0): Manipulación de datos
- **scikit-learn** (>= 1.3.0): Métricas y utilidades
- **matplotlib** (>= 3.7.0): Visualizaciones
- **seaborn** (>= 0.12.0): Visualizaciones estadísticas

## 🎓 Uso

### 1. Inferencia en Tiempo Real con Webcam

Ejecutar el sistema de análisis con tu cámara:

```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
```

**Opciones:**
- `--model`: Directorio con el modelo entrenado (default: `models/best`)
- `--cam`: Índice de la cámara (default: `0`)
  - Intenta `--cam 1`, `--cam 2`, etc. si la cámara principal no funciona
- `--minlen`: Mínimo de frames por repetición (default: `20`)
- `--maxlen`: Máximo de frames por repetición (default: `90`)

**Controles en tiempo de ejecución:**
- **D**: Activar/desactivar modo debug (muestra métricas de detección)
- **ESPACIO**: Modo captura manual on/off
- **Q o ESC**: Salir

**Consejos para mejor detección:**
- ✅ Colócate en **vista lateral** o **frontal** completa
- ✅ Asegúrate de que todo tu cuerpo sea visible
- ✅ Realiza movimientos **lentos y controlados** (2-3 segundos por repetición)
- ✅ Buena iluminación y fondo contrastante

### 2. Entrenar un Modelo Nuevo

Entrenar desde cero con tu propio dataset:

```bash
python scripts/training/train_bigru.py \
    --dataset data/raw/landmarks_dataset_BALANCEADO_v2.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --hidden1 128 \
    --hidden2 64 \
    --dropout 0.3 \
    --use_attention \
    --output_dir models/checkpoints \
    --patience 15
```

**Argumentos principales:**
- `--dataset`: Ruta al archivo CSV con los datos
- `--epochs`: Número de épocas de entrenamiento
- `--batch_size`: Tamaño del batch
- `--lr`: Learning rate
- `--hidden1`, `--hidden2`: Tamaños de capas ocultas
- `--dropout`: Tasa de dropout
- `--use_attention`: Activar mecanismo de atención
- `--patience`: Paciencia para early stopping
- `--output_dir`: Directorio para guardar checkpoints

El script genera:
- `best_model.pt`: Pesos del mejor modelo
- `run_meta.json`: Metadatos y configuración
- `class_names.json`: Nombres de las clases
- `thr_per_class.npy`: Umbrales óptimos por clase

### 3. Preparar Artifacts para Inferencia

Si entrenaste un modelo nuevo, prepara los artifacts:

```bash
python scripts/utils/prepare_artifacts.py \
    --model models/checkpoints/bigru_20241106_123456/best_model.pt \
    --output models/best
```

## 📊 Formato del Dataset

El dataset debe ser un archivo CSV con las siguientes columnas:

```csv
frame_id,video_name,landmark_0_x,landmark_0_y,...,landmark_32_x,landmark_32_y,correcta,E1_tronco,E2_valgo,E3_profundidad
0,video1.mp4,0.5,0.3,...,0.6,0.8,1,0,0,0
1,video1.mp4,0.51,0.31,...,0.61,0.81,1,0,0,0
...
```

**Características:**
- **Landmarks**: 33 puntos × 2 coordenadas (x, y) = 66 features
- **Etiquetas**: Multi-etiqueta binaria (0 o 1) para cada clase
- **Frames**: Secuencias de frames agrupados por `video_name`

## 🧪 Arquitectura del Modelo

```
Input (T, 66)
    ↓
BatchNorm1d
    ↓
BiGRU Layer 1 (hidden_size=128)
    ↓
LayerNorm + Dropout(0.3)
    ↓
BiGRU Layer 2 (hidden_size=64)
    ↓
LayerNorm + Dropout(0.3)
    ↓
Attention Mechanism
    ↓
Weighted Sum (context vector)
    ↓
Fully Connected (64 → 4)
    ↓
Output (4 clases)
```

## 🔬 Evaluación

Para evaluar un modelo en el conjunto de test:

```python
from bulgarian_squat import BiGRUClassifierImproved
from bulgarian_squat.datamodule import BulgarianSquatDataModule
from bulgarian_squat.eval import evaluate_model

# Cargar datos
dm = BulgarianSquatDataModule(csv_path="data/raw/dataset.csv")
dm.setup()
test_loader = dm.test_dataloader()

# Cargar modelo
model = BiGRUClassifierImproved(in_dim=66, num_classes=4)
model.load_state_dict(torch.load("models/best/best_model_bigru.pt"))
model.eval()

# Evaluar
test_loss, metrics = evaluate_model(model, test_loader, criterion, device, verbose=True)
```

## 📈 Resultados Detallados por Clase

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **correcta** | 0.79 | 0.82 | 0.81 |
| **E1_tronco** | 0.44 | 0.34 | 0.38 |
| **E2_valgo** | 0.29 | 0.10 | 0.15 |
| **E3_profundidad** | 0.72 | 0.93 | 0.81 |

**Umbrales óptimos:**
- correcta: 0.31
- E1_tronco: 0.19
- E2_valgo: 0.10
- E3_profundidad: 0.70

## 🛠️ Desarrollo

### Estructura Modular

El código está organizado en módulos independientes para facilitar mantenimiento:

- **models**: Definición de arquitecturas
- **data**: Carga y preprocesamiento
- **training**: Loops de entrenamiento
- **evaluation**: Métricas y validación
- **inference**: Inferencia en producción
- **utils**: Utilidades compartidas

### Agregar Nuevos Modelos

1. Crear archivo en `src/bulgarian_squat/model_nuevo.py`
2. Heredar de `nn.Module` e implementar `forward(x, mask)`
3. Registrar en `__init__.py`
4. Crear script de entrenamiento en `scripts/training/`

### Agregar Nuevas Características

1. Modificar `features.py` para extraer nuevas características
2. Actualizar `in_dim` en configuración del modelo
3. Re-entrenar con nuevo dataset

## 📝 Citación

Si utilizas este proyecto en tu investigación, por favor cita:

```bibtex
@article{bulgarian_squat_2024,
  title={Análisis Automático de Postura en Bulgarian Split Squat usando BiGRU con Atención},
  author={Tu Nombre},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

- **Autor**: Tu Nombre
- **Email**: tu.email@example.com
- **Proyecto**: [https://github.com/tu-usuario/bulgarian-split-squat](https://github.com/tu-usuario/bulgarian-split-squat)

## 🙏 Agradecimientos

- **MediaPipe** por la detección de pose de alta calidad
- **PyTorch** por el framework de deep learning
- Comunidad de investigación en visión por computadora

## 📚 Referencias

1. MediaPipe Pose: [https://google.github.io/mediapipe/solutions/pose](https://google.github.io/mediapipe/solutions/pose)
2. BiGRU Networks: Bidirectional Gated Recurrent Units
3. Attention Mechanisms in Deep Learning
4. Multi-label Classification for Pose Assessment

---

**🏋️ ¡Entrena con técnica correcta! 🏋️**
