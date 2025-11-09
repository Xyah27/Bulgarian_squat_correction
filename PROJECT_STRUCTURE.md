# Estructura del Proyecto Bulgarian Split Squat

## 📂 Organización Final

```
bulgarian-split-squat/
│
├── 📦 src/bulgarian_squat/           # Código fuente principal (paquete Python)
│   ├── __init__.py                   # Inicialización del paquete
│   ├── model_improved.py             # Arquitectura BiGRU+Attention
│   ├── datamodule.py                 # Carga y procesamiento de datos
│   ├── train.py                      # Funciones de entrenamiento
│   ├── eval.py                       # Evaluación y métricas
│   ├── rt_infer.py                   # Inferencia en tiempo real
│   ├── config.py                     # Configuración global
│   ├── features.py                   # Extracción de características
│   ├── labels.py                     # Procesamiento de etiquetas
│   ├── splits.py                     # Train/val/test splits
│   ├── viz.py                        # Visualizaciones
│   └── data_utils.py                 # Utilidades de datos
│
├── 🔧 scripts/                       # Scripts ejecutables
│   ├── training/
│   │   └── train_bigru.py            # Entrenar modelo
│   ├── inference/
│   │   └── run_webcam.py             # Inferencia con webcam
│   └── utils/
│       └── prepare_artifacts.py      # Preparar artifacts del modelo
│
├── 💾 data/                          # Datos del proyecto
│   ├── raw/                          # Datos originales sin procesar
│   │   └── landmarks_dataset_BALANCEADO_v2.csv
│   └── processed/                    # Datos preprocesados (generados)
│
├── 🤖 models/                        # Modelos entrenados
│   ├── best/                         # Mejor modelo para producción
│   │   ├── best_model_bigru.pt       # Pesos del modelo
│   │   ├── run_meta.json             # Metadatos y configuración
│   │   ├── class_names.json          # Nombres de las clases
│   │   └── thr_per_class.npy         # Umbrales óptimos
│   └── checkpoints/                  # Checkpoints de entrenamiento
│       └── .gitkeep
│
├── 📚 docs/                          # Documentación
│   ├── papers/                       # Paper académico
│   │   ├── paper_bulgarian_squat_es.pdf   # Paper compilado
│   │   ├── paper_bulgarian_squat_es.tex   # Código fuente LaTeX
│   │   ├── paper_bulgarian_squat_es.aux   # Archivos auxiliares
│   │   ├── paper_bulgarian_squat_es.log   # Log de compilación
│   │   ├── paper_bulgarian_squat_es.out   # Output
│   │   └── compile_output.txt             # Salida de compilación
│   ├── figures/                      # Figuras y gráficos (14 archivos)
│   │   ├── architecture_diagram.pdf
│   │   ├── bigru_architecture.pdf/png
│   │   ├── confusion_matrix_normalized.pdf/png
│   │   ├── attention_weights_visualization.pdf
│   │   ├── bigru_comparison_training.pdf/png
│   │   ├── bigru_results_comparison.pdf/png
│   │   ├── dataset_distribution.pdf/png
│   │   └── per_class_metrics.pdf/png
│   └── references/                   # Referencias bibliográficas (16 PDFs)
│       ├── 1-s2.0-S0010482520300780-main.pdf
│       ├── 1-s2.0-S0010482521001104-main.pdf
│       ├── 1-s2.0-S1877050924033660-main.pdf
│       ├── 1_s20_S0010482523009502_main.pdf
│       ├── 1_s20_S096663622500178X_main.pdf
│       ├── 1_s20_S1110016825006283_main.pdf
│       ├── 5548-11649-1-PB.pdf
│       ├── AI-Based_Posture_Correction_Real-Time_Exercise_Tracking_and_Feedback_using_Pose_Estimation_Technique.pdf
│       ├── Análisis de postura y corrección de técnica en ejercicios.pdf
│       ├── FormatoDeRevistas.pdf
│       ├── KINOVEAPUBLICADO.pdf
│       ├── Referencias_IEEE.txt
│       ├── s11042_025_21050_3.pdf
│       ├── s11760_025_04436_6.pdf
│       ├── s41598_024_66221_8.pdf
│       └── s43926_025_00200_x.pdf
│
├── 📓 notebooks/                     # Jupyter notebooks de análisis
│   └── .gitkeep
│
├── ⚙️ configs/                       # Archivos de configuración
│
├── 📊 logs/                          # Logs de entrenamiento
│   └── .gitkeep
│
├── 📄 Archivos de Configuración
│   ├── requirements.txt              # Dependencias del proyecto
│   ├── setup.py                      # Instalación del paquete
│   ├── .gitignore                    # Archivos ignorados por git
│   ├── LICENSE                       # Licencia MIT
│   ├── README.md                     # Documentación principal
│   └── CONTRIBUTING.md               # Guía de contribución
│
└── 📁 Archivos Legacy (mantener por referencia)
    ├── CODE/                         # Código original (deprecado)
    ├── test_webcam.py                # Script de prueba (usar scripts/inference/run_webcam.py)
    ├── run_bigru_experiments.py      # Experimentos (deprecado)
    └── Literatura/                   # Referencias bibliográficas
```

## 🎯 Cómo Usar la Nueva Estructura

### 1. Instalación

```bash
# Instalar como paquete
pip install -e .

# O instalar solo dependencias
pip install -r requirements.txt
```

### 2. Entrenamiento

```bash
python scripts/training/train_bigru.py \
    --dataset data/raw/landmarks_dataset_BALANCEADO_v2.csv \
    --epochs 100 \
    --batch_size 32 \
    --output_dir models/checkpoints
```

### 3. Inferencia con Webcam

```bash
python scripts/inference/run_webcam.py \
    --model models/best \
    --cam 0
```

### 4. Uso como Biblioteca

```python
from bulgarian_squat import BiGRUClassifierImproved, PoseStreamer

# Crear modelo
model = BiGRUClassifierImproved(in_dim=66, num_classes=4)

# Usar pose streamer
streamer = PoseStreamer(camera_idx=0)
```

## 📋 Ventajas de la Nueva Estructura

### ✅ Modularidad
- **Separación clara** entre código fuente, scripts y datos
- **Paquete Python** instalable con `pip install -e .`
- **Imports limpios**: `from bulgarian_squat import BiGRUClassifierImproved`

### ✅ Reproducibilidad
- **requirements.txt** con todas las dependencias
- **setup.py** para instalación consistente
- **Metadatos** del modelo en JSON
- **Semilla fija** para experimentos reproducibles

### ✅ Mantenibilidad
- **Código organizado** por funcionalidad
- **Documentación completa** en README.md
- **Guía de contribución** en CONTRIBUTING.md
- **Estilo consistente** con PEP 8

### ✅ Escalabilidad
- **Fácil añadir** nuevos modelos en `src/bulgarian_squat/`
- **Scripts independientes** en `scripts/`
- **Tests** en directorio `tests/` (pendiente)
- **Configuraciones** centralizadas en `configs/`

### ✅ Profesionalismo
- **Estructura estándar** de proyecto Python
- **Licencia MIT** clara
- **Documentación** extensa
- **Versionado** semántico (1.0.0)

## 🚀 Comandos Rápidos

### Entrenamiento rápido
```bash
python scripts/training/train_bigru.py --epochs 50
```

### Inferencia rápida
```bash
python scripts/inference/run_webcam.py --cam 1
```

### Preparar modelo para producción
```bash
python scripts/utils/prepare_artifacts.py \
    --model models/checkpoints/bigru_20241106/best_model.pt \
    --output models/best
```

### Instalar en modo desarrollo
```bash
pip install -e ".[dev]"  # Incluye herramientas de desarrollo
```

## 📦 Archivos Esenciales para Distribución

Si quieres compartir el proyecto, incluye:

```
bulgarian-split-squat/
├── src/bulgarian_squat/     # Todo el código
├── scripts/                  # Scripts ejecutables
├── models/best/              # Modelo pre-entrenado
├── docs/                     # Documentación
│   ├── papers/               # Paper académico (PDF + TEX)
│   ├── figures/              # Figuras (14 archivos PDF/PNG)
│   └── references/           # Referencias bibliográficas (16 PDFs)
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

**NO incluir:**
- `data/raw/` (dataset puede ser grande, compartir enlace)
- `models/checkpoints/` (checkpoints intermedios)
- `logs/` (logs de entrenamiento)
- `__pycache__/`, `*.pyc` (archivos compilados)

## 🔄 Migración desde Código Antiguo

### Antes (CODE/)
```python
import sys
sys.path.insert(0, "CODE/src")
from model_improved import BiGRUClassifierImproved
```

### Ahora (src/bulgarian_squat/)
```python
from bulgarian_squat import BiGRUClassifierImproved
```

### Scripts
- `CODE/run_webcam.py` → `scripts/inference/run_webcam.py`
- `run_bigru_experiments.py` → `scripts/training/train_bigru.py`

## ✨ Próximos Pasos

1. **Tests**: Añadir tests unitarios en `tests/`
2. **CI/CD**: Configurar GitHub Actions
3. **Documentación**: Generar docs con Sphinx
4. **Docker**: Crear Dockerfile para deployment
5. **Web App**: Interfaz web con Streamlit/Gradio

---

**Creado:** 2024-11-06  
**Versión:** 1.0.0  
**Última actualización:** 2024-11-06
