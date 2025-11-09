# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-06

### 🎉 Lanzamiento Inicial

Primera versión estable del sistema de análisis de postura Bulgarian Split Squat.

### ✨ Added (Nuevas Características)

#### Modelo
- Arquitectura BiGRU+Attention con 119K parámetros
- Mecanismo de atención para ponderación de secuencias temporales
- BatchNorm y LayerNorm para estabilidad de entrenamiento
- Multi-label classification (4 clases)
- F1-Score macro: 51.98%, Accuracy: 65.74%

#### Inferencia en Tiempo Real
- Script `run_webcam.py` para análisis con webcam
- Detección automática de repeticiones con FSM
- Soporte para vista frontal y lateral
- Detección automática de vista usando coordenadas 3D
- Modo debug para visualizar métricas
- ~30-45 FPS en CPU

#### Detección de Postura
- 4 clases: correcta, E1_tronco, E2_valgo, E3_profundidad
- 33 landmarks de MediaPipe Pose
- Umbrales óptimos por clase: [0.31, 0.19, 0.10, 0.70]
- Cálculo automático de ángulos de rodilla
- Selección inteligente de pierna activa

#### Entrenamiento
- Script `train_bigru.py` completo con argumentos CLI
- Early stopping con patience=15
- DataModule con train/val/test splits automáticos
- Balanceo de clases con SMOTE
- Logging de métricas por época
- Guardado automático del mejor modelo

#### Estructura del Proyecto
- Código organizado como paquete Python instalable
- Separación clara: src/, scripts/, data/, models/, docs/
- Scripts modulares por funcionalidad
- Configuración profesional con setup.py

#### Documentación
- README.md completo (300+ líneas)
- QUICKSTART.md (inicio en 3 pasos)
- PROJECT_STRUCTURE.md (estructura detallada)
- CONTRIBUTING.md (guía de contribución)
- REORGANIZATION_SUMMARY.md (resumen de cambios)
- Paper científico en español (9 páginas)
- 7 figuras técnicas (arquitectura, confusion matrix, etc.)

#### Configuración
- requirements.txt con 15+ dependencias
- setup.py para instalación como paquete
- .gitignore completo
- LICENSE MIT
- CHANGELOG.md

#### Artifacts del Modelo
- best_model_bigru.pt (pesos del modelo)
- run_meta.json (configuración y metadatos)
- class_names.json (nombres de las clases)
- thr_per_class.npy (umbrales óptimos)

### 🔧 Technical Details

#### Arquitectura del Modelo
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
Weighted Sum
    ↓
FC (64 → 4)
```

#### Dataset
- 27,000+ frames de secuencias de video
- Balanceado con SMOTE y undersampling
- 70% train, 15% val, 15% test
- Features: 66 (33 landmarks × 2 coords)
- Etiquetas multi-label binarias

#### Performance por Clase
| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| correcta | 0.79 | 0.82 | 0.81 |
| E1_tronco | 0.44 | 0.34 | 0.38 |
| E2_valgo | 0.29 | 0.10 | 0.15 |
| E3_profundidad | 0.72 | 0.93 | 0.81 |

### 📦 Dependencies

#### Core
- Python >= 3.8
- PyTorch >= 2.0.0
- MediaPipe >= 0.10.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0

#### Visualization
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

#### Utils
- tqdm >= 4.65.0
- pyyaml >= 6.0

### 📂 File Structure

```
bulgarian-split-squat/
├── src/bulgarian_squat/     # Paquete principal (14 archivos)
├── scripts/                  # Scripts ejecutables (3 subdirectorios)
├── data/                     # Datos (raw/ y processed/)
├── models/                   # Modelos (best/ y checkpoints/)
├── docs/                     # Documentación (papers/ y figures/)
├── notebooks/                # Jupyter notebooks
├── configs/                  # Configuraciones
├── logs/                     # Logs de entrenamiento
└── 9 archivos de config      # README, setup.py, requirements.txt, etc.
```

### 🚀 Usage Examples

#### Inferencia
```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
```

#### Entrenamiento
```bash
python scripts/training/train_bigru.py \
    --dataset data/raw/landmarks_dataset_BALANCEADO_v2.csv \
    --epochs 100 \
    --batch_size 32
```

#### Como Librería
```python
from bulgarian_squat import BiGRUClassifierImproved
model = BiGRUClassifierImproved(in_dim=66, num_classes=4)
```

### 🐛 Known Issues

- E2_valgo tiene baja recall (0.10) debido a desbalance en el dataset
- Detección de vista puede fallar con iluminación muy pobre
- Movimientos muy rápidos (<1 seg) pueden no capturarse completamente

### 🔜 Future Work

- [ ] Añadir tests unitarios con pytest
- [ ] Configurar CI/CD con GitHub Actions
- [ ] Crear Dockerfile para deployment
- [ ] Interfaz web con Streamlit/Gradio
- [ ] API REST con FastAPI
- [ ] Aumentar dataset para E2_valgo
- [ ] Soporte para múltiples personas simultáneas
- [ ] Exportar modelo a ONNX para optimización

---

## [Unreleased]

### Planned Features
- Tests automatizados (pytest)
- CI/CD pipeline (GitHub Actions)
- Docker containerization
- Web interface (Streamlit)
- REST API (FastAPI)
- Model optimization (ONNX, TensorRT)
- Multi-person detection
- Cloud deployment (AWS, Azure, GCP)

---

## Tipos de Cambios

- **Added** para nuevas características
- **Changed** para cambios en funcionalidad existente
- **Deprecated** para características que se eliminarán pronto
- **Removed** para características eliminadas
- **Fixed** para correcciones de bugs
- **Security** para parches de seguridad

---

**Mantenedores:** Tu Nombre <tu.email@example.com>  
**Licencia:** MIT  
**Repositorio:** https://github.com/tu-usuario/bulgarian-split-squat
