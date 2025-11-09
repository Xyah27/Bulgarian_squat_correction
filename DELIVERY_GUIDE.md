# 📦 Guía de Entrega - Bulgarian Split Squat Posture Analysis

## 🎯 Propósito de este Documento

Este documento explica **qué contiene la carpeta de entrega** y cómo utilizarla.

---

## 📁 Estructura de la Carpeta de Entrega

```
bulgarian-split-squat/           ← CARPETA RAÍZ PARA ENTREGAR
│
├── 📦 src/bulgarian_squat/      ← CÓDIGO FUENTE (Paquete Python)
│   ├── __init__.py              ← Inicialización del paquete
│   ├── model_improved.py        ← Arquitectura BiGRU+Attention (modelo principal)
│   ├── datamodule.py            ← Carga y procesamiento de datos
│   ├── train.py                 ← Funciones de entrenamiento
│   ├── eval.py                  ← Evaluación y métricas
│   ├── rt_infer.py              ← Inferencia en tiempo real
│   └── 9 módulos más...         ← Utilidades y configuración
│
├── 🔧 scripts/                  ← SCRIPTS EJECUTABLES
│   ├── training/
│   │   └── train_bigru.py       ← Entrenar modelo desde cero
│   ├── inference/
│   │   └── run_webcam.py        ← ⭐ Inferencia con webcam (PRINCIPAL)
│   └── utils/
│       └── prepare_artifacts.py ← Preparar modelo para producción
│
├── 💾 data/                     ← DATOS
│   ├── raw/                     ← Dataset original
│   │   └── landmarks_dataset_BALANCEADO_v2.csv  (27,000+ samples)
│   └── processed/               ← Datos procesados (generados automáticamente)
│
├── 🤖 models/                   ← MODELOS ENTRENADOS
│   ├── best/                    ← ⭐ MODELO DE PRODUCCIÓN (PRINCIPAL)
│   │   ├── best_model_bigru.pt  ← Pesos del modelo (119K params, ~500KB)
│   │   ├── run_meta.json        ← Configuración y metadatos
│   │   ├── class_names.json     ← Nombres de las 4 clases
│   │   └── thr_per_class.npy    ← Umbrales óptimos [0.31, 0.19, 0.10, 0.70]
│   └── checkpoints/             ← Checkpoints de entrenamiento (vacío inicialmente)
│
├── 📚 docs/                     ← DOCUMENTACIÓN
│   ├── papers/                  ← ⭐ ARTÍCULO CIENTÍFICO
│   │   ├── paper_bulgarian_squat_es.pdf  ← Paper final (9 páginas)
│   │   └── paper_bulgarian_squat_es.tex  ← Código LaTeX (español)
│   ├── figures/                 ← Figuras técnicas (7 PDFs)
│   │   ├── architecture_diagram.pdf
│   │   ├── bigru_architecture.pdf
│   │   ├── confusion_matrix_normalized.pdf
│   │   ├── attention_weights_visualization.pdf
│   │   └── 3 figuras más...
│   └── references/              ← Literatura y referencias
│       ├── Literatura/          ← Carpeta de referencias bibliográficas
│       ├── FormatoDeRevistas.pdf
│       └── Análisis de postura y corrección de técnica en ejercicios.pdf
│
├── 📓 notebooks/                ← Jupyter notebooks de análisis (opcional)
├── ⚙️ configs/                  ← Archivos de configuración (vacío inicialmente)
├── 📊 logs/                     ← Logs de entrenamiento (se generan al entrenar)
│
└── 📄 ARCHIVOS DE CONFIGURACIÓN (Raíz)
    ├── README.md                ← ⭐ DOCUMENTACIÓN PRINCIPAL (300+ líneas)
    ├── QUICKSTART.md            ← ⭐ Guía de inicio rápido (3 pasos)
    ├── PROJECT_STRUCTURE.md     ← Explicación de la estructura del proyecto
    ├── CONTRIBUTING.md          ← Guía para contribuidores
    ├── CHANGELOG.md             ← Historial de cambios y versiones
    ├── requirements.txt         ← ⭐ Dependencias del proyecto (15+ libs)
    ├── setup.py                 ← Instalación como paquete Python
    ├── .gitignore               ← Archivos ignorados por Git
    ├── LICENSE                  ← Licencia MIT
    └── DELIVERY_GUIDE.md        ← ⭐ Este documento
```

---

## ⭐ Archivos Más Importantes

### 🚀 Para Ejecutar Inmediatamente
1. **README.md** → Lee esto primero
2. **QUICKSTART.md** → 3 pasos para empezar
3. **scripts/inference/run_webcam.py** → Script principal de inferencia
4. **models/best/** → Modelo entrenado listo para usar

### 📖 Para Entender el Proyecto
1. **docs/papers/paper_bulgarian_squat_es.pdf** → Artículo científico completo
2. **docs/figures/** → Gráficos y visualizaciones (14 archivos)
3. **docs/references/** → Referencias bibliográficas (16 PDFs)
4. **PROJECT_STRUCTURE.md** → Estructura detallada
5. **RESULTADOS_METRICAS.md** → Métricas completas del modelo

### 🛠️ Para Desarrollar/Entrenar
1. **scripts/training/train_bigru.py** → Entrenar modelo nuevo
2. **src/bulgarian_squat/** → Código fuente completo
3. **requirements.txt** → Dependencias

---

## 🚀 Inicio Rápido (3 Pasos)

### 1️⃣ Instalar Dependencias
```bash
cd "bulgarian-split-squat"
pip install -r requirements.txt
```

### 2️⃣ Ejecutar con Webcam
```bash
python scripts/inference/run_webcam.py --model models/best --cam 0
```

### 3️⃣ ¡Listo! 🎉
Colócate frente a la cámara y realiza Bulgarian Split Squats.

---

## 📊 Contenido del Proyecto

### Código Fuente
- **14 archivos Python** en `src/bulgarian_squat/`
- **3 scripts ejecutables** en `scripts/`
- **Arquitectura modular** y bien documentada

### Modelo Entrenado
- **BiGRU+Attention** (51.98% F1-Score macro, 65.74% Accuracy)
- **119,812 parámetros** (~500 KB)
- **4 clases**: correcta, E1_tronco, E2_valgo, E3_profundidad
- **Umbrales optimizados** para cada clase

### Dataset
- **27,000+ frames** de secuencias de video
- **66 features** (33 landmarks MediaPipe × 2 coords)
- **Multi-label** classification
- **Balanceado** con SMOTE

### Documentación
- **Paper científico** completo (9 páginas en español)
- **7 figuras técnicas** (arquitectura, resultados, confusion matrix)
- **6 guías** (README, QUICKSTART, etc.)
- **Referencias bibliográficas**

---

## 📦 Qué Incluye esta Entrega

### ✅ Incluido
- ✅ Código fuente completo y organizado
- ✅ Modelo pre-entrenado listo para usar
- ✅ Scripts de entrenamiento e inferencia
- ✅ Dataset completo (27K+ samples)
- ✅ Paper científico en PDF y LaTeX
- ✅ Figuras y gráficos técnicos
- ✅ Documentación extensa (6 guías)
- ✅ Dependencias especificadas
- ✅ Licencia MIT

### ❌ No Incluido (se genera al usar)
- ❌ Checkpoints intermedios de entrenamiento
- ❌ Logs de ejecución
- ❌ Archivos `__pycache__/`
- ❌ Datos procesados (se generan automáticamente)

---

## 🎓 Cómo Usar

### Para Ejecutar el Sistema
```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Ejecutar con webcam (cámara 0)
python scripts/inference/run_webcam.py --model models/best --cam 0

# Si tu cámara principal no funciona, prueba con otras
python scripts/inference/run_webcam.py --model models/best --cam 1
```

### Para Entrenar un Modelo Nuevo
```bash
python scripts/training/train_bigru.py \
    --dataset data/raw/landmarks_dataset_BALANCEADO_v2.csv \
    --epochs 100 \
    --batch_size 32 \
    --output_dir models/checkpoints
```

### Para Instalar como Paquete Python
```bash
pip install -e .

# Luego puedes importar:
from bulgarian_squat import BiGRUClassifierImproved
```

---

## 📚 Documentación Disponible

1. **README.md** (300+ líneas)
   - Instalación completa
   - Uso y ejemplos
   - Arquitectura del modelo
   - Resultados detallados

2. **QUICKSTART.md**
   - Inicio en 3 pasos
   - Troubleshooting rápido
   - Consejos de uso

3. **PROJECT_STRUCTURE.md**
   - Estructura del proyecto
   - Explicación de cada directorio
   - Ventajas de la organización

4. **CONTRIBUTING.md**
   - Guía para contribuidores
   - Estilo de código
   - Cómo reportar bugs

4. **RESULTADOS_METRICAS.md**
   - Métricas completas del modelo
   - Tablas de resultados por clase
   - Comparación con trabajos relacionados
   - Instrucciones de compilación del paper

5. **Paper Científico** (docs/papers/)
   - **paper_bulgarian_squat_es.pdf**: Paper compilado (9 páginas)
   - **paper_bulgarian_squat_es.tex**: Código fuente LaTeX
   - Introducción y motivación
   - Metodología completa
   - Resultados experimentales
   - Conclusiones y trabajo futuro

6. **Figuras y Referencias** (docs/)
   - **figures/**: 14 gráficos (PDF + PNG)
     - Arquitectura del modelo
     - Matrices de confusión
     - Comparaciones de entrenamiento
     - Distribución del dataset
   - **references/**: 16 papers de referencia (PDFs)

---

## 🔬 Especificaciones Técnicas

### Modelo
- **Arquitectura**: BiGRU + Attention + BatchNorm + LayerNorm
- **Input**: (T, 66) donde T = longitud de secuencia
- **Output**: (4,) clasificación multi-label
- **Parámetros**: 119,812
- **Tamaño**: ~500 KB

### Performance
- **F1-Score Macro**: 51.98%
- **Accuracy**: 65.74%
- **Inferencia**: ~30-45 FPS (CPU)
- **Latencia**: <50ms por frame

### Clases
1. **correcta** (F1: 0.81) - Técnica correcta
2. **E1_tronco** (F1: 0.38) - Inclinación excesiva del tronco
3. **E2_valgo** (F1: 0.15) - Valgo de rodilla
4. **E3_profundidad** (F1: 0.81) - Profundidad insuficiente

---

## 🆘 Soporte

### Problemas Comunes

**❌ "No se pudo abrir la cámara"**
```bash
# Prueba diferentes índices
python scripts/inference/run_webcam.py --cam 1
python scripts/inference/run_webcam.py --cam 2
```

**❌ "Import 'bulgarian_squat' could not be resolved"**
```bash
# Instalar como paquete
pip install -e .
```

**❌ "ModuleNotFoundError: No module named 'torch'"**
```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Más Ayuda
- Lee **QUICKSTART.md** para troubleshooting detallado
- Revisa **README.md** para documentación completa
- Consulta **docs/papers/** para detalles técnicos

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo `LICENSE` para más detalles.

---

## ✅ Checklist de Entrega

Antes de entregar, verifica que tienes:

- [ ] Carpeta completa `bulgarian-split-squat/`
- [ ] `src/bulgarian_squat/` con 14 archivos Python
- [ ] `scripts/` con 3 subdirectorios (training, inference, utils)
- [ ] `models/best/` con 4 archivos del modelo entrenado
- [ ] `data/raw/` con el dataset CSV
- [ ] `docs/papers/` con paper PDF y TEX (6 archivos)
- [ ] `docs/figures/` con 14 archivos (7 PDFs + 7 PNGs)
- [ ] `docs/references/` con 16 PDFs de referencias
- [ ] `README.md`, `QUICKSTART.md`, `PROJECT_STRUCTURE.md`, `DELIVERY_GUIDE.md`, `RESULTADOS_METRICAS.md`
- [ ] `requirements.txt`, `setup.py`, `.gitignore`, `LICENSE`

---

## 🎯 Resumen Ejecutivo

Este proyecto es un **sistema completo de análisis de postura** para el ejercicio Bulgarian Split Squat que:

✅ **Funciona inmediatamente** con webcam  
✅ **Está completamente documentado** (paper + 6 guías)  
✅ **Incluye modelo pre-entrenado** (51.98% F1)  
✅ **Es reproducible** (requirements.txt + setup.py)  
✅ **Es mantenible** (código modular y limpio)  
✅ **Es extensible** (fácil añadir nuevas funcionalidades)  
✅ **Está listo para producción**  

---

**📌 Tiempo estimado de setup:** 2-3 minutos  
**📌 Tamaño de la carpeta:** ~50-100 MB (con dataset)  
**📌 Versión:** 1.0.0  
**📌 Fecha:** 2024-11-06  

**🎉 ¡Disfruta del proyecto! 🏋️‍♂️**
