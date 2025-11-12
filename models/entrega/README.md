# Modelo Entrenado - Bulgarian Split Squat Classifier

## 📦 Contenido de esta Carpeta

- **bulgarian_squat_model.pt**: Modelo entrenado en formato PyTorch
- **MODEL_INFO.json**: Información completa del modelo y métricas
- **run_meta.json**: Metadatos del entrenamiento
- **class_names.json**: Nombres de las clases
- **complete_metrics.json**: Métricas completas de evaluación
- **thr_per_class.npy**: Umbrales óptimos por clase
- **README.md**: Este archivo

## 🎯 Información del Modelo

- **Framework**: PyTorch
- **Arquitectura**: BiGRU + Attention + BatchNorm
- **Parámetros**: 292,041
- **Input**: Secuencias de 66 features (33 landmarks × 2 coordenadas)
- **Output**: 4 clases (multi-label)

## 📊 Métricas

- **F1-Score Macro**: 51.98%
- **F1-Score Micro**: 58.38%
- **Accuracy**: 65.74%

## 🏋️ Clases

0. **E0_correcta**: Técnica correcta del ejercicio
1. **E1_inclinacion_tronco**: Inclinación excesiva del tronco
2. **E2_valgo_rodilla**: Rodilla colapsando hacia adentro
3. **E3_profundidad_insuficiente**: Rango de movimiento reducido

## 💻 Uso del Modelo

### Cargar el modelo:

```python
import torch
from src.bulgarian_squat.model_improved import BiGRUClassifierImproved

# Crear modelo
model = BiGRUClassifierImproved(
    in_dim=66,
    hidden1=128,
    hidden2=64,
    num_classes=4,
    dropout=0.3,
    use_batch_norm=True,
    use_attention=True
)

# Cargar pesos
checkpoint = torch.load('bulgarian_squat_model.pt')
model.load_state_dict(checkpoint)
model.eval()
```

### Inferencia:

```python
import torch

# Preparar input (batch_size, seq_len, 66)
x = torch.randn(1, 30, 66)  # Ejemplo
mask = torch.ones(1, 30)

# Predecir
with torch.no_grad():
    predictions = model(x, mask)

# predictions shape: (1, 4) - probabilidades para cada clase
```

## 📄 Archivos Relacionados

- Paper: `docs/papers/paper_bulgarian_squat_es.pdf`
- Código fuente: `src/bulgarian_squat/`
- Notebook de resultados: `notebooks/resultados_paper.ipynb`

## 🔗 Repositorio

GitHub: https://github.com/Xyah27/Bulgarian_squat_correction

---

**Autores**: Juan Jose Núñez, Juan Jose Castro  
**Institución**: Universidad San Buenaventura, Cali, Colombia
