"""
Script para preparar el modelo entrenado para entrega
Guarda el modelo PyTorch y crea una versión serializada simple
"""

import torch
import json
import numpy as np
import pickle
from pathlib import Path

def export_model():
    """Exporta el modelo en formato .pt (PyTorch estándar)"""
    from src.bulgarian_squat.model_improved import BiGRUClassifierImproved
    
    print("=" * 70)
    print("PREPARACIÓN DEL MODELO ENTRENADO PARA ENTREGA")
    print("=" * 70)
    
    # Crear directorio de salida
    output_dir = Path('models/entrega')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Cargar modelo PyTorch
    print("\n[1/5] Cargando modelo entrenado...")
    model = BiGRUClassifierImproved(
        in_dim=66,
        hidden1=128,
        hidden2=64,
        num_classes=4,
        dropout=0.3,
        use_batch_norm=True,
        use_attention=True
    )
    
    checkpoint = torch.load('models/best/best_model_bigru.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Modelo cargado: {num_params:,} parámetros")
    
    # 2. Guardar modelo completo
    print("\n[2/5] Guardando modelo PyTorch...")
    torch.save(model.state_dict(), output_dir / 'bulgarian_squat_model.pt')
    print(f"✓ Guardado: {output_dir}/bulgarian_squat_model.pt")
    
    # 3. Copiar archivos de configuración
    print("\n[3/5] Copiando archivos de configuración...")
    import shutil
    shutil.copy('models/best/run_meta.json', output_dir / 'run_meta.json')
    shutil.copy('models/best/class_names.json', output_dir / 'class_names.json')
    shutil.copy('models/best/complete_metrics.json', output_dir / 'complete_metrics.json')
    shutil.copy('models/best/thr_per_class.npy', output_dir / 'thr_per_class.npy')
    print("✓ Archivos de configuración copiados")
    
    # 4. Crear información del modelo
    print("\n[4/5] Generando información del modelo...")
    with open('models/best/run_meta.json', 'r') as f:
        meta = json.load(f)
    
    with open('models/best/complete_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    model_info = {
        'nombre': 'Bulgarian Split Squat Posture Classifier',
        'framework': 'PyTorch',
        'arquitectura': 'BiGRU + Attention + BatchNorm',
        'version': '1.0.0',
        'parametros_totales': num_params,
        'input_shape': [None, 66],
        'output_shape': [4],
        'clases': [
            'E0_correcta',
            'E1_inclinacion_tronco', 
            'E2_valgo_rodilla',
            'E3_profundidad_insuficiente'
        ],
        'metricas': {
            'f1_macro': metrics.get('f1_macro', 0.5198),
            'f1_micro': metrics.get('f1_micro', 0.5838),
            'accuracy': metrics.get('accuracy', 0.6574)
        },
        'configuracion': {
            'hidden_sizes': [128, 64],
            'dropout': 0.3,
            'num_classes': 4,
            'input_dim': 66
        },
        'umbrales_optimos': meta.get('thr_per_class', [0.31, 0.19, 0.10, 0.70]),
        'entrenamiento': {
            'epochs': meta.get('epochs', 50),
            'batch_size': meta.get('batch_size', 32),
            'learning_rate': meta.get('lr', 0.001)
        }
    }
    
    with open(output_dir / 'MODEL_INFO.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"✓ Guardado: {output_dir}/MODEL_INFO.json")
    
    # 5. Crear README para el modelo
    print("\n[5/5] Creando documentación...")
    readme_content = """# Modelo Entrenado - Bulgarian Split Squat Classifier

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
- **Parámetros**: {:,}
- **Input**: Secuencias de 66 features (33 landmarks × 2 coordenadas)
- **Output**: 4 clases (multi-label)

## 📊 Métricas

- **F1-Score Macro**: {:.2%}
- **F1-Score Micro**: {:.2%}
- **Accuracy**: {:.2%}

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
""".format(num_params, 
           model_info['metricas']['f1_macro'],
           model_info['metricas']['f1_micro'],
           model_info['metricas']['accuracy'])
    
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Guardado: {output_dir}/README.md")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("✓ MODELO PREPARADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\nCarpeta de entrega: {output_dir.absolute()}")
    print("\nArchivos generados:")
    for file in sorted(output_dir.glob('*')):
        size_kb = file.stat().st_size / 1024
        print(f"  • {file.name} ({size_kb:.1f} KB)")
    
    print("\n📦 Archivos listos para entrega:")
    print(f"  1. Modelo entrenado: bulgarian_squat_model.pt")
    print(f"  2. Información: MODEL_INFO.json")
    print(f"  3. Configuración: run_meta.json, class_names.json")
    print(f"  4. Métricas: complete_metrics.json")
    print(f"  5. Documentación: README.md")

if __name__ == '__main__':
    export_model()
