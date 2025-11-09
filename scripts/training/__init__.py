"""
scripts.training
================

Módulo de scripts de entrenamiento para el proyecto Bulgarian Split Squat.

Este módulo contiene scripts para entrenar y evaluar modelos de clasificación
de ejercicios usando estimación de pose con MediaPipe.

Módulos disponibles:
-------------------
- train_model: Script principal de entrenamiento
- train_bigru: Entrenamiento específico del modelo BiGRU
- train_from_scratch: Pipeline completo de entrenamiento desde cero

Funciones principales:
---------------------
- Configuración de hiperparámetros
- Pipeline de entrenamiento
- Validación y early stopping
- Guardado de checkpoints
- Generación de métricas

Ejemplo de uso:
--------------
    from scripts.training import train_model
    
    # O ejecutar desde línea de comandos:
    # python scripts/training/train_bigru.py --epochs 100 --batch_size 32

Autor: Juan Jose Núñez, Juan Jose Castro
Institución: Universidad San Buenaventura
"""

__version__ = "1.0.0"
__author__ = "Juan Jose Núñez, Juan Jose Castro"

# Importar funciones principales (si existen en los módulos)
__all__ = [
    'train_model',
    'train_bigru',
]

# Metadata del módulo
SUPPORTED_MODELS = ['BiGRU', 'LSTM', 'GRU', 'Transformer']
DEFAULT_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.001,
    'patience': 15,
    'hidden_sizes': [128, 64],
    'dropout': 0.3,
    'num_classes': 4,
}
