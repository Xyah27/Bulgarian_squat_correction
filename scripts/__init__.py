"""
scripts
=======

Módulo principal de scripts para el proyecto Bulgarian Split Squat.

Este paquete contiene todos los scripts ejecutables del proyecto,
organizados en submódulos según su funcionalidad.

Estructura:
----------
scripts/
├── __init__.py              (este archivo)
├── training/                (Scripts de entrenamiento)
│   ├── __init__.py
│   ├── train_model.py
│   └── train_bigru.py
├── inference/               (Scripts de inferencia)
│   ├── __init__.py
│   └── run_webcam.py
├── utils/                   (Utilidades y helpers)
│   ├── __init__.py
│   ├── visualization.py
│   └── metrics.py
└── data_processing/         (Procesamiento de datos)
    ├── __init__.py
    └── prepare_dataset.py

Submódulos:
----------
- training: Entrenamiento de modelos
- inference: Inferencia en tiempo real y batch
- utils: Funciones de utilidad compartidas
- data_processing: Preparación y augmentación de datos

Uso típico:
----------
    # Entrenar modelo
    python scripts/training/train_bigru.py --epochs 100
    
    # Inferencia en tiempo real
    python scripts/inference/run_webcam.py --model models/best/bigru_best.pth
    
    # Importar utilidades
    from scripts.utils import plot_training_history

Configuración global:
-------------------
Todos los scripts comparten configuraciones comunes definidas en
src/bulgarian_squat/config.py

Autor: Juan Jose Núñez, Juan Jose Castro
Institución: Universidad San Buenaventura
Proyecto: Evaluación de Bulgarian Split Squat con Deep Learning
"""

__version__ = "1.0.0"
__author__ = "Juan Jose Núñez, Juan Jose Castro"
__project__ = "Bulgarian Split Squat - Vision por Computadora"
__institution__ = "Universidad San Buenaventura, Cali, Colombia"

# Importar submódulos principales
from . import training
from . import inference

# Intentar importar utils si existe
try:
    from . import utils
except ImportError:
    utils = None

# Intentar importar data_processing si existe
try:
    from . import data_processing
except ImportError:
    data_processing = None

__all__ = [
    'training',
    'inference',
    'utils',
    'data_processing',
]

# Metadata del proyecto
PROJECT_INFO = {
    'name': 'Bulgarian Split Squat Evaluation',
    'version': __version__,
    'authors': __author__,
    'institution': __institution__,
    'description': 'Sistema automatizado de evaluación de técnica de ejercicio usando MediaPipe y BiGRU',
    'keywords': ['computer vision', 'pose estimation', 'exercise evaluation', 'BiGRU', 'MediaPipe'],
    'classes': ['E0_correcta', 'E1_tronco', 'E2_valgo', 'E3_profundidad'],
    'accuracy': '98.37%',
    'macro_f1': '66.38%',
}

def print_project_info():
    """Imprime información del proyecto"""
    print("\n" + "="*70)
    print(f"{PROJECT_INFO['name']}")
    print("="*70)
    print(f"Versión: {PROJECT_INFO['version']}")
    print(f"Autores: {PROJECT_INFO['authors']}")
    print(f"Institución: {PROJECT_INFO['institution']}")
    print(f"\n{PROJECT_INFO['description']}")
    print(f"\nClases: {', '.join(PROJECT_INFO['classes'])}")
    print(f"Accuracy: {PROJECT_INFO['accuracy']}")
    print(f"Macro-F1: {PROJECT_INFO['macro_f1']}")
    print("="*70 + "\n")

def get_version():
    """Retorna la versión del proyecto"""
    return __version__

def list_available_scripts():
    """Lista todos los scripts disponibles"""
    scripts_list = {
        'Training': [
            'scripts/training/train_model.py',
            'scripts/training/train_bigru.py',
        ],
        'Inference': [
            'scripts/inference/run_webcam.py',
        ],
        'Utilities': [
            'scripts/utils/visualization.py',
            'scripts/utils/metrics.py',
        ]
    }
    
    print("\n" + "="*70)
    print("SCRIPTS DISPONIBLES")
    print("="*70)
    for category, scripts in scripts_list.items():
        print(f"\n{category}:")
        for script in scripts:
            print(f"  • {script}")
    print("="*70 + "\n")
    
    return scripts_list
