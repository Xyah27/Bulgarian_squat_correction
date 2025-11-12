"""
scripts.utils
=============

Módulo de utilidades compartidas para el proyecto Bulgarian Split Squat.

Este módulo contiene funciones y clases de utilidad que son usadas
por múltiples scripts de entrenamiento e inferencia.

Módulos disponibles:
-------------------
- visualization: Funciones para visualizar datos y resultados
- metrics: Cálculo y reporte de métricas de evaluación
- data_helpers: Funciones auxiliares para manejo de datos
- video_utils: Utilidades para procesamiento de video

Funciones principales:
---------------------
Visualización:
- plot_training_history(): Graficar curvas de entrenamiento
- plot_confusion_matrix(): Visualizar matriz de confusión
- plot_per_class_metrics(): Métricas por clase
- visualize_attention_weights(): Visualizar pesos de atención

Métricas:
- compute_classification_metrics(): Calcular precision, recall, F1
- generate_classification_report(): Reporte completo de clasificación
- calculate_macro_f1(): Calcular Macro-F1 score
- compute_confusion_matrix(): Generar matriz de confusión

Video:
- extract_frames(): Extraer frames de video
- save_video_with_overlay(): Guardar video con overlay de predicciones
- resize_frame(): Redimensionar frame manteniendo aspect ratio

Data Helpers:
- normalize_landmarks(): Normalizar coordenadas de landmarks
- pad_sequence(): Rellenar secuencias a longitud fija
- stratified_split(): División estratificada de datos

Ejemplo de uso:
--------------
    from scripts.utils import plot_training_history, compute_classification_metrics
    
    # Visualizar historial de entrenamiento
    plot_training_history(history, save_path='results/training.pdf')
    
    # Calcular métricas
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    print(f"Macro-F1: {metrics['macro_f1']:.2%}")

Dependencias:
------------
- matplotlib: Visualizaciones
- seaborn: Gráficos estadísticos
- sklearn: Métricas de evaluación
- numpy: Operaciones numéricas
- pandas: Manejo de datos tabulares

Autor: Juan Jose Núñez, Juan Jose Castro
Institución: Universidad San Buenaventura
"""

__version__ = "1.0.0"
__author__ = "Juan Jose Núñez, Juan Jose Castro"

# ============================================================================
# IMPORTACIONES OPCIONALES
# ============================================================================
# Los siguientes módulos pueden no existir aún en el proyecto.
# Los errores de importación son manejados con try/except y son esperados.
# Si necesitas estas funcionalidades, crea los archivos correspondientes en:
#   - scripts/utils/visualization.py
#   - scripts/utils/metrics.py
#   - scripts/utils/video_utils.py
# ============================================================================

# Importar funciones principales de visualización
try:
    from .visualization import (  # type: ignore
        plot_training_history,
        plot_confusion_matrix,
        plot_per_class_metrics,
        visualize_attention_weights,
    )
except ImportError:
    # Si no existe el módulo, definir funciones dummy
    def plot_training_history(*args, **kwargs):
        raise NotImplementedError("Módulo visualization no implementado")
    
    def plot_confusion_matrix(*args, **kwargs):
        raise NotImplementedError("Módulo visualization no implementado")
    
    def plot_per_class_metrics(*args, **kwargs):
        raise NotImplementedError("Módulo visualization no implementado")
    
    def visualize_attention_weights(*args, **kwargs):
        raise NotImplementedError("Módulo visualization no implementado")

# Importar funciones de métricas
try:
    from .metrics import (  # type: ignore
        compute_classification_metrics,
        generate_classification_report,
        calculate_macro_f1,
        compute_confusion_matrix_custom,
    )
except ImportError:
    def compute_classification_metrics(*args, **kwargs):
        raise NotImplementedError("Módulo metrics no implementado")
    
    def generate_classification_report(*args, **kwargs):
        raise NotImplementedError("Módulo metrics no implementado")
    
    def calculate_macro_f1(*args, **kwargs):
        raise NotImplementedError("Módulo metrics no implementado")
    
    def compute_confusion_matrix_custom(*args, **kwargs):
        raise NotImplementedError("Módulo metrics no implementado")

# Importar utilidades de video
try:
    from .video_utils import (  # type: ignore
        extract_frames,
        save_video_with_overlay,
        resize_frame,
    )
except ImportError:
    def extract_frames(*args, **kwargs):
        raise NotImplementedError("Módulo video_utils no implementado")
    
    def save_video_with_overlay(*args, **kwargs):
        raise NotImplementedError("Módulo video_utils no implementado")
    
    def resize_frame(*args, **kwargs):
        raise NotImplementedError("Módulo video_utils no implementado")

# Exportar todo
__all__ = [
    # Visualización
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_per_class_metrics',
    'visualize_attention_weights',
    
    # Métricas
    'compute_classification_metrics',
    'generate_classification_report',
    'calculate_macro_f1',
    'compute_confusion_matrix_custom',
    
    # Video
    'extract_frames',
    'save_video_with_overlay',
    'resize_frame',
]

# Constantes útiles
CLASS_NAMES = [
    "E0_correcta",
    "E1_inclinacion_tronco",
    "E2_valgo_rodilla",
    "E3_profundidad_insuficiente"
]

CLASS_COLORS = {
    "E0_correcta": "#2ecc71",           # Verde
    "E1_inclinacion_tronco": "#f39c12",  # Naranja
    "E2_valgo_rodilla": "#e74c3c",      # Rojo
    "E3_profundidad_insuficiente": "#9b59b6",  # Morado
}

METRICS_NAMES = {
    'accuracy': 'Precisión Global',
    'macro_f1': 'Macro-F1 Score',
    'weighted_f1': 'F1 Ponderado',
    'precision': 'Precisión',
    'recall': 'Recall',
}

def get_class_names():
    """Retorna los nombres de las clases"""
    return CLASS_NAMES

def get_class_colors():
    """Retorna los colores por clase"""
    return CLASS_COLORS

def print_utils_info():
    """Imprime información sobre las utilidades disponibles"""
    print("\n" + "="*70)
    print("UTILIDADES DISPONIBLES - Bulgarian Split Squat")
    print("="*70)
    print("\nMódulos:")
    print("  • visualization: Funciones de visualización")
    print("  • metrics: Cálculo de métricas")
    print("  • video_utils: Procesamiento de video")
    print("  • data_helpers: Ayudantes para datos")
    print("\nClases reconocidas:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}. {name}")
    print("\nMétricas principales:")
    for key, name in METRICS_NAMES.items():
        print(f"  • {name} ({key})")
    print("="*70 + "\n")

# Configuración de visualización por defecto
PLOT_CONFIG = {
    'figsize': (10, 6),
    'dpi': 150,
    'style': 'seaborn-v0_8-darkgrid',
    'font_size': 12,
    'title_size': 14,
    'label_size': 11,
}

def get_plot_config():
    """Retorna configuración de plots por defecto"""
    return PLOT_CONFIG.copy()
