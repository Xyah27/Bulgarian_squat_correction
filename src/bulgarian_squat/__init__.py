"""
Bulgarian Split Squat - Posture Analysis System
================================================

Sistema de análisis de postura para ejercicio Bulgarian Split Squat
usando MediaPipe Pose y modelos BiGRU con atención.

Módulos principales:
- model_improved: Arquitectura BiGRU+Attention
- datamodule: Carga y procesamiento de datos
- train: Entrenamiento de modelos
- eval: Evaluación y métricas
- rt_infer: Inferencia en tiempo real con webcam
- config: Configuración global
"""

__version__ = "1.0.0"
__author__ = "Tu Nombre"
__email__ = "tu.email@example.com"

from .model_improved import BiGRUClassifierImproved
from .rt_infer import PoseStreamer, RepDetector

__all__ = [
    "BiGRUClassifierImproved",
    "PoseStreamer",
    "RepDetector",
]
