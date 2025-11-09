"""
Script para generar todas las figuras necesarias para el paper
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pandas as pd

# Configuración de estilo
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Directorios
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models" / "best"
OUTPUT_DIR = BASE_DIR / "docs" / "papers" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar métricas
with open(MODEL_DIR / "complete_metrics.json", 'r') as f:
    metrics = json.load(f)

print("="*70)
print("GENERANDO FIGURAS PARA EL PAPER")
print("="*70)

# ============================================================================
# 1. DISTRIBUCIÓN DEL DATASET
# ============================================================================
print("\n[1/6] Generando distribución del dataset...")

class_names = ["E0 (Correcta)", "E1 (Tronco)", "E2 (Valgo)", "E3 (Profundidad)"]
class_counts = [46, 771, 0, 3]
class_percentages = [5.6, 94.0, 0.0, 0.4]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax.bar(class_names, class_counts, color=colors, alpha=0.7, edgecolor='black')

# Añadir valores en las barras
for i, (bar, count, pct) in enumerate(zip(bars, class_counts, class_percentages)):
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Número de Repeticiones', fontsize=12)
ax.set_xlabel('Clase de Error', fontsize=12)
ax.set_title('Distribución de Clases en el Dataset (820 repeticiones totales)', 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(class_counts) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dataset_distribution.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "dataset_distribution.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: dataset_distribution.pdf")

# ============================================================================
# 2. CURVAS DE ENTRENAMIENTO
# ============================================================================
print("\n[2/6] Generando curvas de entrenamiento...")

history = metrics['training_history']
epochs = list(range(1, len(history['train_loss']) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax = axes[0]
ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
# Marcar mejor epoch
best_epoch = history.get('best_epoch', 30)
ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Curvas de Pérdida durante Entrenamiento', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# F1 Score
ax = axes[1]
ax.plot(epochs, history['train_f1'], 'b-', linewidth=2, label='Train F1', marker='o', markersize=3)
ax.plot(epochs, history['val_f1'], 'r-', linewidth=2, label='Val F1', marker='s', markersize=3)
ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Curvas de F1 Score durante Entrenamiento', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bigru_comparison_training.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "bigru_comparison_training.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: bigru_comparison_training.pdf")

# ============================================================================
# 3. COMPARACIÓN DE MODELOS (ABLACIÓN)
# ============================================================================
print("\n[3/6] Generando comparación de modelos...")

# Datos de ablación (resultados históricos vs actuales)
models = ['Split por\nvideo', 'Split + class\nweights', 'Split\nestratificado']
macro_f1 = [4.99, 20.61, 66.38]
accuracy = [7.86, 20.90, 98.37]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Macro F1
ax = axes[0]
bars = ax.bar(models, macro_f1, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, macro_f1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Macro-F1 Score (%)', fontsize=12)
ax.set_title('Comparación Macro-F1: Estrategias de Split', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(macro_f1) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Accuracy
ax = axes[1]
bars = ax.bar(models, accuracy, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, accuracy):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Comparación Accuracy: Estrategias de Split', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(accuracy) * 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bigru_results_comparison.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "bigru_results_comparison.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: bigru_results_comparison.pdf")

# ============================================================================
# 4. MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n[4/6] Generando matriz de confusión...")

# Matriz de confusión del test set (valores reales del entrenamiento)
conf_matrix = np.array([
    [7, 0, 0],      # E0 (Correcta): 7/7 correctas
    [0, 114, 0],    # E1 (Tronco): 114/114 correctas
    [0, 2, 0]       # E3 (Profundidad): 0/2 correctas, 2 clasificadas como E1
])

# Clases presentes (sin E2)
present_classes = ["E0\n(Correcta)", "E1\n(Tronco)", "E3\n(Profund.)"]

# Normalizar por filas (recall)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Reemplazar NaN con 0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz absoluta
ax = axes[0]
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=present_classes, yticklabels=present_classes,
            cbar_kws={'label': 'Número de muestras'}, ax=ax, linewidths=1, linecolor='black')
ax.set_ylabel('Clase Real', fontsize=12)
ax.set_xlabel('Clase Predicha', fontsize=12)
ax.set_title('Matriz de Confusión (Valores Absolutos)', fontsize=13, fontweight='bold')

# Matriz normalizada
ax = axes[1]
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=present_classes, yticklabels=present_classes,
            cbar_kws={'label': 'Proporción'}, ax=ax, linewidths=1, linecolor='black', vmin=0, vmax=1)
ax.set_ylabel('Clase Real', fontsize=12)
ax.set_xlabel('Clase Predicha', fontsize=12)
ax.set_title('Matriz de Confusión Normalizada (Recall)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: confusion_matrix_normalized.pdf")

# ============================================================================
# 5. MÉTRICAS POR CLASE
# ============================================================================
print("\n[5/6] Generando métricas por clase...")

# Datos del classification report
classes_present = ["E0\n(Correcta)", "E1\n(Tronco)", "E3\n(Profundidad)"]
precision = [100.0, 98.28, 0.0]
recall = [100.0, 100.0, 0.0]
f1_score = [100.0, 99.13, 0.0]

x = np.arange(len(classes_present))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c', alpha=0.8, edgecolor='black')

# Añadir valores en las barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 5:  # Solo mostrar si es visible
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Porcentaje (%)', fontsize=12)
ax.set_xlabel('Clase de Error', fontsize=12)
ax.set_title('Métricas de Clasificación por Clase (Test Set)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes_present)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "per_class_metrics.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "per_class_metrics.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: per_class_metrics.pdf")

# ============================================================================
# 6. ARQUITECTURA DEL MODELO (Diagrama simplificado)
# ============================================================================
print("\n[6/6] Generando diagrama de arquitectura...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Definir bloques de arquitectura
layers = [
    ("Input\n(T x 66)", 0.5, 0.9, '#ecf0f1'),
    ("BatchNorm", 0.5, 0.80, '#3498db'),
    ("BiGRU Layer 1\n(128 hidden)", 0.5, 0.68, '#2ecc71'),
    ("LayerNorm +\nDropout (0.3)", 0.5, 0.58, '#f39c12'),
    ("BiGRU Layer 2\n(64 hidden)", 0.5, 0.46, '#2ecc71'),
    ("LayerNorm +\nDropout (0.3)", 0.5, 0.36, '#f39c12'),
    ("Attention\nMechanism", 0.5, 0.24, '#e74c3c'),
    ("FC Layer\n(128 → 64)", 0.5, 0.14, '#9b59b6'),
    ("Output\n(4 classes)", 0.5, 0.04, '#34495e'),
]

box_width = 0.25
box_height = 0.08

for label, x, y, color in layers:
    # Dibujar rectángulo
    rect = plt.Rectangle((x - box_width/2, y - box_height/2), 
                          box_width, box_height, 
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Añadir texto
    ax.text(x, y, label, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white' if color != '#ecf0f1' else 'black')
    
    # Añadir flecha hacia abajo (excepto último)
    if y > 0.05:
        ax.arrow(x, y - box_height/2 - 0.01, 0, -0.03, 
                head_width=0.03, head_length=0.015, fc='black', ec='black', linewidth=1.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Arquitectura BiGRU+Attention (292K parámetros)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bigru_architecture.pdf", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "bigru_architecture.png", bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: bigru_architecture.pdf")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "="*70)
print("FIGURAS GENERADAS EXITOSAMENTE")
print("="*70)
print(f"\nDirectorio de salida: {OUTPUT_DIR}")
print("\nArchivos generados:")
print("  1. dataset_distribution.pdf/.png")
print("  2. bigru_comparison_training.pdf/.png")
print("  3. bigru_results_comparison.pdf/.png")
print("  4. confusion_matrix_normalized.pdf/.png")
print("  5. per_class_metrics.pdf/.png")
print("  6. bigru_architecture.pdf/.png")
print("\n✓ Todas las figuras están listas para el paper LaTeX")
print("="*70)
