# src/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_val_f1(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history)+1), history, marker='o')
    plt.xlabel("Época")
    plt.ylabel("Macro-F1 (val)")
    plt.title("Curva de validación (Macro-F1)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_confusion_per_class(Y_true, Y_pred, class_names, out_dir):
    """
    Multi-etiqueta: dibuja una confusión binaria por cada clase.
    Guarda un PNG por clase en out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    K = Y_true.shape[1]
    if class_names is None or len(class_names) != K:
        class_names = [f"C{i}" for i in range(K)]

    for k in range(K):
        yt = Y_true[:, k].astype(int)
        yp = Y_pred[:, k].astype(int)
        cm = confusion_matrix(yt, yp, labels=[0,1])

        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Matriz de confusión: {class_names[k]}")
        plt.xticks([0,1], ["0","1"])
        plt.yticks([0,1], ["0","1"])
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"confusion_{class_names[k]}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
