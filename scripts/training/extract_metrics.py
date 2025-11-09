"""
Script para extraer métricas completas del modelo ya entrenado
===============================================================

Re-evalúa best_model_bigru.pt en el dataset completo y genera todas las métricas.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from bulgarian_squat.model_bigru import BiGRUClassifier
from bulgarian_squat.datamodule import RepsDataset, make_loader
from bulgarian_squat.data_utils import load_and_clean, build_repetitions_from_df
from bulgarian_squat.labels import labels_multi
from bulgarian_squat.splits import split_by_video
from bulgarian_squat import config as cfg

def main():
    print("\n" + "="*60)
    print("EXTRACCIÓN DE MÉTRICAS - BiGRU+Attention")
    print("="*60 + "\n")
    
    # Rutas
    dataset_path = project_root / "data" / "raw" / "landmarks_dataset_BALANCEADO_v2.csv"
    model_path = project_root / "models" / "best" / "best_model_bigru.pt"
    meta_path = project_root / "models" / "best" / "run_meta.json"
    thresholds_path = project_root / "models" / "best" / "thr_per_class.npy"
    
    print(f"📂 Dataset: {dataset_path}")
    print(f"🤖 Modelo: {model_path}")
    
    # Cargar configuración
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Cargar umbrales
    thresholds = np.load(thresholds_path)
    print(f"🎯 Umbrales: {thresholds}\n")
    
    # Cargar datos
    print("📊 Cargando dataset...")
    df, x_cols, y_cols, z_cols, v_cols, meta_cols = load_and_clean(str(dataset_path))
    print(f"   ✓ {len(df)} frames cargados")
    
    # Extraer repeticiones
    print("🔄 Extrayendo repeticiones...")
    video_col = 'name_video' if 'name_video' in meta_cols else 'video_id'
    reps, rep_meta = build_repetitions_from_df(df, x_cols, y_cols, z_cols, v_cols, video_col=video_col)
    print(f"   ✓ {len(reps)} repeticiones extraídas")
    
    # Generar labels
    print("🏷️  Generando labels...")
    labels = labels_multi(rep_meta)
    print(f"   ✓ Labels: {labels.shape}")
    
    # Split por video
    print("\n📊 Dividiendo dataset por video...")
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    idx_train, idx_val, idx_test = split_by_video(rep_meta, video_col=video_col, splits=splits)
    
    print(f"   Train: {len(idx_train)} reps")
    print(f"   Val:   {len(idx_val)} reps")
    print(f"   Test:  {len(idx_test)} reps")
    
    # Crear datasets
    train_ds = RepsDataset([reps[i] for i in idx_train], labels[idx_train])
    val_ds = RepsDataset([reps[i] for i in idx_val], labels[idx_val])
    test_ds = RepsDataset([reps[i] for i in idx_test], labels[idx_test])
    
    train_dl = make_loader(train_ds, bs=32, shuffle=False)
    val_dl = make_loader(val_ds, bs=32, shuffle=False)
    test_dl = make_loader(test_ds, bs=32, shuffle=False)
    
    # Cargar modelo
    print("\n🤖 Cargando modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiGRUClassifier(
        in_dim=meta["in_dim"],
        hidden_sizes=[meta["hidden_dim1"], meta["hidden_dim2"]],
        num_classes=meta["num_classes"],
        dropout=meta["dropout"],
        attention=meta["use_attention"]
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Modelo cargado")
    print(f"   ✓ Parámetros: {total_params:,}")
    print(f"   ✓ Device: {device}")
    
    # Evaluar en cada split
    def evaluate_split(dataloader, split_name):
        preds, targets = [], []
        with torch.no_grad():
            for X, Y, M in dataloader:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X, M)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                targets.append(Y.cpu().numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        # Predicciones con umbrales
        pred_binary = (preds > thresholds).astype(int)
        pred_classes = np.argmax(pred_binary, axis=1)
        true_classes = np.argmax(targets, axis=1)
        
        # Métricas
        accuracy = (pred_classes == true_classes).mean()
        f1_macro = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
        f1_weighted = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            true_classes, pred_classes, average=None, zero_division=0
        )
        
        cm = confusion_matrix(true_classes, pred_classes)
        
        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "support_per_class": support.tolist(),
            "confusion_matrix": cm.tolist(),
            "pred_classes": pred_classes.tolist(),
            "true_classes": true_classes.tolist()
        }
    
    print("\n" + "="*60)
    print("EVALUACIÓN")
    print("="*60 + "\n")
    
    print("📊 Evaluando en Train...")
    train_metrics = evaluate_split(train_dl, "train")
    print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"   F1 Macro: {train_metrics['f1_macro']:.4f}")
    
    print("\n📊 Evaluando en Val...")
    val_metrics = evaluate_split(val_dl, "val")
    print(f"   Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   F1 Macro: {val_metrics['f1_macro']:.4f}")
    
    print("\n📊 Evaluando en Test...")
    test_metrics = evaluate_split(test_dl, "test")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   F1 Macro: {test_metrics['f1_macro']:.4f}")
    
    # Reporte detallado para test
    print("\n" + "="*60)
    print("REPORTE DETALLADO (TEST SET)")
    print("="*60 + "\n")
    
    print(classification_report(
        test_metrics['true_classes'],
        test_metrics['pred_classes'],
        target_names=cfg.CLASS_NAMES,
        digits=4
    ))
    
    print("\n📈 Matriz de Confusión (Test):")
    cm = np.array(test_metrics['confusion_matrix'])
    print(cm)
    
    # Guardar resultados completos
    results = {
        "model_config": meta,
        "total_params": total_params,
        "class_names": cfg.CLASS_NAMES,
        "thresholds": thresholds.tolist(),
        "dataset": {
            "total_reps": len(reps),
            "train_size": len(idx_train),
            "val_size": len(idx_val),
            "test_size": len(idx_test)
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        }
    }
    
    output_path = project_root / "models" / "best" / "complete_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Métricas completas guardadas en:")
    print(f"   {output_path}")
    
    # Imprimir tabla de métricas por clase
    print("\n" + "="*60)
    print("MÉTRICAS POR CLASE (TEST)")
    print("="*60 + "\n")
    
    print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        print(f"{class_name:<20} {test_metrics['precision_per_class'][i]:<12.4f} "
              f"{test_metrics['recall_per_class'][i]:<12.4f} "
              f"{test_metrics['f1_per_class'][i]:<12.4f} "
              f"{test_metrics['support_per_class'][i]:<10}")
    
    print("\n" + "="*60)
    print(f"🎯 RESULTADO FINAL - F1 Macro (Test): {test_metrics['f1_macro']:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
