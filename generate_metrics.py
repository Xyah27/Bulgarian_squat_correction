"""
Generar Metricas Completas desde Modelo Existente
==================================================

Carga el modelo ya entrenado y genera todas las metricas detalladas.

Uso:
    python generate_metrics.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import json
from tqdm import tqdm

from bulgarian_squat.model_improved import BiGRUClassifierImproved
from bulgarian_squat.datamodule import RepsDataset, make_loader
from bulgarian_squat.data_utils import load_and_clean, build_repetitions_from_df
from bulgarian_squat.labels import build_labels_from_df_or_rules
from bulgarian_squat.splits import split_by_video

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_NAMES = ["correcta", "E1_tronco", "E2_valgo", "E3_profundidad"]


def compute_optimal_thresholds(probs, targets, num_classes=4):
    """Calcula umbrales optimos por clase usando F1"""
    thresholds = []
    for i in range(num_classes):
        best_thr, best_f1 = 0.5, 0
        for thr in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] > thr).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(best_thr)
    return np.array(thresholds)


def main():
    print("\n" + "="*70)
    print("GENERACION DE METRICAS COMPLETAS")
    print("="*70 + "\n")
    
    # Paths
    model_dir = project_root / "models" / "best"
    dataset_path = project_root / "data" / "raw" / "dataset_procesado_with_numFrames_nameVideo_etiquetado.csv"
    
    # Cargar metadata del modelo
    with open(model_dir / "run_meta.json") as f:
        meta = json.load(f)
    
    print(f"Modelo: {model_dir / 'best_model_bigru.pt'}")
    print(f"Dataset: {dataset_path.name}")
    if 'total_params' in meta:
        print(f"Parametros: {meta['total_params']:,}\n")
    else:
        print()
    
    # Cargar modelo
    print("Cargando modelo...")
    model = BiGRUClassifierImproved(
        in_dim=meta["in_dim"],
        hidden1=meta["hidden_dim1"],
        hidden2=meta["hidden_dim2"],
        num_classes=meta["num_classes"],
        dropout=meta.get("dropout", 0.3),
        use_batch_norm=True,
        use_attention=True
    )
    model.load_state_dict(torch.load(model_dir / "best_model_bigru.pt", map_location="cpu"))
    model.eval()
    print("  OK\n")
    
    # Cargar datos
    print("Cargando dataset...")
    df, x_cols, y_cols, z_cols, v_cols, meta_cols = load_and_clean(str(dataset_path))
    print(f"  {len(df)} frames cargados")
    print(f"  Columnas: x={len(x_cols)}, y={len(y_cols)}, z={len(z_cols)}, v={len(v_cols)}")
    
    print("\nExtrayendo repeticiones...")
    video_col = 'name_video' if 'name_video' in meta_cols else 'video_id'
    
    # Usar solo X,Y (66 dims) ya que el modelo fue entrenado sin Z
    use_z = meta.get("use_z", False)
    print(f"  Usando Z coords: {use_z}")
    
    if not use_z:
        z_cols = []  # No usar coordenadas Z
    
    reps, rep_meta = build_repetitions_from_df(df, x_cols, y_cols, z_cols, v_cols, video_col=video_col, fps=30)
    print(f"  {len(reps)} repeticiones extraidas")
    
    # Verificar dimensión de los datos
    if len(reps) > 0:
        sample_rep = reps[0] if isinstance(reps, list) else reps[0]
        print(f"  Dimension de repeticion: {sample_rep.shape}")
    
    print("\nGenerando labels...")
    angle_rules = {"trunk_incline_deg": 15.0, "knee_min_deg": 90.0}
    labels_result = build_labels_from_df_or_rules(df, rep_meta, video_col, reps, angle_rules=angle_rules)
    if isinstance(labels_result, tuple):
        labels, _, _ = labels_result
    else:
        labels = labels_result
    print(f"  Labels shape: {labels.shape}")
    
    # Split dataset
    print("\nDividiendo dataset...")
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    vids_train, vids_val, vids_test = split_by_video(rep_meta, video_col=video_col, splits=splits, seed=42)
    
    idx_train = [i for i, m in enumerate(rep_meta) if m[video_col] in vids_train]
    idx_val = [i for i, m in enumerate(rep_meta) if m[video_col] in vids_val]
    idx_test = [i for i, m in enumerate(rep_meta) if m[video_col] in vids_test]
    
    print(f"  Train: {len(idx_train)} reps")
    print(f"  Val:   {len(idx_val)} reps")
    print(f"  Test:  {len(idx_test)} reps")
    
    # Convertir a lista si es necesario
    if isinstance(reps, dict):
        reps_list = [reps[i] for i in range(len(reps))]
    else:
        reps_list = reps
    
    # Crear datasets
    val_ds = RepsDataset([reps_list[i] for i in idx_val], labels[idx_val])
    test_ds = RepsDataset([reps_list[i] for i in idx_test], labels[idx_test])
    
    val_dl = make_loader(val_ds, bs=32, shuffle=False)
    test_dl = make_loader(test_ds, bs=32, shuffle=False)
    
    # Evaluar en validacion
    print("\n" + "="*70)
    print("EVALUANDO EN VALIDATION SET")
    print("="*70 + "\n")
    
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, Y, M in tqdm(val_dl, desc="  Procesando", ncols=80):
            logits = model(X, M)
            val_preds.append(torch.sigmoid(logits).numpy())
            val_targets.append(Y.numpy())
    
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    # Calcular umbrales optimos
    print("\nCalculando umbrales optimos...")
    thresholds = compute_optimal_thresholds(val_preds, val_targets, 4)
    print(f"  Umbrales: {thresholds}")
    
    # Guardar umbrales
    np.save(model_dir / "thr_per_class.npy", thresholds)
    print(f"  Guardado: {model_dir / 'thr_per_class.npy'}")
    
    # Evaluar en test
    print("\n" + "="*70)
    print("EVALUANDO EN TEST SET")
    print("="*70 + "\n")
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X, Y, M in tqdm(test_dl, desc="  Procesando", ncols=80):
            logits = model(X, M)
            test_preds.append(torch.sigmoid(logits).numpy())
            test_targets.append(Y.numpy())
    
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # Aplicar umbrales
    test_pred_binary = (test_preds > thresholds).astype(int)
    test_pred_classes = np.argmax(test_pred_binary, axis=1)
    test_true_classes = np.argmax(test_targets, axis=1)
    
    # Metricas
    accuracy = (test_pred_classes == test_true_classes).mean()
    f1_macro = f1_score(test_true_classes, test_pred_classes, average='macro', zero_division=0)
    f1_weighted = f1_score(test_true_classes, test_pred_classes, average='weighted', zero_division=0)
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        test_true_classes, test_pred_classes, average=None, zero_division=0
    )
    
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    
    print("\n" + "="*70)
    print("REPORTE DE CLASIFICACION")
    print("="*70 + "\n")
    
    # Obtener las clases presentes en test
    classes_present = sorted(np.unique(test_true_classes))
    target_names_present = [CLASS_NAMES[i] for i in classes_present]
    
    print(classification_report(
        test_true_classes,
        test_pred_classes,
        labels=classes_present,
        target_names=target_names_present,
        digits=4,
        zero_division=0
    ))
    
    print("\nMatriz de Confusion:")
    print(cm)
    
    print(f"\nMETRICAS FINALES:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Macro:    {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  F1 Weighted: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    
    # Guardar metricas completas
    print("\n" + "="*70)
    print("GUARDANDO METRICAS")
    print("="*70 + "\n")
    
    complete_metrics = {
        "test_metrics": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "support_per_class": support.tolist()
        },
        "confusion_matrix": cm.tolist(),
        "thresholds": thresholds.tolist(),
        "class_names": CLASS_NAMES,
        "dataset_info": {
            "total_reps": len(reps_list),
            "train_size": len(idx_train),
            "val_size": len(idx_val),
            "test_size": len(idx_test)
        }
    }
    
    with open(model_dir / "complete_metrics.json", "w") as f:
        json.dump(complete_metrics, f, indent=2)
    
    print(f"  OK: {model_dir / 'complete_metrics.json'}")
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"\nRESULTADOS:")
    print(f"  Test F1 Macro:  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Umbrales:       {thresholds}")
    print(f"\nArchivos generados:")
    print(f"  {model_dir / 'complete_metrics.json'}")
    print(f"  {model_dir / 'thr_per_class.npy'}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
