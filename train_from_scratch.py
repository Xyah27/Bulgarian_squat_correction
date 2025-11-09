"""
Pipeline Completo de Entrenamiento - Bulgarian Split Squat
===========================================================

Entrena el modelo BiGRU+Attention desde cero y genera todas las metricas.

Uso:
    python train_from_scratch.py
    python train_from_scratch.py --epochs 100 --batch_size 32
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm
import argparse

from bulgarian_squat.model_improved import BiGRUClassifierImproved
from bulgarian_squat.datamodule import RepsDataset, make_loader
from bulgarian_squat.data_utils import load_and_clean, build_repetitions_from_df
from bulgarian_squat.labels import build_labels_from_df_or_rules
from bulgarian_squat.splits import split_by_video

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

# Nombres de clases
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
    parser = argparse.ArgumentParser(description="Entrenar modelo desde cero")
    parser.add_argument("--dataset", type=str, 
                       default="data/raw/dataset_procesado_with_numFrames_nameVideo_etiquetado.csv",
                       help="Ruta al dataset CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden size capa 1")
    parser.add_argument("--hidden2", type=int, default=64, help="Hidden size capa 2")
    parser.add_argument("--output", type=str, default="models/best", help="Directorio de salida")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETO - BULGARIAN SPLIT SQUAT")
    print("="*70 + "\n")
    
    # Configuracion
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output
    checkpoint_dir = project_root / "models" / "checkpoints" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "in_dim": 66,  # Solo X,Y (sin Z)
        "hidden_sizes": [args.hidden1, args.hidden2],
        "num_classes": 4,
        "dropout": 0.3,
        "attention": True,
        "use_z": False,  # No usar coordenadas Z
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Dataset: {dataset_path.name}")
    print(f"Output: {output_dir}")
    print(f"Device: {config['device']}")
    print(f"Config: hidden={config['hidden_sizes']}, lr={config['lr']}, epochs={config['epochs']}\n")
    
    # PASO 1: Cargar datos
    print("="*70)
    print("PASO 1: CARGANDO DATASET")
    print("="*70 + "\n")
    
    print("Cargando CSV...")
    df, x_cols, y_cols, z_cols, v_cols, meta_cols = load_and_clean(str(dataset_path))
    print(f"  OK: {len(df)} frames cargados")
    print(f"  Columnas: x={len(x_cols)}, y={len(y_cols)}, z={len(z_cols)}")
    
    print("\nExtrayendo repeticiones...")
    video_col = 'name_video' if 'name_video' in meta_cols else 'video_id'
    
    # No usar Z ya que el modelo es 66-dim (solo X,Y)
    if not config["use_z"]:
        z_cols = []
        print("  Usando solo coordenadas X,Y (66 dims)")
    
    reps, rep_meta = build_repetitions_from_df(df, x_cols, y_cols, z_cols, v_cols, video_col=video_col, fps=30)
    print(f"  OK: {len(reps)} repeticiones extraidas")
    
    # Verificar dimension
    if len(reps) > 0:
        sample_rep = reps[0] if isinstance(reps, list) else reps[0]
        print(f"  Dimension por frame: {sample_rep.shape[1]} (esperado: {config['in_dim']})")
    
    print("\nGenerando labels...")
    angle_rules = {"trunk_incline_deg": 15.0, "knee_min_deg": 90.0}
    labels_result = build_labels_from_df_or_rules(df, rep_meta, video_col, reps, angle_rules=angle_rules)
    if isinstance(labels_result, tuple):
        labels, _, _ = labels_result
    else:
        labels = labels_result
    print(f"  OK: Labels shape = {labels.shape}")
    
    class_distribution = np.bincount([np.argmax(l) for l in labels])
    print(f"\nDistribucion de clases:")
    for i, count in enumerate(class_distribution):
        print(f"  {CLASS_NAMES[i]}: {count} ({100*count/len(labels):.1f}%)")
    
    # PASO 2: Split dataset
    print("\n" + "="*70)
    print("PASO 2: DIVIDIENDO DATASET (ESTRATIFICADO)")
    print("="*70 + "\n")
    
    # Usar split estratificado, pero agrupar clases muy minoritarias
    # para estratificacion (E2_valgo=0, E3_profundidad=3 son muy pocas)
    from sklearn.model_selection import train_test_split
    
    # Obtener clases
    y_classes = np.argmax(labels, axis=1)
    
    # Para estratificacion, agrupar clases minoritarias como "E1_tronco"
    # ya que no podemos estratificar con clases que tienen <2 ejemplos
    y_strat = y_classes.copy()
    y_strat[y_strat >= 2] = 1  # E2_valgo y E3_profundidad -> E1_tronco para stratify
    
    # First split: train vs (val+test)
    idx_all = np.arange(len(reps))
    idx_train, idx_temp, _, _ = train_test_split(
        idx_all, y_strat, 
        test_size=0.30, 
        stratify=y_strat,
        random_state=42
    )
    
    # Second split: val vs test
    y_temp_strat = y_strat[idx_temp]
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp_strat,
        test_size=0.50,  # 0.50 of 0.30 = 0.15 of total
        stratify=y_temp_strat,
        random_state=42
    )
    
    print(f"  Train: {len(idx_train)} reps ({100*len(idx_train)/len(reps):.1f}%)")
    print(f"  Val:   {len(idx_val)} reps ({100*len(idx_val)/len(reps):.1f}%)")
    print(f"  Test:  {len(idx_test)} reps ({100*len(idx_test)/len(reps):.1f}%)")
    
    # Mostrar distribucion por split
    train_labels = labels[idx_train]
    val_labels = labels[idx_val]
    test_labels = labels[idx_test]
    
    train_dist = np.bincount([np.argmax(l) for l in train_labels], minlength=4)
    val_dist = np.bincount([np.argmax(l) for l in val_labels], minlength=4)
    test_dist = np.bincount([np.argmax(l) for l in test_labels], minlength=4)
    
    print(f"\n  Distribucion Train:")
    for i in range(len(CLASS_NAMES)):
        if train_dist[i] > 0:
            print(f"    {CLASS_NAMES[i]}: {train_dist[i]} ({100*train_dist[i]/len(idx_train):.1f}%)")
    
    print(f"\n  Distribucion Val:")
    for i in range(len(CLASS_NAMES)):
        if val_dist[i] > 0:
            print(f"    {CLASS_NAMES[i]}: {val_dist[i]} ({100*val_dist[i]/len(idx_val):.1f}%)")
    
    print(f"\n  Distribucion Test:")
    for i in range(len(CLASS_NAMES)):
        if test_dist[i] > 0:
            print(f"    {CLASS_NAMES[i]}: {test_dist[i]} ({100*test_dist[i]/len(idx_test):.1f}%)")
    
    # Convertir a lista si es necesario
    if isinstance(reps, dict):
        reps_list = [reps[i] for i in range(len(reps))]
    else:
        reps_list = reps
    
    # Crear datasets
    train_ds = RepsDataset([reps_list[i] for i in idx_train], labels[idx_train])
    val_ds = RepsDataset([reps_list[i] for i in idx_val], labels[idx_val])
    test_ds = RepsDataset([reps_list[i] for i in idx_test], labels[idx_test])
    
    train_dl = make_loader(train_ds, bs=config["batch_size"], shuffle=True)
    val_dl = make_loader(val_ds, bs=config["batch_size"], shuffle=False)
    test_dl = make_loader(test_ds, bs=config["batch_size"], shuffle=False)
    
    # PASO 3: Crear modelo
    print("\n" + "="*70)
    print("PASO 3: CREANDO MODELO")
    print("="*70 + "\n")
    
    model = BiGRUClassifierImproved(
        in_dim=config["in_dim"],
        hidden1=config["hidden_sizes"][0],
        hidden2=config["hidden_sizes"][1],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        use_batch_norm=True,
        use_attention=True
    )
    
    device = torch.device(config["device"])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Arquitectura: BiGRU ({config['hidden_sizes'][0]}->{config['hidden_sizes'][1]}) + Attention")
    print(f"  Parametros: {total_params:,}")
    print(f"  Device: {device}")
    
    # PASO 4: Entrenamiento
    print("\n" + "="*70)
    print("PASO 4: ENTRENANDO MODELO")
    print("="*70 + "\n")
    
    optimizer = Adam(model.parameters(), lr=config["lr"])
    
    # Calcular pesos de clase para balancear el dataset
    train_class_counts = np.bincount([np.argmax(l) for l in labels[idx_train]], minlength=config["num_classes"])
    total_samples = len(idx_train)
    class_weights = torch.FloatTensor([total_samples / (config["num_classes"] * count) if count > 0 else 0.0 
                                       for count in train_class_counts]).to(device)
    
    print(f"Pesos de clase (para balancear):")
    for i, weight in enumerate(class_weights):
        print(f"  {CLASS_NAMES[i]}: {weight:.4f}")
    print()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_f1 = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_acc": [], "val_acc": []
    }
    
    # Tabla de cabecera
    print(f"{'Epoca':<7} {'Train Loss':<12} {'Train F1':<10} {'Train Acc':<11} {'Val Loss':<12} {'Val F1':<10} {'Val Acc':<11} {'Status':<15}")
    print("-" * 100)
    
    for epoch in range(config["epochs"]):
        # TRAIN
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        print(f"\n[Epoca {epoch+1}/{config['epochs']}] Entrenando...", end="", flush=True)
        
        for batch_idx, (X, Y, M) in enumerate(train_dl):
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            
            optimizer.zero_grad()
            logits = model(X, M)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).cpu().detach().numpy())
            train_targets.append(Y.cpu().numpy())
            
            # Mostrar progreso cada 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f".", end="", flush=True)
        
        print(" OK", flush=True)
        
        train_loss /= len(train_dl)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_pred_classes = np.argmax(train_preds, axis=1)
        train_true_classes = np.argmax(train_targets, axis=1)
        train_f1 = f1_score(train_true_classes, train_pred_classes, average='macro', zero_division=0)
        train_acc = (train_pred_classes == train_true_classes).mean()
        
        # VALIDATION
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        print(f"[Epoca {epoch+1}/{config['epochs']}] Validando...", end="", flush=True)
        
        with torch.no_grad():
            for X, Y, M in val_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X, M)
                loss = criterion(logits, Y)
                
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(Y.cpu().numpy())
        
        print(" OK", flush=True)
        
        val_loss /= len(val_dl)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_pred_classes = np.argmax(val_preds, axis=1)
        val_true_classes = np.argmax(val_targets, axis=1)
        val_f1 = f1_score(val_true_classes, val_pred_classes, average='macro', zero_division=0)
        val_acc = (val_pred_classes == val_true_classes).mean()
        
        # Guardar historial
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(val_f1))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        
        # Status
        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            status = "** BEST **"
        else:
            patience_counter += 1
            status = f"Patience {patience_counter}/{config['patience']}"
            if patience_counter >= config["patience"]:
                status = "EARLY STOP"
        
        # Imprimir fila de la tabla
        print(f"{epoch+1:<7} {train_loss:<12.4f} {train_f1:<10.4f} {train_acc:<11.4f} {val_loss:<12.4f} {val_f1:<10.4f} {val_acc:<11.4f} {status:<15}")
        
        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping en epoca {epoch+1}")
            break
    
    print("-" * 100)
    
    # PASO 5: Evaluacion final
    print("\n" + "="*70)
    print("PASO 5: EVALUACION EN TEST SET")
    print("="*70 + "\n")
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
    model.eval()
    
    # Evaluar en test
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X, Y, M in tqdm(test_dl, desc="  Evaluando", ncols=80):
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits = model(X, M)
            test_preds.append(torch.sigmoid(logits).cpu().numpy())
            test_targets.append(Y.cpu().numpy())
    
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # Calcular umbrales optimos
    print("\nCalculando umbrales optimos...")
    thresholds = compute_optimal_thresholds(val_preds, val_targets, config["num_classes"])
    print(f"  Umbrales: {thresholds}")
    
    # Aplicar umbrales
    test_pred_binary = (test_preds > thresholds).astype(int)
    test_pred_classes = np.argmax(test_pred_binary, axis=1)
    test_true_classes = np.argmax(test_targets, axis=1)
    
    # Metricas completas
    accuracy = (test_pred_classes == test_true_classes).mean()
    f1_macro = f1_score(test_true_classes, test_pred_classes, average='macro', zero_division=0)
    f1_weighted = f1_score(test_true_classes, test_pred_classes, average='weighted', zero_division=0)
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        test_true_classes, test_pred_classes, average=None, zero_division=0
    )
    
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    
    print("\n" + "="*70)
    print("REPORTE DE CLASIFICACION (TEST)")
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
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 Macro:    {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    
    # PASO 6: Guardar artifacts
    print("\n" + "="*70)
    print("PASO 6: GUARDANDO MODELO Y METRICAS")
    print("="*70 + "\n")
    
    # Limpiar directorio de salida (eliminar archivos anteriores)
    if output_dir.exists():
        import shutil
        for item in output_dir.glob("*"):
            if item.is_file():
                item.unlink()
                print(f"  Eliminado: {item.name}")
    
    # Copiar mejor modelo
    import shutil
    shutil.copy(checkpoint_dir / "best_model.pt", output_dir / "best_model_bigru.pt")
    print(f"\n  Guardado: {output_dir / 'best_model_bigru.pt'}")
    
    # Guardar metadata
    run_meta = {
        "in_dim": config["in_dim"],
        "num_classes": config["num_classes"],
        "use_z": False,
        "hidden_dim1": config["hidden_sizes"][0],
        "hidden_dim2": config["hidden_sizes"][1],
        "dropout": config["dropout"],
        "use_attention": config["attention"],
        "use_batch_norm": True,
        "model_type": "BiGRU+Attention+BatchNorm",
        "total_params": total_params,
        "best_val_f1": float(best_val_f1),
        "test_metrics": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        },
        "description": f"Modelo BiGRU+Attention ({f1_macro*100:.2f}% Macro-F1)"
    }
    
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"  Guardado: {output_dir / 'run_meta.json'}")
    
    # Guardar class names
    class_names_dict = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names_dict, f, indent=2)
    print(f"  Guardado: {output_dir / 'class_names.json'}")
    
    # Guardar umbrales
    np.save(output_dir / "thr_per_class.npy", thresholds)
    print(f"  Guardado: {output_dir / 'thr_per_class.npy'}")
    
    # Guardar metricas completas
    complete_metrics = {
        "config": config,
        "training_history": history,
        "best_val_f1": float(best_val_f1),
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
    
    with open(output_dir / "complete_metrics.json", "w") as f:
        json.dump(complete_metrics, f, indent=2)
    print(f"  Guardado: {output_dir / 'complete_metrics.json'}")
    
    # PASO 7: Tabla Resumen de Metricas
    print("\n" + "="*70)
    print("PASO 7: RESUMEN DE METRICAS")
    print("="*70 + "\n")
    
    print("METRICAS POR CLASE:")
    print("-" * 80)
    print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    # Solo mostrar clases presentes en test
    for idx, class_idx in enumerate(classes_present):
        name = CLASS_NAMES[class_idx]
        print(f"{name:<20} {precision[idx]:<12.4f} {recall[idx]:<12.4f} {f1_per_class[idx]:<12.4f} {int(support[idx]):<10}")
    
    print("-" * 80)
    print(f"{'PROMEDIO MACRO':<20} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1_macro:<12.4f} {int(support.sum()):<10}")
    print(f"{'PROMEDIO WEIGHTED':<20} {'-':<12} {'-':<12} {f1_weighted:<12.4f} {int(support.sum()):<10}")
    print("-" * 80)
    
    print("\nMATRIZ DE CONFUSION:")
    print("-" * 80)
    header = "True\\Pred".ljust(15)
    for class_idx in classes_present:
        header += CLASS_NAMES[class_idx][:10].ljust(15)
    print(header)
    print("-" * 80)
    for i, class_idx in enumerate(classes_present):
        row = CLASS_NAMES[class_idx][:12].ljust(15)
        for j in range(len(classes_present)):
            row += str(cm[i, j]).ljust(15)
        print(row)
    print("-" * 80)
    
    # PASO 7: Sistema listo
    print("\n" + "="*70)
    print("PASO 8: SISTEMA LISTO")
    print("="*70 + "\n")
    
    print("Para usar con camara:")
    print(f"  python scripts/inference/run_webcam.py --model {output_dir} --cam 1")
    
    print("\nArchivos generados en models/best/:")
    for item in sorted(output_dir.glob("*")):
        if item.is_file():
            size = item.stat().st_size / 1024  # KB
            print(f"  {item.name:<30} ({size:>8.1f} KB)")
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nRESULTADOS FINALES:")
    print(f"  Best Val F1:   {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    print(f"  Test F1 Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Parametros:    {total_params:,}")
    print(f"  Epocas:        {len(history['train_loss'])}/{config['epochs']}")
    print(f"  Output:        {output_dir}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
