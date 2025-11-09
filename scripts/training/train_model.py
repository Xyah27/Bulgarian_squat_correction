"""
Script de entrenamiento simplificado para BiGRU+Attention
=========================================================

Entrena el modelo usando el código existente del proyecto.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm

from bulgarian_squat.model_bigru import BiGRUClassifier
from bulgarian_squat.datamodule import RepsDataset, make_loader, collate_pad
from bulgarian_squat.data_utils import load_landmarks_csv, reps_from_df
from bulgarian_squat.labels import labels_multi
from bulgarian_squat.splits import split_by_video
from bulgarian_squat.train import train_one_epoch
from bulgarian_squat.eval import evaluate_model, compute_optimal_thresholds
from bulgarian_squat import config as cfg

def main():
    print("\n" + "="*60)
    print("ENTRENAMIENTO MODELO BiGRU + ATTENTION")
    print("="*60 + "\n")
    
    # Configuración
    dataset_path = project_root / "data" / "raw" / "landmarks_dataset_BALANCEADO_v2.csv"
    output_dir = project_root / "models" / "checkpoints" / f"bigru_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hiperparámetros
    config = {
        "in_dim": 66,
        "hidden_sizes": [128, 64],
        "num_classes": 4,
        "dropout": 0.3,
        "attention": True,
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 100,
        "patience": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"📊 Dataset: {dataset_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {config['device']}")
    print(f"⚙️  Config: hidden={config['hidden_sizes']}, lr={config['lr']}, attention={config['attention']}\n")
    
    # Cargar datos
    print("📂 Cargando dataset...")
    df = load_landmarks_csv(str(dataset_path))
    
    # Extraer repeticiones
    print("🔄 Extrayendo repeticiones...")
    reps, rep_meta = reps_from_df(df, min_frames=20)
    print(f"   ✓ {len(reps)} repeticiones extraídas")
    
    # Generar labels
    print("🏷️  Generando labels...")
    labels = labels_multi(rep_meta)
    print(f"   ✓ Labels generados: {labels.shape}")
    print(f"   Distribución: {np.bincount([np.argmax(l) for l in labels])}")
    
    # Split por video
    print("\n📊 Dividiendo dataset por video...")
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    idx_train, idx_val, idx_test = split_by_video(
        rep_meta, 
        video_col='name_video', 
        splits=splits
    )
    
    print(f"   Train: {len(idx_train)} reps")
    print(f"   Val:   {len(idx_val)} reps")
    print(f"   Test:  {len(idx_test)} reps")
    
    # Crear datasets
    train_ds = RepsDataset([reps[i] for i in idx_train], labels[idx_train])
    val_ds = RepsDataset([reps[i] for i in idx_val], labels[idx_val])
    test_ds = RepsDataset([reps[i] for i in idx_test], labels[idx_test])
    
    # Crear dataloaders
    train_dl = make_loader(train_ds, bs=config["batch_size"], shuffle=True)
    val_dl = make_loader(val_ds, bs=config["batch_size"], shuffle=False)
    test_dl = make_loader(test_ds, bs=config["batch_size"], shuffle=False)
    
    # Crear modelo
    print("\n🏗️  Creando modelo BiGRU+Attention...")
    model = BiGRUClassifier(
        in_dim=config["in_dim"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        attention=config["attention"]
    )
    
    device = torch.device(config["device"])
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Parámetros: {total_params:,}")
    
    # Optimizer y loss
    optimizer = Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    # Entrenamiento
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60 + "\n")
    
    best_val_f1 = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": []
    }
    
    for epoch in range(config["epochs"]):
        print(f"\nÉpoca {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_dl, desc="Entrenando")
        for X, Y, M in pbar:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            
            optimizer.zero_grad()
            logits = model(X, M)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).cpu().detach().numpy())
            train_targets.append(Y.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_dl)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_acc = ((train_preds > 0.5) == train_targets).mean()
        
        # Validación
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for X, Y, M in val_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                logits = model(X, M)
                loss = criterion(logits, Y)
                
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(Y.cpu().numpy())
        
        val_loss /= len(val_dl)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_acc = ((val_preds > 0.5) == val_targets).mean()
        
        # Calcular F1 macro
        from sklearn.metrics import f1_score
        train_f1 = f1_score(train_targets, train_preds > 0.5, average='macro', zero_division=0)
        val_f1 = f1_score(val_targets, val_preds > 0.5, average='macro', zero_division=0)
        
        # Guardar historial
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"✓ Mejor modelo guardado (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n⚠️  Early stopping en época {epoch+1}")
                break
    
    # Evaluación final en test
    print("\n" + "="*60)
    print("EVALUACIÓN EN TEST SET")
    print("="*60 + "\n")
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X, Y, M in test_dl:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits = model(X, M)
            test_preds.append(torch.sigmoid(logits).cpu().numpy())
            test_targets.append(Y.cpu().numpy())
    
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # Calcular umbrales óptimos
    print("🎯 Calculando umbrales óptimos...")
    thresholds = compute_optimal_thresholds(val_preds, val_targets)
    print(f"   Umbrales: {thresholds}")
    
    # Métricas con umbrales óptimos
    from sklearn.metrics import classification_report, confusion_matrix
    
    test_pred_binary = (test_preds > thresholds).astype(int)
    test_pred_classes = np.argmax(test_pred_binary, axis=1)
    test_true_classes = np.argmax(test_targets, axis=1)
    
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(
        test_true_classes, 
        test_pred_classes,
        target_names=cfg.CLASS_NAMES,
        digits=4
    ))
    
    print("\n📈 Matriz de Confusión:")
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    print(cm)
    
    # Guardar resultados
    results = {
        "config": config,
        "best_val_f1": float(best_val_f1),
        "test_metrics": {
            "accuracy": float(((test_pred_classes == test_true_classes).sum() / len(test_true_classes))),
            "f1_macro": float(f1_score(test_true_classes, test_pred_classes, average='macro', zero_division=0))
        },
        "thresholds": thresholds.tolist(),
        "class_names": cfg.CLASS_NAMES,
        "total_params": total_params,
        "history": history,
        "confusion_matrix": cm.tolist()
    }
    
    # Guardar metadata
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Guardar archivos adicionales
    np.save(output_dir / "thr_per_class.npy", thresholds)
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(cfg.CLASS_NAMES, f)
    
    print(f"\n✅ Entrenamiento completado")
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"🎯 Mejor F1 (val): {best_val_f1:.4f}")
    print(f"🎯 F1 Macro (test): {results['test_metrics']['f1_macro']:.4f}")
    print(f"🎯 Accuracy (test): {results['test_metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
