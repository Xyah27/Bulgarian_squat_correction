"""
Script de entrenamiento del modelo BiGRU+Attention
==================================================

Entrena el modelo con el dataset de Bulgarian Split Squat.

Uso:
    python scripts/training/train_bigru.py --config configs/bigru_config.yaml
    
Opciones:
    --config: Ruta al archivo de configuración
    --epochs: Número de épocas (default: 100)
    --batch_size: Tamaño de batch (default: 32)
    --lr: Learning rate (default: 0.001)
    --patience: Early stopping patience (default: 15)
"""

import sys
from pathlib import Path

# Añadir src al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from datetime import datetime

from bulgarian_squat.model_improved import BiGRUClassifierImproved
from bulgarian_squat.datamodule import BulgarianSquatDataModule
from bulgarian_squat.train import train_one_epoch
from bulgarian_squat.eval import evaluate_model, compute_optimal_thresholds
from bulgarian_squat import config as cfg


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo BiGRU+Attention")
    parser.add_argument("--dataset", type=str, 
                       default="data/raw/landmarks_dataset_BALANCEADO_v2.csv",
                       help="Ruta al dataset CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden size capa 1")
    parser.add_argument("--hidden2", type=int, default=64, help="Hidden size capa 2")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--use_attention", action="store_true", default=True, 
                       help="Usar mecanismo de atención")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
                       help="Directorio para guardar modelos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Configurar semilla
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"🚀 Entrenamiento Modelo BiGRU+Attention")
    print(f"{'='*60}\n")
    print(f"📊 Dataset: {args.dataset}")
    print(f"💻 Device: {device}")
    print(f"🔧 Configuración:")
    print(f"   - Épocas: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Hidden sizes: {args.hidden1}, {args.hidden2}")
    print(f"   - Dropout: {args.dropout}")
    print(f"   - Atención: {args.use_attention}")
    print(f"   - Patience: {args.patience}\n")
    
    # Cargar datos
    print("📂 Cargando datos...")
    dm = BulgarianSquatDataModule(
        csv_path=args.dataset,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        seed=args.seed
    )
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    in_dim = dm.num_features
    num_classes = dm.num_classes
    
    print(f"✓ Datos cargados:")
    print(f"   - Train: {len(dm.train_dataset)} samples")
    print(f"   - Val: {len(dm.val_dataset)} samples")
    print(f"   - Test: {len(dm.test_dataset)} samples")
    print(f"   - Features: {in_dim}")
    print(f"   - Classes: {num_classes}\n")
    
    # Crear modelo
    print("🏗️  Creando modelo...")
    model = BiGRUClassifierImproved(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        use_batch_norm=True,
        use_attention=args.use_attention
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Modelo creado:")
    print(f"   - Parámetros totales: {total_params:,}")
    print(f"   - Parámetros entrenables: {trainable_params:,}\n")
    
    # Optimizador y criterio
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bigru_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"🎯 Iniciando entrenamiento...\n")
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["macro_f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"  ✓ Mejor modelo guardado (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping en época {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"✅ Entrenamiento completado")
    print(f"{'='*60}\n")
    
    # Cargar mejor modelo para evaluación final
    model.load_state_dict(torch.load(run_dir / "best_model.pt"))
    
    # Evaluación en test
    print("📊 Evaluación en conjunto de test...")
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device, verbose=True)
    
    # Calcular umbrales óptimos
    print("\n🎯 Calculando umbrales óptimos...")
    optimal_thresholds = compute_optimal_thresholds(model, val_loader, device)
    print(f"Umbrales óptimos: {optimal_thresholds}")
    
    # Guardar metadatos
    metadata = {
        "model_type": "BiGRU+Attention",
        "timestamp": timestamp,
        "in_dim": in_dim,
        "num_classes": num_classes,
        "hidden_dim1": args.hidden1,
        "hidden_dim2": args.hidden2,
        "dropout": args.dropout,
        "use_attention": args.use_attention,
        "use_batch_norm": True,
        "use_z": False,
        "best_val_f1": float(best_val_f1),
        "test_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in test_metrics.items()},
        "optimal_thresholds": optimal_thresholds.tolist(),
        "training_config": vars(args),
        "history": history
    }
    
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Guardar umbrales
    np.save(run_dir / "thr_per_class.npy", optimal_thresholds)
    
    # Guardar nombres de clases
    class_names = {str(i): name for i, name in enumerate(dm.class_names)}
    with open(run_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\n📁 Archivos guardados en: {run_dir}")
    print(f"   - best_model.pt")
    print(f"   - run_meta.json")
    print(f"   - thr_per_class.npy")
    print(f"   - class_names.json")
    
    print(f"\n🎉 ¡Entrenamiento exitoso!")
    print(f"   Mejor F1 (val): {best_val_f1:.4f}")
    print(f"   F1 (test): {test_metrics['macro_f1']:.4f}")
    print(f"   Accuracy (test): {test_metrics['accuracy']:.4f}\n")


if __name__ == "__main__":
    main()
