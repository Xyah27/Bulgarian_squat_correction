"""
Preparar artifacts del modelo para inferencia
==============================================

Copia el modelo entrenado y genera los archivos necesarios para inferencia.

Uso:
    python scripts/utils/prepare_artifacts.py --model models/checkpoints/bigru_20241106/best_model.pt --output models/best
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import json
import shutil
import argparse

from src.bulgarian_squat.model_improved import BiGRUClassifierImproved


def main():
    parser = argparse.ArgumentParser(description="Preparar artifacts del modelo")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado (.pt)")
    parser.add_argument("--output", type=str, default="models/best", help="Directorio de salida")
    parser.add_argument("--meta", type=str, help="Ruta al run_meta.json (opcional, se busca automáticamente)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Preparar Artifacts del Modelo ===\n")
    print(f"📂 Modelo: {model_path}")
    print(f"📁 Salida: {output_dir}\n")
    
    # Buscar run_meta.json
    if args.meta:
        meta_path = Path(args.meta)
    else:
        meta_path = model_path.parent / "run_meta.json"
    
    if not meta_path.exists():
        print(f"⚠️  No se encontró run_meta.json en {meta_path}")
        print("   Usando configuración por defecto...")
        
        meta = {
            "model_type": "BiGRU+Attention",
            "in_dim": 66,
            "num_classes": 4,
            "hidden_dim1": 128,
            "hidden_dim2": 64,
            "dropout": 0.3,
            "use_attention": True,
            "use_batch_norm": True,
            "use_z": False
        }
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"✓ Metadatos cargados desde {meta_path}")
    
    # Copiar modelo
    shutil.copy(model_path, output_dir / "best_model_bigru.pt")
    print(f"✓ Modelo copiado: best_model_bigru.pt")
    
    # Guardar metadatos
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadatos guardados: run_meta.json")
    
    # Buscar o crear class_names.json
    class_names_path = model_path.parent / "class_names.json"
    if class_names_path.exists():
        shutil.copy(class_names_path, output_dir / "class_names.json")
        print(f"✓ Nombres de clases copiados: class_names.json")
    else:
        class_names = {
            "0": "correcta",
            "1": "E1_tronco",
            "2": "E2_valgo",
            "3": "E3_profundidad"
        }
        with open(output_dir / "class_names.json", "w") as f:
            json.dump(class_names, f, indent=2)
        print(f"✓ Nombres de clases creados (default): class_names.json")
    
    # Buscar o crear thr_per_class.npy
    thr_path = model_path.parent / "thr_per_class.npy"
    if thr_path.exists():
        shutil.copy(thr_path, output_dir / "thr_per_class.npy")
        thresholds = np.load(thr_path)
        print(f"✓ Umbrales copiados: thr_per_class.npy")
        print(f"  Valores: {thresholds}")
    else:
        # Umbrales por defecto
        thresholds = np.array([0.31, 0.19, 0.10, 0.70])
        np.save(output_dir / "thr_per_class.npy", thresholds)
        print(f"✓ Umbrales creados (default): thr_per_class.npy")
        print(f"  Valores: {thresholds}")
    
    print(f"\n✅ Artifacts preparados en: {output_dir}")
    print("\nArchivos generados:")
    print("  - best_model_bigru.pt")
    print("  - run_meta.json")
    print("  - class_names.json")
    print("  - thr_per_class.npy")
    
    print("\n📋 Para usar en inferencia:")
    print(f"  python scripts/inference/run_webcam.py --model {output_dir}\n")


if __name__ == "__main__":
    main()
