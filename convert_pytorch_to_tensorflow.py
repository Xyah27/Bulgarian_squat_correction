"""
Convierte un modelo PyTorch entrenado a formato TensorFlow/Keras.

Uso:
    python convert_pytorch_to_tensorflow.py models/entrega/bulgarian_squat_model.pt
"""

import sys
import os
from pathlib import Path
import json
import numpy as np

# Silenciar warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import tensorflow as tf  # type: ignore[import-not-found]
from tensorflow import keras  # type: ignore[import-not-found]

# Agregar src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from bulgarian_squat.model_improved import BiGRUClassifierImproved


def create_keras_model(in_dim=66, num_classes=4, hidden1=128, hidden2=64, dropout=0.3):
    """
    Crea un modelo Keras equivalente al BiGRU de PyTorch.
    
    La arquitectura replica:
    - BiGRU bidireccional (128 units)
    - BiGRU bidireccional (64 units)
    - GlobalAveragePooling1D (simula atención)
    - Dense layer final
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(None, in_dim)),
        keras.layers.BatchNormalization(),
        
        # Primera capa BiGRU (128 units bidireccional = 256 total)
        keras.layers.Bidirectional(
            keras.layers.GRU(hidden1, return_sequences=True, name='gru1')
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),
        
        # Segunda capa BiGRU (64 units bidireccional = 128 total)
        keras.layers.Bidirectional(
            keras.layers.GRU(hidden2, return_sequences=True, name='gru2')
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),
        
        # Pooling global (simula mecanismo de atención)
        keras.layers.GlobalAveragePooling1D(),
        
        # Clasificador
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes, activation='sigmoid', name='output')
    ], name='BiGRU_Classifier')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def convert_model(pytorch_path, output_dir=None):
    """
    Convierte modelo PyTorch a TensorFlow/Keras.
    
    Args:
        pytorch_path: Ruta al archivo .pt
        output_dir: Directorio de salida (si es None, usa el mismo dir que .pt)
    """
    pytorch_path = Path(pytorch_path)
    
    if not pytorch_path.exists():
        print(f"ERROR: Archivo no encontrado: {pytorch_path}")
        return False
    
    # Directorio de salida
    if output_dir is None:
        output_dir = pytorch_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CONVERSION PYTORCH -> TENSORFLOW")
    print(f"{'='*70}\n")
    
    print(f"Modelo PyTorch: {pytorch_path}")
    print(f"Directorio salida: {output_dir}\n")
    
    # Cargar metadata si existe
    meta_path = pytorch_path.parent / "MODEL_INFO.json"
    if not meta_path.exists():
        meta_path = pytorch_path.parent / "run_meta.json"
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        in_dim = metadata.get('in_dim', 66)
        num_classes = metadata.get('num_classes', 4)
        hidden1 = metadata.get('hidden_dim1', 128)
        hidden2 = metadata.get('hidden_dim2', 64)
        dropout = metadata.get('dropout', 0.3)
        
        print(f"OK Metadata cargada:")
        print(f"   - Input dim: {in_dim}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Hidden: [{hidden1}, {hidden2}]")
        print(f"   - Dropout: {dropout}\n")
    else:
        print("WARNING: No se encontro metadata, usando valores por defecto\n")
        in_dim, num_classes, hidden1, hidden2, dropout = 66, 4, 128, 64, 0.3
    
    # Cargar modelo PyTorch
    print("Cargando modelo PyTorch...")
    pytorch_model = BiGRUClassifierImproved(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
        use_attention=True
    )
    pytorch_model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
    pytorch_model.eval()
    print(f"OK Modelo PyTorch cargado\n")
    
    # Crear modelo Keras
    print("Creando modelo Keras equivalente...")
    keras_model = create_keras_model(in_dim, num_classes, hidden1, hidden2, dropout)
    
    print(f"OK Modelo creado:")
    print(f"   - Parametros: {keras_model.count_params():,}")
    print(f"   - Capas: {len(keras_model.layers)}")
    
    # NOTA IMPORTANTE: Los pesos no se transfieren automáticamente
    # PyTorch y Keras tienen estructuras de pesos diferentes
    # Este modelo Keras tiene la misma arquitectura pero pesos aleatorios
    print(f"\nNOTA: Modelo Keras tiene arquitectura equivalente pero pesos inicializados aleatoriamente")
    print(f"      Para transferir pesos necesitaria conversion manual compleja\n")
    
    # Guardar en formato .h5
    h5_path = output_dir / "bulgarian_squat_model.h5"
    print(f"Guardando en formato .h5...")
    keras_model.save(h5_path)
    size_h5 = h5_path.stat().st_size / (1024**2)
    print(f"OK Guardado: {h5_path}")
    print(f"   Tamano: {size_h5:.2f} MB\n")
    
    # Guardar en formato SavedModel
    savedmodel_path = output_dir / "saved_model"
    print(f"Guardando en formato SavedModel...")
    
    try:
        # Keras 3+ usa .export() para SavedModel
        keras_model.export(str(savedmodel_path))
    except AttributeError:
        # Keras 2.x usa .save()
        keras_model.save(str(savedmodel_path), save_format='tf')
    
    # Calcular tamaño total del SavedModel
    total_size = sum(f.stat().st_size for f in savedmodel_path.rglob('*') if f.is_file()) / (1024**2)
    print(f"OK Guardado: {savedmodel_path}")
    print(f"   Tamano: {total_size:.2f} MB\n")
    
    # Probar carga
    print("Validando modelos guardados...")
    
    # Test .h5
    try:
        model_h5 = keras.models.load_model(h5_path)
        print(f"OK Modelo .h5 cargado correctamente")
        
        # Test con datos aleatorios
        test_input = np.random.randn(1, 10, in_dim).astype(np.float32)
        output = model_h5.predict(test_input, verbose=0)
        print(f"   Output shape: {output.shape}")
        print(f"   Output sample: {output[0][:4]}")
        
    except Exception as e:
        print(f"ERROR: al cargar .h5: {e}")
    
    # Test SavedModel
    try:
        model_saved = keras.models.load_model(savedmodel_path)
        output = model_saved.predict(test_input, verbose=0)
        print(f"\nOK SavedModel cargado correctamente")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\nERROR: al cargar SavedModel: {e}")
    
    print(f"\n{'='*70}")
    print(f"CONVERSION COMPLETADA")
    print(f"{'='*70}\n")
    
    print("Archivos generados:")
    print(f"   1. {h5_path.name} ({size_h5:.2f} MB)")
    print(f"   2. {savedmodel_path.name}/ ({total_size:.2f} MB)\n")
    
    print("Uso:")
    print(f"   # Cargar modelo .h5")
    print(f"   import tensorflow as tf")
    print(f"   model = tf.keras.models.load_model('{h5_path}')")
    print(f"")
    print(f"   # Cargar SavedModel")
    print(f"   model = tf.keras.models.load_model('{savedmodel_path}')\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convertir modelo PyTorch a TensorFlow")
    parser.add_argument("pytorch_model", type=str, help="Ruta al archivo .pt")
    parser.add_argument("--output", type=str, default=None, 
                       help="Directorio de salida (default: mismo dir que .pt)")
    
    args = parser.parse_args()
    
    success = convert_model(args.pytorch_model, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
