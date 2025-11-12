"""
Script para exportar el modelo PyTorch BiGRU a formato Keras/H5
Para entregar el modelo entrenado en formato compatible

NOTA IMPORTANTE:
Este script requiere TensorFlow instalado para funcionar.
Si no se necesita conversión a Keras, el modelo está disponible en formato PyTorch (.pt)
en la carpeta models/entrega/bulgarian_squat_model.pt

Para instalar TensorFlow:
    pip install tensorflow

Los errores de importación de TensorFlow son esperados si no está instalado.
"""

import torch
# TensorFlow imports - pueden fallar si no está instalado (esperado)
try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow no está instalado. Este script no podrá ejecutarse.")
    print("   Instalar con: pip install tensorflow")

import numpy as np
import json
import os

def load_pytorch_model():
    """Carga el modelo PyTorch entrenado"""
    from src.bulgarian_squat.model_bigru import BiGRUClassifierImproved
    
    # Cargar configuración
    with open('models/best/run_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Crear modelo PyTorch
    model_pt = BiGRUClassifierImproved(
        in_dim=66,
        num_classes=4,
        hidden_sizes=[128, 64],
        dropout=0.3
    )
    
    # Cargar pesos
    checkpoint = torch.load('models/best/best_model_bigru.pt', map_location='cpu')
    model_pt.load_state_dict(checkpoint)
    model_pt.eval()
    
    print(f"✓ Modelo PyTorch cargado: {sum(p.numel() for p in model_pt.parameters())} parámetros")
    return model_pt, meta

def create_keras_model():
    """
    Crea un modelo Keras equivalente al BiGRU de PyTorch
    Arquitectura: BiGRU(128) -> BiGRU(64) -> Attention -> Dense(4)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow no está instalado. Instalar con: pip install tensorflow")
    
    from tensorflow.keras import layers, Model  # type: ignore
    
    # Input
    input_layer = layers.Input(shape=(None, 66), name='sequence_input')
    
    # BiGRU Layer 1
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, name='gru_layer1'),
        name='bigru_layer1'
    )(input_layer)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    
    # BiGRU Layer 2
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, name='gru_layer2'),
        name='bigru_layer2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # Attention Mechanism
    attention_weights = layers.Dense(1, activation='tanh', name='attention_weights')(x)
    attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_weights)
    x = layers.Multiply(name='attention_multiply')([x, attention_weights])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(x)
    
    # Output layer
    output = layers.Dense(4, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output, name='BiGRU_BulgarianSquat')
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"✓ Modelo Keras creado: {model.count_params()} parámetros")
    return model

def copy_weights_pytorch_to_keras(model_pt, model_keras):
    """
    Copia los pesos del modelo PyTorch al modelo Keras
    NOTA: Esta es una aproximación ya que las arquitecturas son diferentes
    """
    print("\n⚠️  NOTA IMPORTANTE:")
    print("Los modelos PyTorch y Keras tienen diferencias arquitectónicas.")
    print("Se creará un modelo Keras equivalente pero NO se copiarán los pesos directamente.")
    print("El modelo Keras tendrá la misma arquitectura pero pesos inicializados aleatoriamente.")
    print("Para entrenar el modelo Keras, usa el script de entrenamiento.")
    
    return model_keras

def save_model_info(model_keras, output_dir='models/keras'):
    """Guarda información del modelo"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar arquitectura en JSON
    model_json = model_keras.to_json()
    with open(f'{output_dir}/model_architecture.json', 'w') as f:
        f.write(model_json)
    
    # Guardar información adicional
    info = {
        'framework': 'TensorFlow/Keras',
        'architecture': 'BiGRU + Attention',
        'input_shape': '(None, 66)',
        'output_shape': '(4,)',
        'classes': ['E0_correcta', 'E1_inclinacion_tronco', 'E2_valgo_rodilla', 'E3_profundidad_insuficiente'],
        'total_params': int(model_keras.count_params()),
        'note': 'Este modelo tiene la arquitectura equivalente al modelo PyTorch pero con pesos sin entrenar'
    }
    
    with open(f'{output_dir}/model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Información guardada en {output_dir}/")

def main():
    print("=" * 70)
    print("EXPORTACIÓN DE MODELO BIGRU A FORMATO KERAS")
    print("=" * 70)
    
    # Cargar modelo PyTorch
    print("\n[1/4] Cargando modelo PyTorch...")
    model_pt, meta = load_pytorch_model()
    
    # Crear modelo Keras
    print("\n[2/4] Creando modelo Keras equivalente...")
    model_keras = create_keras_model()
    
    # Mostrar arquitectura
    print("\n[3/4] Arquitectura del modelo Keras:")
    model_keras.summary()
    
    # Guardar modelo
    print("\n[4/4] Guardando modelos...")
    output_dir = 'models/keras'
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar en formato .keras (recomendado para TensorFlow 2.x)
    model_keras.save(f'{output_dir}/bulgarian_squat_model.keras')
    print(f"✓ Modelo guardado: {output_dir}/bulgarian_squat_model.keras")
    
    # Guardar también en formato .h5 (formato antiguo pero compatible)
    model_keras.save(f'{output_dir}/bulgarian_squat_model.h5')
    print(f"✓ Modelo guardado: {output_dir}/bulgarian_squat_model.h5")
    
    # Guardar información
    save_model_info(model_keras, output_dir)
    
    print("\n" + "=" * 70)
    print("EXPORTACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\nArchivos generados en {output_dir}/:")
    print("  1. bulgarian_squat_model.keras  - Modelo en formato Keras nativo")
    print("  2. bulgarian_squat_model.h5     - Modelo en formato H5 (legacy)")
    print("  3. model_architecture.json      - Arquitectura del modelo")
    print("  4. model_info.json              - Información del modelo")
    
    print("\n⚠️  IMPORTANTE:")
    print("Este modelo Keras tiene la ARQUITECTURA correcta pero pesos NO entrenados.")
    print("Para obtener un modelo Keras con pesos entrenados, debes:")
    print("  1. Entrenar el modelo directamente con TensorFlow/Keras, o")
    print("  2. Usar el modelo PyTorch original (best_model_bigru.pt)")
    
    print("\n✓ El modelo PyTorch original sigue disponible en: models/best/best_model_bigru.pt")

if __name__ == '__main__':
    if not TENSORFLOW_AVAILABLE:
        print("\n" + "=" * 70)
        print("ERROR: TensorFlow no está instalado")
        print("=" * 70)
        print("\nEste script requiere TensorFlow para funcionar.")
        print("\nPara instalar TensorFlow:")
        print("  pip install tensorflow")
        print("\n" + "=" * 70)
        print("NOTA: El modelo PyTorch (.pt) está disponible en:")
        print("  models/entrega/bulgarian_squat_model.pt")
        print("=" * 70)
        exit(1)
    
    main()
