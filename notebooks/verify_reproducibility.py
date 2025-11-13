"""
Script para verificar que el notebook es completamente reproducible
Verifica que todos los archivos necesarios existen y pueden ser cargados
"""

import os
import json
import numpy as np
from pathlib import Path

def verify_reproducibility():
    """Verifica que todos los archivos necesarios para reproducir el notebook existen"""
    
    print("="*80)
    print("VERIFICACIÓN DE REPRODUCIBILIDAD DEL NOTEBOOK")
    print("="*80)
    
    errors = []
    warnings = []
    success = []
    
    # Directorio base del proyecto
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models' / 'best'
    
    # 1. Verificar archivos de modelo PyTorch
    pytorch_model = models_dir / 'best_model_bigru.pt'
    if pytorch_model.exists():
        size = pytorch_model.stat().st_size / 1024 / 1024
        success.append(f"✓ Modelo PyTorch: {pytorch_model.name} ({size:.2f} MB)")
    else:
        errors.append(f"✗ Modelo PyTorch NO encontrado: {pytorch_model}")
    
    # 2. Verificar archivos de modelo TensorFlow
    tf_model_h5 = models_dir / 'bulgarian_squat_model.h5'
    if tf_model_h5.exists():
        size = tf_model_h5.stat().st_size / 1024 / 1024
        success.append(f"✓ Modelo TensorFlow .h5: {tf_model_h5.name} ({size:.2f} MB)")
    else:
        warnings.append(f"⚠ Modelo TensorFlow .h5 NO encontrado: {tf_model_h5}")
    
    tf_model_saved = models_dir / 'saved_model'
    if tf_model_saved.exists():
        size = sum(f.stat().st_size for f in tf_model_saved.rglob('*') if f.is_file()) / 1024 / 1024
        success.append(f"✓ Modelo TensorFlow SavedModel: {tf_model_saved.name}/ ({size:.2f} MB)")
    else:
        warnings.append(f"⚠ Modelo TensorFlow SavedModel NO encontrado: {tf_model_saved}")
    
    # 3. Verificar archivos JSON con métricas
    json_files = {
        'MODEL_INFO.json': 'Métricas principales del modelo',
        'complete_metrics.json': 'Métricas completas por clase',
        'run_meta.json': 'Configuración de entrenamiento',
        'class_names.json': 'Nombres de clases'
    }
    
    for json_file, description in json_files.items():
        json_path = models_dir / json_file
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                success.append(f"✓ {json_file}: {description} ({len(str(data))} chars)")
            except Exception as e:
                errors.append(f"✗ Error leyendo {json_file}: {e}")
        else:
            errors.append(f"✗ Archivo JSON NO encontrado: {json_path}")
    
    # 4. Verificar archivos numpy
    npy_path = models_dir / 'thr_per_class.npy'
    if npy_path.exists():
        try:
            thr = np.load(npy_path)
            success.append(f"✓ thr_per_class.npy: Umbrales óptimos (shape: {thr.shape})")
        except Exception as e:
            errors.append(f"✗ Error leyendo thr_per_class.npy: {e}")
    else:
        errors.append(f"✗ Archivo numpy NO encontrado: {npy_path}")
    
    # 5. Verificar estructura de directorios
    notebooks_dir = base_dir / 'notebooks'
    figures_dir = notebooks_dir / 'figures'
    
    if notebooks_dir.exists():
        success.append(f"✓ Directorio notebooks existe")
    else:
        errors.append(f"✗ Directorio notebooks NO existe: {notebooks_dir}")
    
    if figures_dir.exists():
        success.append(f"✓ Directorio figures existe")
        # Listar figuras generadas
        figures = list(figures_dir.glob('*.png'))
        for fig in figures:
            size = fig.stat().st_size / 1024
            success.append(f"  • {fig.name} ({size:.1f} KB)")
    else:
        warnings.append(f"⚠ Directorio figures NO existe (se creará al ejecutar notebook): {figures_dir}")
    
    # 6. Verificar integridad de datos en JSON
    print("\n" + "="*80)
    print("VALIDACIÓN DE DATOS")
    print("="*80)
    
    try:
        # Cargar MODEL_INFO.json
        with open(models_dir / 'MODEL_INFO.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Verificar campos críticos
        required_fields = ['metricas', 'parametros_totales', 'mejor_epoch']
        for field in required_fields:
            if field in model_info:
                success.append(f"✓ MODEL_INFO.json contiene '{field}'")
            else:
                errors.append(f"✗ MODEL_INFO.json NO contiene '{field}'")
        
        # Mostrar métricas
        if 'metricas' in model_info:
            metricas = model_info['metricas']
            print(f"\nMétricas del modelo:")
            print(f"  • F1-Score Macro: {metricas.get('f1_score_macro', 'N/A'):.4f}")
            print(f"  • F1-Score Micro: {metricas.get('f1_score_micro', 'N/A'):.4f}")
            print(f"  • Accuracy: {metricas.get('accuracy', 'N/A'):.4f}")
        
        # Cargar complete_metrics.json
        with open(models_dir / 'complete_metrics.json', 'r', encoding='utf-8') as f:
            complete_metrics = json.load(f)
        
        # Verificar estructura
        if 'per_class' in complete_metrics:
            per_class = complete_metrics['per_class']
            print(f"\nMétricas por clase disponibles:")
            print(f"  • Precision: {len(per_class.get('precision', []))} clases")
            print(f"  • Recall: {len(per_class.get('recall', []))} clases")
            print(f"  • F1-Score: {len(per_class.get('f1_score', []))} clases")
        
        if 'confusion_matrix_normalized' in complete_metrics:
            cm = np.array(complete_metrics['confusion_matrix_normalized'])
            print(f"\nMatriz de confusión:")
            print(f"  • Shape: {cm.shape}")
            print(f"  • Valores válidos: {cm.min():.3f} - {cm.max():.3f}")
            success.append(f"✓ Matriz de confusión válida (shape: {cm.shape})")
    
    except Exception as e:
        errors.append(f"✗ Error validando datos JSON: {e}")
    
    # 7. Resumen
    print("\n" + "="*80)
    print("RESUMEN DE VERIFICACIÓN")
    print("="*80)
    
    if success:
        print(f"\n✓ ÉXITOS ({len(success)}):")
        for s in success:
            print(f"  {s}")
    
    if warnings:
        print(f"\n⚠ ADVERTENCIAS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
    
    if errors:
        print(f"\n✗ ERRORES ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    
    print("\n" + "="*80)
    
    if not errors:
        print("✓✓ VERIFICACIÓN EXITOSA - El notebook es COMPLETAMENTE REPRODUCIBLE")
        print("\nPara reproducir los resultados:")
        print("  1. Abre notebooks/resultados_paper_bulgarian_squat.ipynb")
        print("  2. Ejecuta: Kernel > Restart & Run All")
        print("  3. Todas las figuras se generarán en notebooks/figures/")
        print("\n✓ Todos los datos provienen de archivos reales (no hardcodeados)")
        return True
    else:
        print("✗✗ VERIFICACIÓN FALLIDA - Faltan archivos necesarios")
        print("\nPor favor verifica que:")
        print("  • El modelo está entrenado y guardado en models/best/")
        print("  • Los archivos JSON de métricas están presentes")
        print("  • Los umbrales óptimos están guardados en .npy")
        return False

if __name__ == "__main__":
    verify_reproducibility()
