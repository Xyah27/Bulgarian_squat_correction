# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto Bulgarian Split Squat Posture Analysis! 🎉

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Reportar Bugs](#reportar-bugs)
- [Solicitar Features](#solicitar-features)
- [Pull Requests](#pull-requests)
- [Estilo de Código](#estilo-de-código)
- [Testing](#testing)

## Código de Conducta

Este proyecto se adhiere a un código de conducta. Al participar, se espera que mantengas un ambiente respetuoso y profesional.

## Cómo Contribuir

### 1. Fork el Repositorio

```bash
git clone https://github.com/tu-usuario/bulgarian-split-squat.git
cd bulgarian-split-squat
```

### 2. Crear una Rama

```bash
git checkout -b feature/mi-nueva-feature
# o
git checkout -b fix/mi-bug-fix
```

### 3. Instalar Dependencias de Desarrollo

```bash
pip install -e ".[dev]"
```

### 4. Hacer Cambios

- Asegúrate de que tu código siga las convenciones de estilo
- Añade tests para nuevas funcionalidades
- Actualiza la documentación según sea necesario

### 5. Ejecutar Tests

```bash
pytest tests/
```

### 6. Commit y Push

```bash
git add .
git commit -m "feat: añadir nueva funcionalidad X"
git push origin feature/mi-nueva-feature
```

### 7. Crear Pull Request

Ve a GitHub y crea un Pull Request desde tu rama.

## Reportar Bugs

Para reportar un bug, abre un [Issue](https://github.com/tu-usuario/bulgarian-split-squat/issues) e incluye:

- **Descripción clara** del problema
- **Pasos para reproducir** el error
- **Comportamiento esperado** vs comportamiento actual
- **Versión** de Python, PyTorch, y otras dependencias
- **Screenshots** si es relevante
- **Logs** o mensajes de error

### Template de Bug Report

```markdown
**Descripción**
Breve descripción del bug

**Pasos para Reproducir**
1. Ejecutar comando X
2. Ver error Y

**Comportamiento Esperado**
Lo que debería suceder

**Comportamiento Actual**
Lo que está sucediendo

**Entorno**
- OS: [Windows/Linux/Mac]
- Python: [versión]
- PyTorch: [versión]
- CUDA: [sí/no]

**Logs**
```
Pega aquí los logs relevantes
```
```

## Solicitar Features

Para solicitar una nueva funcionalidad:

1. Abre un [Issue](https://github.com/tu-usuario/bulgarian-split-squat/issues)
2. Usa la etiqueta `enhancement`
3. Describe claramente:
   - **Problema** que resuelve
   - **Solución propuesta**
   - **Alternativas** consideradas
   - **Casos de uso**

## Pull Requests

### Checklist antes de enviar

- [ ] El código sigue las convenciones de estilo del proyecto
- [ ] He añadido tests que prueban mi código
- [ ] Todos los tests pasan localmente
- [ ] He actualizado la documentación
- [ ] Mi commit sigue el formato convencional
- [ ] He añadido mi nombre a los contribuidores (si es tu primera contribución)

### Formato de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
tipo(alcance): descripción breve

[cuerpo opcional]

[footer opcional]
```

**Tipos:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formato, punto y coma, etc (sin cambios en código)
- `refactor`: Refactorización de código
- `test`: Añadir o corregir tests
- `chore`: Tareas de mantenimiento

**Ejemplos:**
```bash
feat(model): añadir arquitectura Transformer
fix(inference): corregir detección de vista frontal
docs(readme): actualizar instrucciones de instalación
```

## Estilo de Código

### Python

Seguimos [PEP 8](https://pep8.org/) con algunas excepciones:

- **Longitud de línea**: 100 caracteres (no 79)
- **Comillas**: Preferir dobles `"` sobre simples `'`
- **Type hints**: Usar en funciones públicas

**Herramientas:**
```bash
# Formatear código
black src/ scripts/

# Linting
flake8 src/ scripts/

# Type checking
mypy src/
```

### Docstrings

Usar formato Google:

```python
def mi_funcion(param1: int, param2: str) -> bool:
    """
    Breve descripción de la función.
    
    Descripción más detallada si es necesario.
    
    Args:
        param1: Descripción del parámetro 1
        param2: Descripción del parámetro 2
    
    Returns:
        Descripción del valor de retorno
    
    Raises:
        ValueError: Cuando param1 es negativo
    
    Example:
        >>> mi_funcion(5, "test")
        True
    """
    # implementación
```

## Testing

### Estructura de Tests

```
tests/
├── test_models.py
├── test_datamodule.py
├── test_inference.py
└── test_utils.py
```

### Escribir Tests

```python
import pytest
from bulgarian_squat import BiGRUClassifierImproved

def test_model_forward():
    """Test que el forward pass funciona correctamente"""
    model = BiGRUClassifierImproved(in_dim=66, num_classes=4)
    x = torch.randn(2, 30, 66)  # batch=2, seq_len=30
    mask = torch.ones(2, 30)
    
    output = model(x, mask)
    
    assert output.shape == (2, 4)
    assert not torch.isnan(output).any()
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Con coverage
pytest --cov=bulgarian_squat --cov-report=html

# Tests específicos
pytest tests/test_models.py

# Tests con marcadores
pytest -m "not slow"
```

## Áreas que Necesitan Contribuciones

- 🐛 **Bug fixes**: Revisar [issues abiertos](https://github.com/tu-usuario/bulgarian-split-squat/issues)
- 📚 **Documentación**: Mejorar README, docstrings, tutoriales
- 🧪 **Tests**: Aumentar cobertura de tests
- 🎨 **Visualizaciones**: Mejorar gráficos y análisis
- 🚀 **Optimización**: Mejorar rendimiento de inferencia
- 🌐 **Internacionalización**: Traducir documentación
- 📊 **Datasets**: Contribuir con nuevos datos de entrenamiento

## Preguntas

Si tienes preguntas, puedes:

1. Revisar [Issues cerrados](https://github.com/tu-usuario/bulgarian-split-squat/issues?q=is%3Aissue+is%3Aclosed)
2. Abrir un [nuevo Issue](https://github.com/tu-usuario/bulgarian-split-squat/issues/new)
3. Contactar por email: tu.email@example.com

---

¡Gracias por contribuir! 🙌
