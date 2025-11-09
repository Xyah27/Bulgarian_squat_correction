# 📤 Instrucciones para Subir el Proyecto a GitHub

## Paso 1: Instalar Git

Si no tienes Git instalado, descárgalo e instálalo:

**Opción A: Descargar desde el sitio oficial**
- Ir a: https://git-scm.com/download/win
- Descargar el instalador para Windows
- Ejecutar el instalador con las opciones por defecto

**Opción B: Instalar con winget (Windows 11/10)**
```powershell
winget install --id Git.Git -e --source winget
```

Después de instalar, **reinicia PowerShell** o tu terminal.

## Paso 2: Configurar Git (Primera vez)

Abre PowerShell y configura tu nombre y email:

```powershell
git config --global user.name "Juan Jose Nuñez"
git config --global user.email "tu_email@ejemplo.com"
```

## Paso 3: Crear un Repositorio en GitHub

1. Ve a https://github.com
2. Inicia sesión (o crea una cuenta si no tienes)
3. Haz clic en el botón **"New"** (o el ícono **+** arriba a la derecha)
4. Completa los datos:
   - **Repository name**: `bulgarian-squat-evaluation` (o el nombre que prefieras)
   - **Description**: "Sistema de evaluación automática de Bulgarian Split Squat con MediaPipe y BiGRU"
   - **Visibility**: Elige "Public" o "Private"
   - ⚠️ **NO marques** "Add a README file" (ya lo tenemos)
   - ⚠️ **NO marques** "Add .gitignore" (ya lo tenemos)
5. Haz clic en **"Create repository"**

GitHub te mostrará instrucciones. **Guarda la URL** que aparece (será algo como: `https://github.com/tu-usuario/bulgarian-squat-evaluation.git`)

## Paso 4: Inicializar el Repositorio Local

Abre PowerShell y navega a la carpeta del proyecto:

```powershell
cd "c:\Users\JUAN JOSE\Desktop\Workspace\Electiva IA\Proyecto VISION BULGARA"
```

Inicializa Git:

```powershell
git init
```

## Paso 5: Añadir los Archivos

Añade todos los archivos al staging area:

```powershell
git add .
```

Verifica qué archivos se añadirán:

```powershell
git status
```

## Paso 6: Hacer el Primer Commit

Crea el primer commit:

```powershell
git commit -m "Initial commit: Bulgarian Squat Evaluation System"
```

## Paso 7: Conectar con GitHub

Conecta tu repositorio local con GitHub (reemplaza la URL con la tuya):

```powershell
git remote add origin https://github.com/TU-USUARIO/bulgarian-squat-evaluation.git
```

Verifica la conexión:

```powershell
git remote -v
```

## Paso 8: Subir el Código a GitHub

Sube los archivos a GitHub:

```powershell
git branch -M main
git push -u origin main
```

Si es la primera vez, te pedirá autenticación:
- **Usuario**: Tu nombre de usuario de GitHub
- **Contraseña**: ⚠️ **NO uses tu contraseña de GitHub**, necesitas un **Personal Access Token**

### Crear un Personal Access Token

1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Haz clic en **"Generate new token"** → **"Generate new token (classic)"**
3. Dale un nombre descriptivo: "Bulgarian Squat Project"
4. Selecciona los permisos:
   - ✅ **repo** (todos los sub-permisos)
5. Haz clic en **"Generate token"**
6. **COPIA EL TOKEN** (no lo volverás a ver)
7. Usa este token como contraseña cuando Git te lo pida

## Paso 9: Verificar en GitHub

1. Ve a tu repositorio en GitHub: `https://github.com/TU-USUARIO/bulgarian-squat-evaluation`
2. Deberías ver todos tus archivos
3. El README.md se mostrará automáticamente en la página principal

## 🔄 Comandos para Actualizaciones Futuras

Cuando hagas cambios y quieras actualizarlos en GitHub:

```powershell
# Ver qué archivos cambiaron
git status

# Añadir archivos específicos
git add archivo1.py archivo2.py

# O añadir todos los cambios
git add .

# Hacer commit con un mensaje descriptivo
git commit -m "Descripción de los cambios"

# Subir a GitHub
git push
```

## 📋 Comandos Útiles

```powershell
# Ver el historial de commits
git log --oneline

# Ver las diferencias antes de hacer commit
git diff

# Deshacer cambios en un archivo (antes de commit)
git checkout -- archivo.py

# Ver ramas
git branch

# Crear una nueva rama
git checkout -b nombre-rama

# Cambiar de rama
git checkout nombre-rama

# Ver el estado actual
git status
```

## 🚨 Solución de Problemas Comunes

### Error: "Git no se reconoce como comando"
**Solución**: Reinstala Git y reinicia PowerShell

### Error: "Permission denied"
**Solución**: 
1. Verifica que estés usando un Personal Access Token, no tu contraseña
2. O configura SSH: https://docs.github.com/es/authentication/connecting-to-github-with-ssh

### Error: "Repository already exists"
**Solución**: 
```powershell
git remote remove origin
git remote add origin https://github.com/TU-USUARIO/nuevo-repositorio.git
```

### Archivos muy grandes
Si tienes archivos muy grandes (>100MB):
1. Añádelos al `.gitignore`
2. O usa Git LFS: https://git-lfs.github.com/

### Error: "Authentication failed"
**Solución**:
1. Verifica tu token de acceso
2. O usa GitHub CLI: `gh auth login`

## 📚 Recursos Adicionales

- [Documentación oficial de Git](https://git-scm.com/doc)
- [GitHub Docs](https://docs.github.com)
- [Tutorial interactivo de Git](https://learngitbranching.js.org/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

## ✅ Checklist Final

- [ ] Git instalado
- [ ] Git configurado (nombre y email)
- [ ] Repositorio creado en GitHub
- [ ] Repositorio local inicializado
- [ ] Archivos añadidos con `git add`
- [ ] Primer commit realizado
- [ ] Remote configurado
- [ ] Código subido a GitHub
- [ ] Verificado en github.com

---

**¡Listo! Tu proyecto ya está en GitHub 🎉**

Si tienes problemas, revisa la sección de "Solución de Problemas" o consulta la documentación oficial.
