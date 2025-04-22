# FileSystem MCP

**FileSystem MCP** es un asistente inteligente de línea de comandos para gestionar operaciones del sistema de archivos mediante lenguaje natural, aprovechando el modelo **Meta Llama‑3 8B Instruct** con capacidades de _function calling_.

## 📝 Descripción

Este proyecto implementa un sistema **Model‑Call‑Process (MCP)** que permite:
- Crear, copiar, mover, renombrar y eliminar archivos y directorios.
- Listar contenidos de carpetas y obtener información detallada de archivos.
- Interactuar de forma conversacional, sin tener que recordar comandos del sistema.
- Ejecutar un modelo Llama‑3 **localmente**, sin depender de servicios en la nube.

## 🚀 Características destacadas

- **Interfaz conversacional** en español.
- **Sistema de permisos** configurable (READ_ONLY, BASIC, STANDARD, ADVANCED, ADMIN).
- **Confirmaciones inteligentes** para evitar operaciones destructivas.
- **UI enriquecida** con [Rich](https://github.com/Textualize/rich) para formatos de texto, tablas y progreso.
- **Function calling** optimizado con Llama‑3 Instruct Q4_K_M (8B parámetros).

## 🖥️ Requisitos del sistema

- **Hardware mínimo**: GPU con **8 GB de VRAM** (recomendado NVIDIA).
- **CPU**: moderado (multinúcleo recomendado).
- **RAM**: 16 GB o más.
- **Espacio en disco**: ~5 GB para el modelo.
- **Sistema operativo**: Windows, macOS o Linux.
- **Python** 3.8 o superior.

## 📦 Instalación

1. Clona o descarga este repositorio:
   ```bash
   git clone https://github.com/Haslick/filesystem-mcp.git
   cd filesystem-mcp
   ```
2. Asegúrate de tener Python 3.8+ instalado.
3. Ejecuta el script; instalará dependencias automáticamente la primera vez:
   ```bash
   python filesystem_mcp.py
   ```
   - Se instalarán `llama-cpp-python` y `rich` si faltan.
   - Se descargará (o solicitará ruta) el modelo de ~5 GB.

## ⚙️ Uso

Al ejecutar el script, se te guiará para configurar:
- Directorio base de operaciones.
- Nivel de permisos.
- Confirmaciones de seguridad.

Después de la configuración inicial, ingresa tus peticiones en lenguaje natural:
```text
Tú: Mueve los archivos informe.pdf y datos.csv a la carpeta Backup
```
Y el asistente procesará la operación.

### 🎛️ Argumentos de línea de comandos

Puedes pasar opciones al invocar `filesystem_mcp.py`:

| Argumento               | Descripción                                          |
|-------------------------|------------------------------------------------------|
| `--base-dir <ruta>`     | Directorio base donde se realizarán las operaciones |
| `--model <ruta>`        | Ruta al archivo GGUF del modelo Llama‑3             |
| `--permission-level`    | Nivel de permisos inicial (READ_ONLY, BASIC, …)      |
| `--confirm-all`         | Forzar confirmación para **todas** las operaciones    |

Ejemplo:
```bash
python filesystem_mcp.py --base-dir ~/MisProyectos --permission-level ADMIN
```

### 🛠️ Comandos especiales

Durante la sesión, usa comandos precedidos por `/`:

| Comando                 | Descripción                                 |
|-------------------------|---------------------------------------------|
| `/ayuda`                | Muestra la ayuda de comandos                |
| `/dir [ruta]`           | Ver o cambiar el directorio base            |
| `/permisos [NIVEL]`     | Consultar o cambiar nivel de permisos       |
| `/herramientas`         | Lista las operaciones permitidas            |
| `/estado`               | Muestra el estado actual (configuración)    |
| `/limpiar`              | Limpia la pantalla                          |
| `/salir`                | Finaliza la sesión                          |

## 📚 Ejemplos de uso

- **Crear directorio**:
  ```text
  Tú: Crea una carpeta llamada Proyectos
  Asistente: ¡Listo! He creado la carpeta 'Proyectos'.
  ```

- **Listar contenido**:
  ```text
  Tú: ¿Qué hay en la carpeta actual?
  Asistente:
    Archivos:
    - informe.txt
    - datos.csv
    Carpetas:
    - Proyectos
  ```

- **Mover varios archivos**:
  ```text
  Tú: Mueve img1.png e img2.png a la carpeta Imágenes
  Asistente: He movido img1.png y img2.png a 'Imágenes'.
  ```

## 💡 Ejecución local del modelo

El modelo **Meta-Llama-3‑8B Instruct** se ejecuta **completamente en tu máquina** mediante `llama-cpp-python`. Esto garantiza:

- **Privacidad total** de tus datos.
- **Baja latencia** (no hay llamadas a la nube).
- **Optimización Q4-K_M** que permite correrlo en **8 GB de VRAM**.

> **Importante**: Ajusta `n_gpu_layers` en `CONFIG["MODEL_PARAMS"]` según tu GPU.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para reportar errores o proponer mejoras:
1. Abre un _issue_ en GitHub.
2. Realiza un _fork_, crea una rama y envía un _pull request_.

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

Desarrollado por **Leonardo** con ayuda de Claude Sonnet 3.7 thinking mode (Abril 2025).

