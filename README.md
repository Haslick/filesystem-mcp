# FileSystem MCP

Un asistente inteligente para operaciones del sistema de archivos utilizando el modelo Llama-3 con capacidades de function calling optimizadas.

## Descripción

Este script implementa un sistema Model-Call-Process (MCP) que permite realizar operaciones en el sistema de archivos a través de lenguaje natural, aprovechando las capacidades específicas de function calling del modelo Llama-3.

## Características

- 💬 **Interfaz conversacional** - Interactúa con tus archivos usando lenguaje natural
- 🔒 **Sistema de permisos** - Diferentes niveles de acceso para operaciones seguras
- 📁 **Operaciones completas** - Crear, copiar, mover, renombrar y eliminar archivos/directorios
- 🛡️ **Confirmaciones inteligentes** - Protección contra operaciones destructivas
- 🎨 **Interfaz enriquecida** - Formato de texto mejorado con Rich

## Instalación

1. Descarga el script `filesystem_mcp.py`
2. Asegúrate de tener Python 3.8+ instalado
3. Las dependencias necesarias (llama-cpp-python y rich) se instalarán automáticamente la primera vez que ejecutes el script
4. El modelo necesario se descargará automáticamente durante la primera ejecución (requiere aproximadamente 5GB de espacio)

## Requisitos

- Python 3.8 o superior
- Modelo Llama-3 (se descargará automáticamente si no está disponible)
- Bibliotecas: llama-cpp-python, rich (instalación automática si faltan)
- Espacio en disco: ~5GB para el modelo

## Uso

1. Ejecuta el script: `python filesystem_mcp.py`
2. El programa te guiará para configurar el directorio base y descargar el modelo si es necesario
3. Interactúa con el asistente usando lenguaje natural

### Comandos especiales

- `/ayuda` - Muestra la lista de comandos disponibles
- `/dir [ruta]` - Cambia el directorio base
- `/permisos` - Gestiona los niveles de permiso
- `/herramientas` - Lista las operaciones disponibles
- `/salir` - Termina el programa

## Ejemplos de uso

### Organizar archivos

Tú: Mueve todos los archivos PDF a la carpeta Documentos

[El sistema ejecuta la operación]

Asistente: Se movieron 2 archivos a Documentos exitosamente.


### Obtener información

Tú: ¿Qué hay en la carpeta actual?

[El sistema lista el contenido]

Asistente: El contenido de la carpeta actual es:
  * Archivos: archivo1.txt, datos.csv
  * Carpetas: Documentos, Imágenes

## Autor

Desarrollado por Leonardo con ayuda de Claude Sonnet 3.7 thinking mode (Abril 2025)
