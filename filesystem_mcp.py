#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FileSystem MCP - Un asistente inteligente para operaciones del sistema de archivos
utilizando el modelo Llama-3 con capacidades de function calling optimizadas.

Este script implementa un sistema Model-Context-Protocol (MCP) que permite realizar
operaciones en el sistema de archivos a través de lenguaje natural, aprovechando
las capacidades específicas de function calling del modelo Llama-3.

⚠️ Este proyecto no está pensado para uso en producción. Es un ejercicio didáctico
que desarrollé con fines de aprendizaje y exploración del uso de LLMs locales 
combinados con el patrón Model-Context-Protocol (MCP).

Toda la lógica está contenida en un único archivo con el objetivo de que cualquier
persona que esté empezando a experimentar con estos temas pueda seguir fácilmente 
la dinámica completa de un asistente LLM autocontenido.

Author: Leonardo - Mejorado con ayuda de Claude Sonnet 3.7 thinking mode
Date: Abril 2025
"""

import os
import shutil
import json
import logging
import re
import time
import getpass
from typing import Dict, Any, Callable, Optional, List, Tuple, Union, Set
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import argparse
import hashlib
import stat
from pathlib import Path
import importlib
import sys

# Bibliotecas externas
from llama_cpp import Llama
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.logging import RichHandler
    from rich.highlighter import ReprHighlighter
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    print("📦 Biblioteca 'rich' no encontrada. Instalando...")
    import subprocess
    subprocess.check_call(["pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.logging import RichHandler
    from rich.highlighter import ReprHighlighter
    from rich.syntax import Syntax
    RICH_AVAILABLE = True

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Inicializar consola Rich para salida con formato
console = Console()

# Configuración básica y rutas
CONFIG = {
    # Rutas y archivos
    "MODEL_PATH": "Meta-Llama-3-8B-Instruct-function-calling-Q4_K_M.gguf",
    "BASE_DIR": "",
    "CONFIG_DIR": os.path.expanduser("~/.mcp_fs"),
    "LOG_FILE": os.path.expanduser("~/.mcp_fs/mcp_fs.log"),
    "HISTORY_FILE": os.path.expanduser("~/.mcp_fs/history.jsonl"),
    "PERMISSIONS_FILE": os.path.expanduser("~/.mcp_fs/permissions.json"),
    
    # Parámetros del modelo
    "MODEL_PARAMS": {
        "verbose": False,
        "n_gpu_layers": 32,  # Ajustar según la GPU disponible
        "n_ctx": 8192,
        "seed": 42,  # Semilla para reproducibilidad
    },
    
    # Temperaturas para diferentes operaciones
    "TEMPERATURE": {
        "decision": 0.2,  # Baja temperatura para decisiones precisas
        "conversation": 0.7,  # Mayor temperatura para respuestas naturales
    },
    
    # Configuración de seguridad
    "SECURITY": {
        "confirm_destructive": True,  # Confirmar operaciones destructivas
        "confirm_all": False,  # Confirmar todas las operaciones
        "allowed_extensions": ["*"],  # * = todas las extensiones
        "forbidden_extensions": ["exe", "bat", "sh", "ps1"],  # Archivos peligrosos
        "max_batch_size": 10,  # Número máximo de operaciones por lote
    },
    
    # Configuración de UI
    "UI": {
        "user_color": "green",
        "assistant_color": "blue",
        "error_color": "red",
        "success_color": "green",
        "info_color": "yellow",
        "show_timestamps": True,
    },
}

# Lista de operaciones destructivas que requieren confirmación adicional
DESTRUCTIVE_OPERATIONS = [
    "delete_file", 
    "delete_directory", 
    "move_file", 
    "move_directory",
    "rename_file",
    "rename_directory"
]

# ==============================================================================
# CLASES DE UTILIDAD
# ==============================================================================

class ChatCommands:
    """Gestiona los comandos especiales en el chat que no se envían al modelo."""
    
    def __init__(self, mcp_instance):
        """Inicializa el gestor de comandos."""
        self.mcp = mcp_instance
        self.prefix = "/"
        self.commands = {
            "ayuda": self.cmd_help,
            "help": self.cmd_help,
            "comandos": self.cmd_help,
            "permisos": self.cmd_permissions,
            "herramientas": self.cmd_tools,
            "dir": self.cmd_change_dir,
            "salir": self.cmd_exit,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "estado": self.cmd_status,
            "status": self.cmd_status,
            "limpiar": self.cmd_clear,
            "clear": self.cmd_clear,
        }
    
    def is_command(self, text):
        """Verifica si un texto es un comando."""
        return text.startswith(self.prefix)
    
    def process_command(self, text):
        """Procesa un comando y devuelve la respuesta."""
        # Eliminar el prefijo
        cmd_text = text[len(self.prefix):].strip()
        
        # Separar el comando y los argumentos
        parts = cmd_text.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        # Ejecutar el comando si existe
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            return f"❌ Comando desconocido: {cmd}. Usa '{self.prefix}ayuda' para ver los comandos disponibles."
    
    def cmd_help(self, args):
        """Muestra la ayuda de los comandos disponibles."""
        help_text = [
            "[bold blue]Comandos Disponibles:[/bold blue]",
            f"{self.prefix}ayuda - Muestra esta ayuda",
            f"{self.prefix}permisos - Gestiona los permisos de operaciones",
            f"{self.prefix}herramientas - Lista las herramientas disponibles",
            f"{self.prefix}dir [ruta] - Cambia el directorio base",
            f"{self.prefix}estado - Muestra el estado del sistema",
            f"{self.prefix}limpiar - Limpia la pantalla",
            f"{self.prefix}salir - Sale del programa"
        ]
        return "\n".join(help_text)
    
    def cmd_permissions(self, args):
        """Gestiona los permisos de operaciones."""
        if not args:
            # Mostrar estado actual y opciones
            levels = [level.name for level in PermissionLevel]
            current = self.mcp.permissions.current_level.name
            
            perm_text = [
                f"[bold blue]Nivel de Permisos Actual:[/bold blue] [yellow]{current}[/yellow]",
                "",
                "[bold]Niveles disponibles:[/bold]",
                "1. READ_ONLY - Solo operaciones de lectura",
                "2. BASIC - Operaciones básicas (crear)",
                "3. STANDARD - Operaciones estándar (mover, copiar)",
                "4. ADVANCED - Operaciones avanzadas (renombrar)",
                "5. ADMIN - Todas las operaciones (eliminar)",
                "",
                f"Para cambiar, usa '{self.prefix}permisos NIVEL' (ej: '{self.prefix}permisos STANDARD')"
            ]
            return "\n".join(perm_text)
        else:
            # Intentar cambiar el nivel de permisos
            try:
                new_level = args.strip().upper()
                level = PermissionLevel[new_level]
                self.mcp.permissions.set_permission_level(level)
                return f"✅ Nivel de permisos cambiado a: [bold green]{level.name}[/bold green]"
            except KeyError:
                return f"❌ Nivel de permisos inválido: {args}. Usa uno de: {', '.join([l.name for l in PermissionLevel])}"
    
    def cmd_tools(self, args):
        """Lista las herramientas disponibles para el asistente."""
        # Obtener operaciones permitidas
        allowed = self.mcp.permissions.get_allowed_operations()
        
        # Agrupar por categorías
        categories = {
            "Lectura": ["list_directory", "get_file_info"],
            "Creación": ["create_directory"],
            "Copia": ["copy_file", "copy_directory", "copy_multiple_files"],
            "Movimiento": ["move_file", "move_directory", "move_multiple_files"],
            "Renombrado": ["rename_file", "rename_directory"],
            "Eliminación": ["delete_file", "delete_directory", "delete_multiple_files"]
        }
        
        # Construir la salida
        tools_text = ["[bold blue]Herramientas Disponibles:[/bold blue]"]
        
        for category, operations in categories.items():
            category_ops = []
            for op in operations:
                if op in allowed:
                    desc = next((f["description"] for f in AVAILABLE_FUNCTIONS if f["name"] == op), "")
                    category_ops.append(f"  • [green]{op}[/green] - {desc}")
                else:
                    category_ops.append(f"  • [dim]{op}[/dim] - [red]No permitido[/red]")
            
            if category_ops:
                tools_text.append(f"\n[bold]{category}:[/bold]")
                tools_text.extend(category_ops)
        
        return "\n".join(tools_text)
    
    def cmd_change_dir(self, args):
        """Cambia el directorio base."""
        if not args:
            return f"Directorio base actual: [yellow]{self.mcp.config['BASE_DIR']}[/yellow]\nPara cambiar usa '{self.prefix}dir nueva_ruta'"
        
        try:
            new_dir = os.path.abspath(args)
            if not os.path.exists(new_dir):
                if Confirm.ask(f"El directorio '{new_dir}' no existe. ¿Deseas crearlo?"):
                    os.makedirs(new_dir, exist_ok=True)
                else:
                    return "❌ Operación cancelada."
            
            # Actualizar la configuración
            old_dir = self.mcp.config["BASE_DIR"]
            self.mcp.config["BASE_DIR"] = new_dir
            
            # Guardar configuración para futuras ejecuciones
            if Confirm.ask("¿Guardar esta configuración para futuras ejecuciones?"):
                config_dir = self.mcp.config["CONFIG_DIR"]
                config_file = os.path.join(config_dir, "config.json")
                
                try:
                    # Cargar configuración existente si existe
                    if os.path.exists(config_file):
                        with open(config_file, 'r', encoding='utf-8') as f:
                            saved_config = json.load(f)
                    else:
                        saved_config = {}
                    
                    # Actualizar directorio base
                    saved_config["BASE_DIR"] = new_dir
                    
                    # Guardar configuración
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(saved_config, f, indent=2)
                except Exception as e:
                    return f"⚠️ Directorio cambiado pero no se pudo guardar la configuración: {str(e)}"
            
            return f"✅ Directorio base cambiado de [dim]{old_dir}[/dim] a [green]{new_dir}[/green]"
            
        except Exception as e:
            return f"❌ Error al cambiar el directorio: {str(e)}"
    
    def cmd_status(self, args):
        """Muestra el estado del sistema."""
        status_text = [
            "[bold blue]Estado del Sistema:[/bold blue]",
            f"Directorio base: [yellow]{self.mcp.config['BASE_DIR']}[/yellow]",
            f"Modelo: [yellow]{self.mcp.config['MODEL_PATH']}[/yellow]",
            f"Nivel de permisos: [yellow]{self.mcp.permissions.current_level.name}[/yellow]",
            f"Tokens utilizados: [yellow]{self.mcp.token_count}[/yellow]",
            f"Llamadas a funciones: [yellow]{self.mcp.function_calls}[/yellow]",
            f"Confirmar operaciones destructivas: [yellow]{self.mcp.config['SECURITY']['confirm_destructive']}[/yellow]",
            f"Confirmar todas las operaciones: [yellow]{self.mcp.config['SECURITY']['confirm_all']}[/yellow]"
        ]
        return "\n".join(status_text)
    
    def cmd_clear(self, args):
        """Limpia la pantalla."""
        os.system('cls' if os.name == 'nt' else 'clear')
        return "✅ Pantalla limpiada."
    
    def cmd_exit(self, args):
        """Salir del programa."""
        raise KeyboardInterrupt()

class PermissionLevel(Enum):
    """Niveles de permiso para operaciones del sistema de archivos."""
    READ_ONLY = auto()     # Solo lectura (list_directory, get_file_info)
    BASIC = auto()         # Operaciones básicas no destructivas (crear)
    STANDARD = auto()      # Operaciones estándar (mover, copiar)
    ADVANCED = auto()      # Operaciones avanzadas (renombrar)
    ADMIN = auto()         # Todas las operaciones (eliminar)

@dataclass
class Permission:
    """Define permisos para operaciones específicas."""
    operation: str
    description: str
    min_level: PermissionLevel
    requires_confirmation: bool = False

@dataclass
class FunctionCallResult:
    """Estructura para almacenar el resultado de una llamada a función."""
    success: bool
    result: Dict[str, Any]
    function_name: str
    arguments: Dict[str, Any]
    execution_time: float

class PermissionManager:
    """
    Gestiona los permisos para operaciones del sistema de archivos.
    Controla qué operaciones están permitidas según el nivel de acceso configurado.
    """
    
    def __init__(self, permissions_file: str, current_level: PermissionLevel = PermissionLevel.STANDARD):
        self.permissions_file = permissions_file
        self.current_level = current_level
        self.permissions: Dict[str, Permission] = {}
        self._load_permissions()
    
    def _load_permissions(self):
        """Carga la configuración de permisos desde el archivo o usa valores predeterminados."""
        # Definir permisos predeterminados
        default_permissions = [
            Permission("list_directory", "Listar contenido de un directorio", PermissionLevel.READ_ONLY),
            Permission("get_file_info", "Obtener información de un archivo", PermissionLevel.READ_ONLY),
            Permission("create_directory", "Crear un directorio nuevo", PermissionLevel.BASIC),
            Permission("copy_file", "Copiar un archivo", PermissionLevel.STANDARD),
            Permission("copy_directory", "Copiar un directorio completo", PermissionLevel.STANDARD),
            Permission("move_file", "Mover un archivo", PermissionLevel.STANDARD, True),
            Permission("move_directory", "Mover un directorio", PermissionLevel.STANDARD, True),
            Permission("rename_file", "Renombrar un archivo", PermissionLevel.ADVANCED, True),
            Permission("rename_directory", "Renombrar un directorio", PermissionLevel.ADVANCED, True),
            Permission("delete_file", "Eliminar un archivo", PermissionLevel.ADMIN, True),
            Permission("delete_directory", "Eliminar un directorio", PermissionLevel.ADMIN, True),
            Permission("move_multiple_files", "Mover múltiples archivos", PermissionLevel.STANDARD, True),
            Permission("copy_multiple_files", "Copiar múltiples archivos", PermissionLevel.STANDARD, True),
            Permission("delete_multiple_files", "Eliminar múltiples archivos", PermissionLevel.ADMIN, True),
        ]
        
        # Crear diccionario de permisos
        for perm in default_permissions:
            self.permissions[perm.operation] = perm
        
        # Intentar cargar desde archivo si existe
        if os.path.exists(self.permissions_file):
            try:
                with open(self.permissions_file, 'r', encoding='utf-8') as f:
                    perm_data = json.load(f)
                
                if "current_level" in perm_data:
                    self.current_level = PermissionLevel[perm_data["current_level"]]
                
                # Actualizar permisos personalizados
                if "custom_permissions" in perm_data:
                    for op_name, op_data in perm_data["custom_permissions"].items():
                        if op_name in self.permissions:
                            self.permissions[op_name].min_level = PermissionLevel[op_data["min_level"]]
                            self.permissions[op_name].requires_confirmation = op_data["requires_confirmation"]
            except Exception as e:
                logging.warning(f"Error al cargar permisos: {str(e)}. Usando valores predeterminados.")
    
    def save_permissions(self):
        """Guarda la configuración de permisos actual en un archivo."""
        os.makedirs(os.path.dirname(self.permissions_file), exist_ok=True)
        
        perm_data = {
            "current_level": self.current_level.name,
            "custom_permissions": {}
        }
        
        for op_name, perm in self.permissions.items():
            perm_data["custom_permissions"][op_name] = {
                "min_level": perm.min_level.name,
                "requires_confirmation": perm.requires_confirmation
            }
        
        with open(self.permissions_file, 'w', encoding='utf-8') as f:
            json.dump(perm_data, f, indent=2)
    
    def is_operation_allowed(self, operation_name: str) -> bool:
        """Verifica si una operación está permitida con el nivel de acceso actual."""
        if operation_name not in self.permissions:
            return False
        
        perm = self.permissions[operation_name]
        return self.current_level.value >= perm.min_level.value
    
    def requires_confirmation(self, operation_name: str) -> bool:
        """Verifica si una operación requiere confirmación adicional."""
        if operation_name not in self.permissions:
            return True  # Si no sabemos qué es, mejor confirmar
        
        return self.permissions[operation_name].requires_confirmation
    
    def set_permission_level(self, level: PermissionLevel):
        """Establece el nivel de permiso actual."""
        self.current_level = level
        self.save_permissions()
    
    def get_allowed_operations(self) -> List[str]:
        """Obtiene la lista de operaciones permitidas con el nivel actual."""
        return [op for op, perm in self.permissions.items() 
                if self.current_level.value >= perm.min_level.value]

# ==============================================================================
# DEFINICIÓN DE MENSAJES PARA EL SISTEMA
# ==============================================================================

# Prompt principal del sistema con instrucciones para el modelo
SYSTEM_PROMPT = """
Eres un asistente especializado en operaciones del sistema de archivos. Estás equipado con herramientas para manipular archivos y directorios de manera segura y eficiente.

## TU ROL
1. ANALIZAR la petición del usuario para entender qué operación de sistema de archivos necesita
2. DECIDIR si es necesario usar una herramienta o simplemente responder conversacionalmente
3. EJECUTAR la acción apropiada usando la herramienta correcta
4. COMUNICAR el resultado de manera clara al usuario

## REGLAS ESTRICTAS PARA EL USO DE HERRAMIENTAS

1. CUANDO USAR HERRAMIENTAS:
   - Usa list_directory cuando el usuario quiera ver contenido de carpetas
   - Usa get_file_info cuando quiera detalles sobre archivos específicos
   - Usa create_directory cuando quiera crear nuevas carpetas
   - Usa move_file/move_directory cuando quiera mover archivos/carpetas
   - Usa copy_file/copy_directory cuando quiera copiar archivos/carpetas
   - Usa rename_file/rename_directory cuando quiera renombrar archivos/carpetas
   - Usa delete_file/delete_directory cuando quiera eliminar archivos/carpetas
   - Para operaciones múltiples, usa las versiones con "multiple" en el nombre

2. NORMAS SOBRE FORMATO DE RESPUESTAS:
   - Cuando uses una herramienta, tu respuesta DEBE ser EXACTAMENTE un objeto JSON con el formato correcto
   - NO incluyas texto adicional antes o después del JSON
   - NO uses comillas, acentos graves (```) o formateo markdown alrededor del JSON
   - El JSON debe ser válido y seguir EXACTAMENTE el formato especificado

3. NORMAS PARA OPERACIONES MÚLTIPLES:
   - Si el usuario menciona varios archivos a la vez, usa la versión "multiple" de la función
   - No solicites confirmación para cada archivo individual

4. INFORMACIÓN INCOMPLETA:
   - Si la petición del usuario no especifica todos los detalles necesarios, PREGUNTA específicamente por la información que falta
   - Asume rutas relativas al directorio base cuando no se especifique la ruta completa

## EJEMPLOS DE RESPUESTAS CORRECTAS

Ejemplo 1 - Listar directorio:
{"function_call":{"name":"list_directory","arguments":{"path":"."}}}

Ejemplo 2 - Mover múltiples archivos:
{"function_call":{"name":"move_multiple_files","arguments":{"files":["doc1.txt","doc2.txt"],"destination":"Documentos"}}}

Ejemplo 3 - Crear directorio:
{"function_call":{"name":"create_directory","arguments":{"path":"Proyectos/Web"}}}

Para conversaciones normales que no requieren herramientas, responde de manera natural y conversacional.
"""

# Ejemplos few-shot para enseñar al modelo el comportamiento correcto
FEW_SHOT_EXAMPLES = [
    # Ejemplo 1: Crear directorio
    {"role": "user", "content": "Crea una carpeta llamada Documentos"},
    {"role": "assistant", "content": '{"function_call":{"name":"create_directory","arguments":{"path":"Documentos"}}}'},
    {"role": "function", "name": "create_directory", "content": '{"result": "Directorio \'Documentos\' creado exitosamente."}'},
    {"role": "assistant", "content": "¡Listo! He creado la carpeta 'Documentos' para ti."},
    
    # Ejemplo 2: Listar directorio
    {"role": "user", "content": "Muéstrame qué hay en la carpeta actual"},
    {"role": "assistant", "content": '{"function_call":{"name":"list_directory","arguments":{"path":"."}}}'},
    {"role": "function", "name": "list_directory", "content": '{"result": {"files": ["archivo1.txt", "archivo2.docx"], "directories": ["Documentos"]}}'},
    {"role": "assistant", "content": "El contenido de la carpeta actual es:\n\nArchivos:\n- archivo1.txt\n- archivo2.docx\n\nCarpetas:\n- Documentos"},
    
    # Ejemplo 3: Mover múltiples archivos
    {"role": "user", "content": "Mueve los archivos texto1.txt y texto2.txt a la carpeta Documentos"},
    {"role": "assistant", "content": '{"function_call":{"name":"move_multiple_files","arguments":{"files":["texto1.txt","texto2.txt"],"destination":"Documentos"}}}'},
    {"role": "function", "name": "move_multiple_files", "content": '{"result": "Se movieron 2 archivos a Documentos exitosamente."}'},
    {"role": "assistant", "content": "He movido los archivos texto1.txt y texto2.txt a la carpeta Documentos."},
    
    # Ejemplo 4: Respuesta conversacional
    {"role": "user", "content": "¿Qué tipos de archivos puedo manejar con este asistente?"},
    {"role": "assistant", "content": "Puedo ayudarte a manejar prácticamente cualquier tipo de archivo en tu sistema. Puedo crear carpetas, mover, copiar, renombrar y eliminar archivos o directorios completos. También puedo listar el contenido de carpetas y mostrarte información detallada sobre archivos específicos como su tamaño y fecha de modificación.\n\n¿Hay alguna operación específica con archivos que necesites realizar?"}
]

# ==============================================================================
# DEFINICIÓN DE JSON SCHEMA PARA FUNCIONES
# ==============================================================================

AVAILABLE_FUNCTIONS = [
    # === OPERACIONES DE LECTURA ===
    {
        "name": "list_directory",
        "description": "Lista el contenido de un directorio, mostrando archivos y subdirectorios.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta del directorio a listar, relativa al directorio base. Utiliza '.' para el directorio actual.",
                    "examples": [".", "Documentos", "Proyectos/Web"]
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "get_file_info",
        "description": "Obtiene información detallada sobre un archivo: tamaño, fecha de creación, modificación, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta del archivo sobre el que se quiere obtener información, relativa al directorio base.",
                    "examples": ["documento.txt", "Fotos/vacaciones.jpg"]
                }
            },
            "required": ["path"]
        }
    },
    
    # === OPERACIONES DE CREACIÓN ===
    {
        "name": "create_directory",
        "description": "Crea un nuevo directorio en la ruta especificada.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta del directorio a crear, relativa al directorio base. Puede incluir subdirectorios que se crearán automáticamente.",
                    "examples": ["NuevaCarpeta", "Proyectos/Web/Frontend"]
                }
            },
            "required": ["path"]
        }
    },
    
    # === OPERACIONES DE COPIA ===
    {
        "name": "copy_file",
        "description": "Copia un archivo de una ubicación a otra.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta de origen del archivo a copiar, relativa al directorio base.",
                    "examples": ["documento.txt", "Fotos/vacaciones.jpg"]
                },
                "destination": {
                    "type": "string",
                    "description": "Ruta de destino donde copiar el archivo, relativa al directorio base.",
                    "examples": ["Backup/documento.txt", "Compartido/vacaciones.jpg"]
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "copy_directory",
        "description": "Copia un directorio completo y su contenido de una ubicación a otra.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta de origen del directorio a copiar, relativa al directorio base.",
                    "examples": ["Documentos", "Proyectos/Web"]
                },
                "destination": {
                    "type": "string",
                    "description": "Ruta de destino donde copiar el directorio, relativa al directorio base.",
                    "examples": ["Backup/Documentos", "Compartido/Web"]
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "copy_multiple_files",
        "description": "Copia múltiples archivos a un directorio de destino.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de rutas de los archivos a copiar, relativas al directorio base.",
                    "examples": [["documento1.txt", "documento2.txt"], ["Fotos/vacaciones1.jpg", "Fotos/vacaciones2.jpg"]]
                },
                "destination": {
                    "type": "string",
                    "description": "Directorio de destino donde copiar los archivos, relativo al directorio base.",
                    "examples": ["Backup", "Compartido/Fotos"]
                }
            },
            "required": ["files", "destination"]
        }
    },
    
    # === OPERACIONES DE MOVIMIENTO ===
    {
        "name": "move_file",
        "description": "Mueve un archivo de una ubicación a otra.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta de origen del archivo a mover, relativa al directorio base.",
                    "examples": ["documento.txt", "Fotos/vacaciones.jpg"]
                },
                "destination": {
                    "type": "string",
                    "description": "Ruta de destino donde mover el archivo, relativa al directorio base.",
                    "examples": ["Archivados/documento.txt", "Compartido/vacaciones.jpg"]
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "move_directory",
        "description": "Mueve un directorio completo y su contenido de una ubicación a otra.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta de origen del directorio a mover, relativa al directorio base.",
                    "examples": ["Documentos", "Proyectos/Web"]
                },
                "destination": {
                    "type": "string",
                    "description": "Ruta de destino donde mover el directorio, relativa al directorio base.",
                    "examples": ["Archivados/Documentos", "Trabajo/Web"]
                }
            },
            "required": ["source", "destination"]
        }
    },
    {
        "name": "move_multiple_files",
        "description": "Mueve múltiples archivos a un directorio de destino.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de rutas de los archivos a mover, relativas al directorio base.",
                    "examples": [["documento1.txt", "documento2.txt"], ["Fotos/vacaciones1.jpg", "Fotos/vacaciones2.jpg"]]
                },
                "destination": {
                    "type": "string",
                    "description": "Directorio de destino donde mover los archivos, relativo al directorio base.",
                    "examples": ["Archivados", "Compartido/Fotos"]
                }
            },
            "required": ["files", "destination"]
        }
    },
    
    # === OPERACIONES DE RENOMBRADO ===
    {
        "name": "rename_file",
        "description": "Renombra un archivo manteniendo su ubicación.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta actual del archivo, relativa al directorio base.",
                    "examples": ["documento_viejo.txt", "Fotos/img001.jpg"]
                },
                "new_name": {
                    "type": "string",
                    "description": "Nuevo nombre para el archivo, sin incluir la ruta.",
                    "examples": ["documento_nuevo.txt", "vacaciones_2024.jpg"]
                }
            },
            "required": ["source", "new_name"]
        }
    },
    {
        "name": "rename_directory",
        "description": "Renombra un directorio manteniendo su ubicación.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Ruta actual del directorio, relativa al directorio base.",
                    "examples": ["Docs", "Proyectos/Viejo"]
                },
                "new_name": {
                    "type": "string",
                    "description": "Nuevo nombre para el directorio, sin incluir la ruta.",
                    "examples": ["Documentos", "Actual"]
                }
            },
            "required": ["source", "new_name"]
        }
    },
    
    # === OPERACIONES DE ELIMINACIÓN ===
    {
        "name": "delete_file",
        "description": "Elimina un archivo del sistema de archivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta del archivo a eliminar, relativa al directorio base.",
                    "examples": ["documento.txt", "Fotos/vacaciones.jpg"]
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "delete_directory",
        "description": "Elimina un directorio y todo su contenido del sistema de archivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta del directorio a eliminar, relativa al directorio base.",
                    "examples": ["Temp", "Proyectos/Obsoleto"]
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "delete_multiple_files",
        "description": "Elimina múltiples archivos del sistema de archivos.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de rutas de los archivos a eliminar, relativas al directorio base.",
                    "examples": [["temp1.txt", "temp2.txt"], ["Logs/old1.log", "Logs/old2.log"]]
                }
            },
            "required": ["files"]
        }
    }
]

# ==============================================================================
# CLASE PRINCIPAL FileSystemMCP
# ==============================================================================

class FileSystemMCP:
    """
    Implementación avanzada del patrón Model-Call-Process para operaciones del sistema de archivos.
    Utiliza un modelo Llama-3 con capacidades de function calling para interpretar y ejecutar
    operaciones de archivos a través de lenguaje natural.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el asistente MCP con la configuración proporcionada.
        
        Args:
            config: Diccionario con toda la configuración necesaria
        """
        self.config = config
        
        # Crear directorios necesarios
        os.makedirs(config["BASE_DIR"], exist_ok=True)
        os.makedirs(config["CONFIG_DIR"], exist_ok=True)
        
        # Inicializar sistema de logging
        self._setup_logging()
        
        # Inicializar gestor de permisos
        self.permissions = PermissionManager(config["PERMISSIONS_FILE"])
        
        # Historial de mensajes para el modelo
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Añadir ejemplos few-shot al historial de mensajes
        self.messages.extend(FEW_SHOT_EXAMPLES)
        
        # Inicializar el sistema de comandos
        self.chat_commands = None  # Se inicializará desde main()
        
        # Inicializar el modelo
        self._initialize_model()
        
        # Mapeo de nombres de funciones a sus implementaciones
        self._initialize_function_map()
        
        # Contador de tokens y estadísticas
        self.token_count = 0
        self.function_calls = 0
        
        logging.info(f"FileSystemMCP inicializado. Directorio base: {config['BASE_DIR']}")
        
        # Mostrar información de inicio
        console.print(Panel.fit(
            f"[bold green]FileSystem MCP - Asistente Inteligente[/bold green]\n"
            f"[yellow]Directorio base:[/yellow] {self.config['BASE_DIR']}\n"
            f"[yellow]Nivel de permisos:[/yellow] {self.permissions.current_level.name}\n"
            f"[dim]Usa '{self.chat_commands.prefix if self.chat_commands else "/"}ayuda' para ver los comandos disponibles[/dim]",
            title="Sistema Inicializado",
            border_style="green"
        ))
    
    def _setup_logging(self):
        """Configura el sistema de logging con soporte para Rich."""
        os.makedirs(os.path.dirname(self.config["LOG_FILE"]), exist_ok=True)
        
        # Configurar el logger principal
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config["LOG_FILE"]),
                RichHandler(rich_tracebacks=True) if RICH_AVAILABLE else logging.StreamHandler()
            ]
        )

    def _initialize_model(self):
        """Inicializa el modelo Llama con los parámetros configurados."""
        try:
            logging.info(f"Cargando modelo desde {self.config['MODEL_PATH']}...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[yellow]Cargando modelo...", total=None)
                self.llm = Llama(model_path=self.config["MODEL_PATH"], **self.config["MODEL_PARAMS"])
                progress.update(task, completed=True)
            
            logging.info("Modelo cargado exitosamente")
            console.print("[green]Modelo cargado exitosamente[/green]")
        except Exception as e:
            logging.critical(f"Error al cargar el modelo: {str(e)}")
            console.print(f"[bold red]Error al cargar el modelo: {str(e)}[/bold red]")
            raise
    
    def _initialize_function_map(self):
        """Inicializa el mapeo de nombres de funciones a sus implementaciones."""
        self.function_map = {
            # Operaciones de lectura
            "list_directory": self.list_directory,
            "get_file_info": self.get_file_info,
            
            # Operaciones de creación
            "create_directory": self.create_directory,
            
            # Operaciones de copia
            "copy_file": self.copy_file,
            "copy_directory": self.copy_directory,
            "copy_multiple_files": self.copy_multiple_files,
            
            # Operaciones de movimiento
            "move_file": self.move_file,
            "move_directory": self.move_directory,
            "move_multiple_files": self.move_multiple_files,
            
            # Operaciones de renombrado
            "rename_file": self.rename_file,
            "rename_directory": self.rename_directory,
            
            # Operaciones de eliminación
            "delete_file": self.delete_file,
            "delete_directory": self.delete_directory,
            "delete_multiple_files": self.delete_multiple_files,
        }
    
    def _validate_path(self, path: str) -> str:
        """
        Valida que una ruta sea segura (sin path traversal) y la normaliza.
        
        Args:
            path: Ruta relativa para validar
            
        Returns:
            Ruta normalizada
            
        Raises:
            ValueError: Si la ruta intenta acceder fuera del directorio base
        """
        # Eliminar caracteres peligrosos y normalizar separadores
        clean_path = os.path.normpath(path).replace('\\', '/')
        
        # Comprobar path traversal
        if clean_path.startswith('..') or '/../' in clean_path:
            raise ValueError(f"Ruta no permitida: '{path}'. No se permite acceder a directorios superiores.")
        
        # Comprobar extensiones prohibidas
        if '.' in clean_path:
            extension = clean_path.split('.')[-1].lower()
            if (extension in self.config["SECURITY"]["forbidden_extensions"] and 
                '*' not in self.config["SECURITY"]["allowed_extensions"]):
                raise ValueError(f"Extensión de archivo prohibida: '{extension}'")
        
        return clean_path
    
    def _get_absolute_path(self, relative_path: str) -> str:
        """
        Convierte una ruta relativa a absoluta dentro del directorio base.
        
        Args:
            relative_path: Ruta relativa al directorio base
            
        Returns:
            Ruta absoluta
        """
        validated_path = self._validate_path(relative_path)
        return os.path.join(self.config["BASE_DIR"], validated_path)
    
    def _confirm_operation(self, operation_name: str, details: str) -> bool:
        """
        Solicita confirmación al usuario para una operación potencialmente peligrosa.
        
        Args:
            operation_name: Nombre de la operación a confirmar
            details: Detalles de la operación para mostrar al usuario
            
        Returns:
            True si el usuario confirma, False en caso contrario
        """
        # Comprobar si esta operación requiere confirmación
        requires_confirmation = False
        
        # Confirmar siempre si así está configurado
        if self.config["SECURITY"]["confirm_all"]:
            requires_confirmation = True
        
        # Confirmar operaciones destructivas si así está configurado
        elif (self.config["SECURITY"]["confirm_destructive"] and 
              operation_name in DESTRUCTIVE_OPERATIONS):
            requires_confirmation = True
        
        # Confirmar si los permisos lo requieren
        elif self.permissions.requires_confirmation(operation_name):
            requires_confirmation = True
        
        if requires_confirmation:
            # Mostrar detalles de la operación
            if operation_name in DESTRUCTIVE_OPERATIONS:
                console.print(f"[bold yellow]⚠️ Operación potencialmente destructiva: {operation_name}[/bold yellow]")
            console.print(f"[yellow]Detalles: {details}[/yellow]")
            
            # Solicitar confirmación
            return Confirm.ask("¿Confirmar esta operación?")
        
        return True
    
    def _check_permission(self, operation_name: str) -> bool:
        """
        Verifica si una operación está permitida según la configuración de permisos.
        
        Args:
            operation_name: Nombre de la operación a verificar
            
        Returns:
            True si la operación está permitida, False en caso contrario
        """
        allowed = self.permissions.is_operation_allowed(operation_name)
        
        if not allowed:
            logging.warning(f"Operación '{operation_name}' no permitida con el nivel de permisos actual")
            console.print(f"[bold red]⛔ Operación '{operation_name}' no permitida.[/bold red]")
            console.print(f"[red]Se requiere un nivel de permisos mayor.[/red]")
        
        return allowed
    
    def _extract_function_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extrae información de una llamada a función desde la respuesta del modelo.
        Implementa un algoritmo robusto que intenta varios métodos para extraer el JSON.
        
        Args:
            content: Contenido de la respuesta del modelo
            
        Returns:
            Diccionario con la información de la llamada a función, o None si no hay llamada
        """
        # 1. Intento directo: el contenido ya es un JSON válido
        content = content.strip()
        try:
            data = json.loads(content)
            if "function_call" in data:
                return {
                    "name": data["function_call"]["name"],
                    "arguments": data["function_call"]["arguments"]
                }
            return None
        except json.JSONDecodeError:
            pass
        
        # 2. Buscar estructura JSON con regex
        try:
            # Patrón más flexible para encontrar la estructura JSON
            pattern = r'({[\s\S]*"function_call"[\s\S]*})'
            matches = re.search(pattern, content)
            if matches:
                # Obtener el texto que coincide con el patrón
                json_str = matches.group(1)
                # Limpiar posibles caracteres no JSON
                json_str = re.sub(r'```json|```', '', json_str).strip()
                data = json.loads(json_str)
                if "function_call" in data:
                    return {
                        "name": data["function_call"]["name"],
                        "arguments": data["function_call"]["arguments"]
                    }
        except Exception:
            pass
        
        # 3. Intentar encontrar function_call anidado en cualquier estructura
        try:
            # Buscar cualquier objeto JSON en el texto
            pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
            for match in re.finditer(pattern, content):
                try:
                    json_obj = json.loads(match.group(0))
                    # Buscar recursivamente "function_call" en el objeto
                    def find_function_call(obj):
                        if isinstance(obj, dict):
                            if "function_call" in obj:
                                return obj["function_call"]
                            for key, value in obj.items():
                                result = find_function_call(value)
                                if result:
                                    return result
                        elif isinstance(obj, list):
                            for item in obj:
                                result = find_function_call(item)
                                if result:
                                    return result
                        return None
                    
                    fc = find_function_call(json_obj)
                    if fc and "name" in fc and "arguments" in fc:
                        return {
                            "name": fc["name"],
                            "arguments": fc["arguments"]
                        }
                except:
                    continue
        except Exception:
            pass
        
        # No se encontró ninguna llamada a función válida
        return None
    
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> FunctionCallResult:
        """
        Ejecuta una función del sistema de archivos con los argumentos proporcionados.
        
        Args:
            function_name: Nombre de la función a ejecutar
            arguments: Argumentos para la función
            
        Returns:
            Resultado de la ejecución
        """
        start_time = time.time()
        
        # Verificar que la función existe
        if function_name not in self.function_map:
            logging.error(f"Función '{function_name}' no implementada")
            return FunctionCallResult(
                success=False,
                result={"error": f"Función '{function_name}' no implementada o no disponible"},
                function_name=function_name,
                arguments=arguments,
                execution_time=time.time() - start_time
            )

        # Verificar permisos
        if not self._check_permission(function_name):
            return FunctionCallResult(
                success=False,
                result={"error": f"Operación '{function_name}' no permitida con el nivel de permisos actual"},
                function_name=function_name,
                arguments=arguments,
                execution_time=time.time() - start_time
            )
        
        # Preparar detalles para confirmación
        details = f"Función: {function_name}, Argumentos: {json.dumps(arguments, ensure_ascii=False)}"
        
        # Solicitar confirmación si es necesario
        if not self._confirm_operation(function_name, details):
            return FunctionCallResult(
                success=False,
                result={"error": "Operación cancelada por el usuario"},
                function_name=function_name,
                arguments=arguments,
                execution_time=time.time() - start_time
            )
        
        # Ejecutar la función
        try:
            function = self.function_map[function_name]
            result = function(**arguments)
            success = "error" not in result
            
            logging.info(f"Función '{function_name}' ejecutada. Resultado: {json.dumps(result, ensure_ascii=False)}")
            
            return FunctionCallResult(
                success=success,
                result=result,
                function_name=function_name,
                arguments=arguments,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logging.error(f"Error al ejecutar '{function_name}': {str(e)}")
            logging.debug(traceback.format_exc())
            
            return FunctionCallResult(
                success=False,
                result={"error": f"Error al ejecutar la función: {str(e)}"},
                function_name=function_name,
                arguments=arguments,
                execution_time=time.time() - start_time
            )
    
    def process_user_input(self, user_input: str) -> str:
        """
        Procesa la entrada del usuario, decide si es un comando directo o si usar una función y genera una respuesta.
        
        Args:
            user_input: Texto ingresado por el usuario
            
        Returns:
            Respuesta del asistente o resultado del comando
        """
        # Comprobar si es un comando especial
        if self.chat_commands.is_command(user_input):
            return self.chat_commands.process_command(user_input)
        
        # Añadir el mensaje del usuario al historial
        self.messages.append({"role": "user", "content": user_input})
        
        # Primera llamada al modelo: decide si llamar a una función
        try:
            with console.status("[bold blue]Pensando...[/bold blue]", spinner="dots"):
                resp = self.llm.create_chat_completion(
                    messages=self.messages,
                    functions=AVAILABLE_FUNCTIONS,
                    function_call="auto",
                    temperature=self.config["TEMPERATURE"]["decision"]
                )
            
            # Actualizar contador de tokens
            if "usage" in resp:
                self.token_count += resp["usage"].get("total_tokens", 0)
            
            model_content = resp["choices"][0]["message"]["content"]
            
            # Extraer información de la llamada a función
            function_call = self._extract_function_call(model_content)
            
            if function_call:
                # Hay una llamada a función
                fn_name = function_call["name"]
                arguments = function_call["arguments"]
                
                logging.info(f"Llamada a función detectada: {fn_name} con argumentos: {arguments}")
                console.print(f"[dim]Ejecutando: {fn_name}[/dim]")
                
                # Ejecutar la función
                result = self._execute_function(fn_name, arguments)
                self.function_calls += 1
                
                if result.success:
                    console.print(f"[green]✓ {fn_name} completado en {result.execution_time:.2f}s[/green]")
                else:
                    console.print(f"[red]✗ {fn_name} falló: {result.result.get('error', 'Error desconocido')}[/red]")
                
                # Añadir la respuesta de la función al historial
                self.messages.append({
                    "role": "function",
                    "name": fn_name,
                    "content": json.dumps(result.result, ensure_ascii=False)
                })
                
                # Segunda llamada: el modelo genera la respuesta final
                with console.status("[bold blue]Generando respuesta...[/bold blue]", spinner="dots"):
                    resp2 = self.llm.create_chat_completion(
                        messages=self.messages,
                        temperature=self.config["TEMPERATURE"]["conversation"]
                    )
                
                # Actualizar contador de tokens
                if "usage" in resp2:
                    self.token_count += resp2["usage"].get("total_tokens", 0)
                
                assistant_msg = resp2["choices"][0]["message"]["content"].strip()
            else:
                # El modelo respondió directamente sin usar funciones
                assistant_msg = model_content.strip()
            
            # Añadir la respuesta del asistente al historial
            self.messages.append({"role": "assistant", "content": assistant_msg})
            
            return assistant_msg
            
        except Exception as e:
            logging.error(f"Error en procesamiento: {str(e)}")
            logging.debug(traceback.format_exc())
            return f"Lo siento, ocurrió un error en el procesamiento: {str(e)}"
    
    def run_chat_loop(self):
        """Ejecuta el bucle principal de conversación."""
        logging.info(f"Chat FS iniciado. Directorio base: {self.config['BASE_DIR']}")
        
        try:
            while True:
                # Obtener entrada del usuario
                if self.config["UI"]["show_timestamps"]:
                    timestamp = f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                else:
                    timestamp = ""
                
                user_color = self.config["UI"]["user_color"]
                prompt_text = f"{timestamp}[bold {user_color}]Tú:[/bold {user_color}] "
                
                user_input = Prompt.ask(prompt_text)
                user_input = user_input.strip()
                
                # Verificar comandos especiales
                if self.chat_commands and self.chat_commands.is_command(user_input):
                    try:
                        response = self.chat_commands.process_command(user_input)
                        
                        # Mostrar la respuesta con formato
                        if self.config["UI"]["show_timestamps"]:
                            timestamp = f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                        else:
                            timestamp = ""
                        
                        info_color = self.config["UI"]["info_color"]
                        console.print(f"\n{timestamp}[bold {info_color}]Sistema:[/bold {info_color}] {response}\n")
                        
                    except KeyboardInterrupt:
                        console.print("\n\n[bold green]¡Hasta luego![/bold green]")
                        break
                    
                    continue
                
                # Verificar comandos de salida
                if user_input.lower() in ("exit", "quit", "salir"):
                    console.print("\n[bold green]¡Hasta luego![/bold green]")
                    break
                
                # Procesar la entrada y obtener la respuesta
                assistant_response = self.process_user_input(user_input)
                
                # Mostrar la respuesta con formato
                if self.config["UI"]["show_timestamps"]:
                    timestamp = f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                else:
                    timestamp = ""
                
                assistant_color = self.config["UI"]["assistant_color"]
                console.print(f"\n{timestamp}[bold {assistant_color}]Asistente:[/bold {assistant_color}] {assistant_response}\n")
                
        except KeyboardInterrupt:
            console.print("\n\n[bold green]¡Hasta luego![/bold green]")
        except Exception as e:
            logging.error(f"Error en el bucle principal: {str(e)}")
            console.print(f"\n[bold red]Error en la aplicación: {str(e)}[/bold red]")
            console.print("[red]Consulta los logs para más detalles.[/red]")

    # ==============================================================================
    # IMPLEMENTACIÓN DE FUNCIONES DEL SISTEMA DE ARCHIVOS
    # ==============================================================================
    
    # === OPERACIONES DE LECTURA ===
    
    def list_directory(self, path: str) -> Dict[str, Any]:
        """
        Lista el contenido de un directorio, mostrando archivos y subdirectorios.
        
        Args:
            path: Ruta del directorio a listar
            
        Returns:
            Diccionario con los resultados: archivos y directorios encontrados
        """
        try:
            validated_path = self._validate_path(path)
            target_dir = self._get_absolute_path(validated_path)
            
            if not os.path.isdir(target_dir):
                return {"error": f"El directorio '{validated_path}' no existe."}
            
            # Obtener listado de archivos y directorios
            items = os.listdir(target_dir)
            
            # Separar en archivos y directorios
            files = []
            directories = []
            
            for item in items:
                item_path = os.path.join(target_dir, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            # Ordenar resultados
            files.sort()
            directories.sort()
            
            return {
                "result": {
                    "path": validated_path,
                    "files": files,
                    "directories": directories,
                    "total_files": len(files),
                    "total_directories": len(directories)
                }
            }
            
        except Exception as e:
            logging.error(f"Error al listar directorio '{path}': {str(e)}")
            return {"error": f"Error al listar directorio: {str(e)}"}
    
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Obtiene información detallada sobre un archivo.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Diccionario con información del archivo
        """
        try:
            validated_path = self._validate_path(path)
            file_path = self._get_absolute_path(validated_path)
            
            if not os.path.exists(file_path):
                return {"error": f"El archivo '{validated_path}' no existe."}
            
            if not os.path.isfile(file_path):
                return {"error": f"'{validated_path}' no es un archivo."}
            
            # Obtener estadísticas del archivo
            stats = os.stat(file_path)
            
            # Calcular hash del archivo (para archivos pequeños)
            file_hash = ""
            if stats.st_size < 10_000_000:  # 10MB
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    file_hash = "No disponible"
            
            # Determinar tipo de archivo
            file_type = "Desconocido"
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js']:
                file_type = "Texto"
            elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']:
                file_type = "Imagen"
            elif extension in ['.mp3', '.wav', '.ogg', '.flac', '.aac']:
                file_type = "Audio"
            elif extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                file_type = "Video"
            elif extension in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
                file_type = "Documento"
            elif extension in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                file_type = "Archivo comprimido"
            elif extension in ['.py', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb']:
                file_type = "Código fuente"
            
            return {
                "result": {
                    "name": os.path.basename(file_path),
                    "path": validated_path,
                    "size": stats.st_size,
                    "size_human": self._format_size(stats.st_size),
                    "created": datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    "accessed": datetime.fromtimestamp(stats.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
                    "is_readable": os.access(file_path, os.R_OK),
                    "is_writable": os.access(file_path, os.W_OK),
                    "is_executable": os.access(file_path, os.X_OK),
                    "file_type": file_type,
                    "extension": extension[1:] if extension else "",
                    "md5": file_hash
                }
            }
            
        except Exception as e:
            logging.error(f"Error al obtener información del archivo '{path}': {str(e)}")
            return {"error": f"Error al obtener información: {str(e)}"}
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Formatea un tamaño en bytes a una representación legible.
        
        Args:
            size_bytes: Tamaño en bytes
            
        Returns:
            Cadena formateada con unidades
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    # === OPERACIONES DE CREACIÓN ===
    
    def create_directory(self, path: str) -> Dict[str, str]:
        """
        Crea un directorio en la ruta especificada.
        
        Args:
            path: Ruta del directorio a crear
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_path = self._validate_path(path)
            target_dir = self._get_absolute_path(validated_path)
            
            os.makedirs(target_dir, exist_ok=True)
            logging.info(f"Directorio creado: {validated_path}")
            
            return {"result": f"Directorio '{validated_path}' creado exitosamente."}
            
        except Exception as e:
            logging.error(f"Error al crear directorio '{path}': {str(e)}")
            return {"error": f"Error al crear directorio: {str(e)}"}
    
    # === OPERACIONES DE COPIA ===
    
    def copy_file(self, source: str, destination: str) -> Dict[str, str]:
        """
        Copia un archivo de una ubicación a otra.
        
        Args:
            source: Ruta del archivo de origen
            destination: Ruta de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            validated_dst = self._validate_path(destination)
            
            src_path = self._get_absolute_path(validated_src)
            dst_path = self._get_absolute_path(validated_dst)
            
            # Verificar que el archivo de origen existe
            if not os.path.isfile(src_path):
                return {"error": f"El archivo '{validated_src}' no existe."}
            
            # Crear el directorio de destino si no existe
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copiar el archivo
            shutil.copy2(src_path, dst_path)
            logging.info(f"Archivo copiado: {validated_src} → {validated_dst}")
            
            return {"result": f"Archivo copiado de '{validated_src}' a '{validated_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al copiar archivo '{source}' a '{destination}': {str(e)}")
            return {"error": f"Error al copiar archivo: {str(e)}"}
    
    def copy_directory(self, source: str, destination: str) -> Dict[str, str]:
        """
        Copia un directorio completo de una ubicación a otra.
        
        Args:
            source: Ruta del directorio de origen
            destination: Ruta de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            validated_dst = self._validate_path(destination)
            
            src_path = self._get_absolute_path(validated_src)
            dst_path = self._get_absolute_path(validated_dst)
            
            # Verificar que el directorio de origen existe
            if not os.path.isdir(src_path):
                return {"error": f"El directorio '{validated_src}' no existe."}
            
            # Verificar si el directorio de destino ya existe
            if os.path.exists(dst_path):
                if not os.path.isdir(dst_path):
                    return {"error": f"El destino '{validated_dst}' ya existe y no es un directorio."}
            else:
                # Crear directorios padre si no existen
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copiar el directorio recursivamente
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            logging.info(f"Directorio copiado: {validated_src} → {validated_dst}")
            
            return {"result": f"Directorio copiado de '{validated_src}' a '{validated_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al copiar directorio '{source}' a '{destination}': {str(e)}")
            return {"error": f"Error al copiar directorio: {str(e)}"}
    
    def copy_multiple_files(self, files: List[str], destination: str) -> Dict[str, Any]:
        """
        Copia múltiples archivos a un directorio de destino.
        
        Args:
            files: Lista de rutas de archivos a copiar
            destination: Directorio de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            # Validar que no exceda el límite de operaciones por lote
            if len(files) > self.config["SECURITY"]["max_batch_size"]:
                return {"error": f"Demasiados archivos para copiar en un solo lote (máximo: {self.config['SECURITY']['max_batch_size']})."}
            
            validated_dst = self._validate_path(destination)
            dst_dir = self._get_absolute_path(validated_dst)
            
            # Verificar que el destino es un directorio y crearlo si no existe
            if os.path.exists(dst_dir) and not os.path.isdir(dst_dir):
                return {"error": f"El destino '{validated_dst}' existe y no es un directorio."}
            
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copiar cada archivo
            successful = []
            failed = []
            
            for file_path in files:
                try:
                    validated_src = self._validate_path(file_path)
                    src_path = self._get_absolute_path(validated_src)
                    
                    # Verificar que el archivo existe
                    if not os.path.isfile(src_path):
                        failed.append({"file": validated_src, "error": "No existe o no es un archivo"})
                        continue
                    
                    # Ruta de destino para este archivo
                    file_name = os.path.basename(src_path)
                    file_dst = os.path.join(dst_dir, file_name)
                    
                    # Copiar el archivo
                    shutil.copy2(src_path, file_dst)
                    successful.append(validated_src)
                    
                except Exception as e:
                    failed.append({"file": file_path, "error": str(e)})
            
            # Registrar resultado
            if failed:
                logging.warning(f"Copia múltiple parcial: {len(successful)} éxitos, {len(failed)} fallos")
                return {
                    "result": f"Se copiaron {len(successful)} de {len(files)} archivos a '{validated_dst}'.",
                    "successful": successful,
                    "failed": failed
                }
            else:
                logging.info(f"Copia múltiple completada: {len(successful)} archivos a {validated_dst}")
                return {"result": f"Se copiaron {len(successful)} archivos a '{validated_dst}' exitosamente."}
                
        except Exception as e:
            logging.error(f"Error al copiar múltiples archivos a '{destination}': {str(e)}")
            return {"error": f"Error al copiar archivos: {str(e)}"}
    
    # === OPERACIONES DE MOVIMIENTO ===
    
    def move_file(self, source: str, destination: str) -> Dict[str, str]:
        """
        Mueve un archivo de una ubicación a otra.
        
        Args:
            source: Ruta del archivo de origen
            destination: Ruta de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            validated_dst = self._validate_path(destination)
            
            src_path = self._get_absolute_path(validated_src)
            dst_path = self._get_absolute_path(validated_dst)
            
            # Verificar que el archivo de origen existe
            if not os.path.isfile(src_path):
                return {"error": f"El archivo '{validated_src}' no existe."}
            
            # Crear el directorio de destino si no existe
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Mover el archivo
            shutil.move(src_path, dst_path)
            logging.info(f"Archivo movido: {validated_src} → {validated_dst}")
            
            return {"result": f"Archivo movido de '{validated_src}' a '{validated_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al mover archivo '{source}' a '{destination}': {str(e)}")
            return {"error": f"Error al mover archivo: {str(e)}"}
    
    def move_directory(self, source: str, destination: str) -> Dict[str, str]:
        """
        Mueve un directorio completo de una ubicación a otra.
        
        Args:
            source: Ruta del directorio de origen
            destination: Ruta de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            validated_dst = self._validate_path(destination)
            
            src_path = self._get_absolute_path(validated_src)
            dst_path = self._get_absolute_path(validated_dst)
            
            # Verificar que el directorio de origen existe
            if not os.path.isdir(src_path):
                return {"error": f"El directorio '{validated_src}' no existe."}
            
            # Crear directorios padre si no existen
            parent_dir = os.path.dirname(dst_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            # Mover el directorio
            shutil.move(src_path, dst_path)
            logging.info(f"Directorio movido: {validated_src} → {validated_dst}")
            
            return {"result": f"Directorio movido de '{validated_src}' a '{validated_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al mover directorio '{source}' a '{destination}': {str(e)}")
            return {"error": f"Error al mover directorio: {str(e)}"}
    
    def move_multiple_files(self, files: List[str], destination: str) -> Dict[str, Any]:
        """
        Mueve múltiples archivos a un directorio de destino.
        
        Args:
            files: Lista de rutas de archivos a mover
            destination: Directorio de destino
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            # Validar que no exceda el límite de operaciones por lote
            if len(files) > self.config["SECURITY"]["max_batch_size"]:
                return {"error": f"Demasiados archivos para mover en un solo lote (máximo: {self.config['SECURITY']['max_batch_size']})."}
            
            validated_dst = self._validate_path(destination)
            dst_dir = self._get_absolute_path(validated_dst)
            
            # Verificar que el destino es un directorio y crearlo si no existe
            if os.path.exists(dst_dir) and not os.path.isdir(dst_dir):
                return {"error": f"El destino '{validated_dst}' existe y no es un directorio."}
            
            os.makedirs(dst_dir, exist_ok=True)
            
            # Mover cada archivo
            successful = []
            failed = []
            
            for file_path in files:
                try:
                    validated_src = self._validate_path(file_path)
                    src_path = self._get_absolute_path(validated_src)
                    
                    # Verificar que el archivo existe
                    if not os.path.isfile(src_path):
                        failed.append({"file": validated_src, "error": "No existe o no es un archivo"})
                        continue
                    
                    # Ruta de destino para este archivo
                    file_name = os.path.basename(src_path)
                    file_dst = os.path.join(dst_dir, file_name)
                    
                    # Mover el archivo
                    shutil.move(src_path, file_dst)
                    successful.append(validated_src)
                    
                except Exception as e:
                    failed.append({"file": file_path, "error": str(e)})
            
            # Registrar resultado
            if failed:
                logging.warning(f"Movimiento múltiple parcial: {len(successful)} éxitos, {len(failed)} fallos")
                return {
                    "result": f"Se movieron {len(successful)} de {len(files)} archivos a '{validated_dst}'.",
                    "successful": successful,
                    "failed": failed
                }
            else:
                logging.info(f"Movimiento múltiple completado: {len(successful)} archivos a {validated_dst}")
                return {"result": f"Se movieron {len(successful)} archivos a '{validated_dst}' exitosamente."}
                
        except Exception as e:
            logging.error(f"Error al mover múltiples archivos a '{destination}': {str(e)}")
            return {"error": f"Error al mover archivos: {str(e)}"}
    
    # === OPERACIONES DE RENOMBRADO ===
    
    def rename_file(self, source: str, new_name: str) -> Dict[str, str]:
        """
        Renombra un archivo manteniendo su ubicación.
        
        Args:
            source: Ruta actual del archivo
            new_name: Nuevo nombre para el archivo (sin ruta)
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            src_path = self._get_absolute_path(validated_src)
            
            # Verificar que el archivo existe
            if not os.path.isfile(src_path):
                return {"error": f"El archivo '{validated_src}' no existe."}
            
            # Validar el nuevo nombre (no debe contener separadores de ruta)
            if os.path.sep in new_name or (os.path.altsep and os.path.altsep in new_name):
                return {"error": f"El nuevo nombre '{new_name}' no debe contener separadores de ruta."}
            
            # Calcular la nueva ruta
            dst_dir = os.path.dirname(src_path)
            dst_path = os.path.join(dst_dir, new_name)
            
            # Renombrar el archivo
            os.rename(src_path, dst_path)
            
            # Calcular la nueva ruta relativa
            rel_dst = os.path.join(os.path.dirname(validated_src), new_name)
            logging.info(f"Archivo renombrado: {validated_src} → {rel_dst}")
            
            return {"result": f"Archivo renombrado de '{validated_src}' a '{rel_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al renombrar archivo '{source}' a '{new_name}': {str(e)}")
            return {"error": f"Error al renombrar archivo: {str(e)}"}
    
    def rename_directory(self, source: str, new_name: str) -> Dict[str, str]:
        """
        Renombra un directorio manteniendo su ubicación.
        
        Args:
            source: Ruta actual del directorio
            new_name: Nuevo nombre para el directorio (sin ruta)
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_src = self._validate_path(source)
            src_path = self._get_absolute_path(validated_src)
            
            # Verificar que el directorio existe
            if not os.path.isdir(src_path):
                return {"error": f"El directorio '{validated_src}' no existe."}
            
            # Validar el nuevo nombre (no debe contener separadores de ruta)
            if os.path.sep in new_name or (os.path.altsep and os.path.altsep in new_name):
                return {"error": f"El nuevo nombre '{new_name}' no debe contener separadores de ruta."}
            
            # Calcular la nueva ruta
            dst_dir = os.path.dirname(src_path)
            dst_path = os.path.join(dst_dir, new_name)
            
            # Renombrar el directorio
            os.rename(src_path, dst_path)
            
            # Calcular la nueva ruta relativa
            rel_dst = os.path.join(os.path.dirname(validated_src), new_name)
            logging.info(f"Directorio renombrado: {validated_src} → {rel_dst}")
            
            return {"result": f"Directorio renombrado de '{validated_src}' a '{rel_dst}'."}
            
        except Exception as e:
            logging.error(f"Error al renombrar directorio '{source}' a '{new_name}': {str(e)}")
            return {"error": f"Error al renombrar directorio: {str(e)}"}
    
    # === OPERACIONES DE ELIMINACIÓN ===
    
    def delete_file(self, path: str) -> Dict[str, str]:
        """
        Elimina un archivo del sistema de archivos.
        
        Args:
            path: Ruta del archivo a eliminar
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_path = self._validate_path(path)
            file_path = self._get_absolute_path(validated_path)
            
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                return {"error": f"El archivo '{validated_path}' no existe."}
            
            if not os.path.isfile(file_path):
                return {"error": f"'{validated_path}' no es un archivo."}
            
            # Eliminar el archivo
            os.remove(file_path)
            logging.info(f"Archivo eliminado: {validated_path}")
            
            return {"result": f"Archivo '{validated_path}' eliminado exitosamente."}
            
        except Exception as e:
            logging.error(f"Error al eliminar archivo '{path}': {str(e)}")
            return {"error": f"Error al eliminar archivo: {str(e)}"}
    
    def delete_directory(self, path: str) -> Dict[str, str]:
        """
        Elimina un directorio y todo su contenido.
        
        Args:
            path: Ruta del directorio a eliminar
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            validated_path = self._validate_path(path)
            dir_path = self._get_absolute_path(validated_path)
            
            # Verificar que el directorio existe
            if not os.path.exists(dir_path):
                return {"error": f"El directorio '{validated_path}' no existe."}
            
            if not os.path.isdir(dir_path):
                return {"error": f"'{validated_path}' no es un directorio."}
            
            # Eliminar el directorio recursivamente
            shutil.rmtree(dir_path)
            logging.info(f"Directorio eliminado: {validated_path}")
            
            return {"result": f"Directorio '{validated_path}' y todo su contenido eliminado exitosamente."}
            
        except Exception as e:
            logging.error(f"Error al eliminar directorio '{path}': {str(e)}")
            return {"error": f"Error al eliminar directorio: {str(e)}"}
    
    def delete_multiple_files(self, files: List[str]) -> Dict[str, Any]:
        """
        Elimina múltiples archivos del sistema de archivos.
        
        Args:
            files: Lista de rutas de archivos a eliminar
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            # Validar que no exceda el límite de operaciones por lote
            if len(files) > self.config["SECURITY"]["max_batch_size"]:
                return {"error": f"Demasiados archivos para eliminar en un solo lote (máximo: {self.config['SECURITY']['max_batch_size']})."}
            
            # Eliminar cada archivo
            successful = []
            failed = []
            
            for file_path in files:
                try:
                    validated_path = self._validate_path(file_path)
                    path = self._get_absolute_path(validated_path)
                    
                    # Verificar que el archivo existe
                    if not os.path.exists(path):
                        failed.append({"file": validated_path, "error": "No existe"})
                        continue
                    
                    if not os.path.isfile(path):
                        failed.append({"file": validated_path, "error": "No es un archivo"})
                        continue
                    
                    # Eliminar el archivo
                    os.remove(path)
                    successful.append(validated_path)
                    
                except Exception as e:
                    failed.append({"file": file_path, "error": str(e)})
            
            # Registrar resultado
            if failed:
                logging.warning(f"Eliminación múltiple parcial: {len(successful)} éxitos, {len(failed)} fallos")
                return {
                    "result": f"Se eliminaron {len(successful)} de {len(files)} archivos.",
                    "successful": successful,
                    "failed": failed
                }
            else:
                logging.info(f"Eliminación múltiple completada: {len(successful)} archivos")
                return {"result": f"Se eliminaron {len(successful)} archivos exitosamente."}
                
        except Exception as e:
            logging.error(f"Error al eliminar múltiples archivos: {str(e)}")
            return {"error": f"Error al eliminar archivos: {str(e)}"}


# ==============================================================================
# FUNCIONES DE ENTRADA
# ==============================================================================

def load_config():
    """Carga la configuración guardada o usa valores predeterminados."""
    config = CONFIG.copy()
    
    # Directorio de configuración
    config_dir = config["CONFIG_DIR"]
    os.makedirs(config_dir, exist_ok=True)
    
    # Archivo de configuración
    config_file = os.path.join(config_dir, "config.json")
    
    # Variable para indicar si es primera ejecución
    is_first_run = not os.path.exists(config_file)
    
    # Intentar cargar configuración guardada
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                
            # Actualizar configuración con los valores guardados
            for key, value in saved_config.items():
                if key in config:
                    config[key] = value
        except Exception as e:
            console.print(f"[yellow]Advertencia: No se pudo cargar la configuración guardada: {str(e)}[/yellow]")
    
    # Agregar flag de primera ejecución
    config["IS_FIRST_RUN"] = is_first_run
    
    return config

def check_and_install_dependencies():
    """Verifica las dependencias necesarias y guía al usuario para instalarlas."""
    console.print(Panel.fit(
        "[bold blue]Verificación de Dependencias[/bold blue]\n"
        "Asegurando que todas las bibliotecas necesarias estén instaladas.",
        border_style="blue"
    ))
    
    # Lista de dependencias principales
    dependencies = [
        {"name": "llama-cpp-python", "package": "llama-cpp-python", "import_name": "llama_cpp"},
        {"name": "Rich", "package": "rich", "import_name": "rich"},
    ]
    
    missing_deps = []
    
    # Verificar cada dependencia
    for dep in dependencies:
        with console.status(f"Verificando {dep['name']}..."):
            try:
                importlib.import_module(dep["import_name"])
                console.print(f"[green]✓ {dep['name']} está instalado[/green]")
            except ImportError:
                console.print(f"[yellow]⚠ {dep['name']} no está instalado[/yellow]")
                missing_deps.append(dep)
    
    # Instalar dependencias faltantes
    if missing_deps:
        console.print("\n[bold]Se requiere instalar las siguientes dependencias:[/bold]")
        for dep in missing_deps:
            console.print(f"- {dep['name']} ({dep['package']})")
        
        if Confirm.ask("¿Instalar ahora?"):
            for dep in missing_deps:
                with console.status(f"Instalando {dep['name']}..."):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", dep["package"]])
                        console.print(f"[green]✓ {dep['name']} instalado correctamente[/green]")
                    except Exception as e:
                        console.print(f"[red]✗ Error al instalar {dep['name']}: {str(e)}[/red]")
                        console.print("[yellow]Puedes intentar instalarlo manualmente con:[/yellow]")
                        console.print(f"[dim]pip install {dep['package']}[/dim]")
                        
                        if dep["import_name"] == "llama_cpp":
                            console.print("\n[yellow]Nota: llama-cpp-python puede requerir opciones especiales de instalación[/yellow]")
                            console.print("[yellow]para aprovechar aceleración GPU. Visita: https://github.com/abetlen/llama-cpp-python[/yellow]")
                        
                        if not Confirm.ask("¿Continuar con la verificación?"):
                            console.print("[red]Instalación de dependencias interrumpida. El programa puede no funcionar correctamente.[/red]")
                            return False
                        
        else:
            console.print("[yellow]Instalación pospuesta. El programa puede no funcionar correctamente.[/yellow]")
            return False
    
    return True

def check_and_download_model(config):
    """Verifica si el modelo existe y guía al usuario para descargarlo si es necesario."""
    console.print(Panel.fit(
        "[bold blue]Verificación del Modelo[/bold blue]\n"
        "Comprobando si el modelo de lenguaje está disponible.",
        border_style="blue"
    ))
    
    model_path = config["MODEL_PATH"]
    
    # Verificar si el modelo existe
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"[green]✓ Modelo encontrado:[/green] {model_path} ({model_size_mb:.1f} MB)")
        return True
    
    console.print(f"[yellow]⚠ El modelo no se encuentra en la ruta especificada:[/yellow] {model_path}")
    console.print("\n[bold]Opciones disponibles:[/bold]")
    console.print("1) Especificar una ruta alternativa")
    console.print("2) Descargar el modelo (requiere conexión a Internet)")
    console.print("3) Salir y obtener el modelo manualmente")
    
    choice = Prompt.ask("Selecciona una opción", choices=["1", "2", "3"], default="2")
    
    if choice == "1":
        new_path = Prompt.ask("Introduce la ruta completa al archivo del modelo")
        if os.path.exists(new_path):
            config["MODEL_PATH"] = new_path
            console.print(f"[green]✓ Usando modelo en:[/green] {new_path}")
            return True
        else:
            console.print(f"[red]✗ No se encontró el archivo en:[/red] {new_path}")
            return check_and_download_model(config)
            
    elif choice == "2":
        model_url = "https://huggingface.co/tensorblock/Meta-Llama-3-8B-Instruct-function-calling-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-function-calling-Q4_K_M.gguf?download=true"
        download_dir = os.path.dirname(model_path) or "."
        
        console.print(f"[yellow]Se descargará el modelo desde Hugging Face:[/yellow]")
        console.print(f"[dim]{model_url}[/dim]")
        console.print(f"[yellow]El archivo tiene aproximadamente 5GB y puede tomar tiempo dependiendo de tu conexión.[/yellow]")
        
        if Confirm.ask("¿Proceder con la descarga?"):
            try:
                # Verificamos si wget o curl están disponibles
                download_command = None
                try:
                    subprocess.check_call(["wget", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    download_command = ["wget", "-O", model_path, model_url]
                except:
                    try:
                        subprocess.check_call(["curl", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        download_command = ["curl", "-L", "-o", model_path, model_url]
                    except:
                        console.print("[yellow]No se encontró wget ni curl. Intentando descarga con Python...[/yellow]")
                
                if download_command:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("[yellow]Descargando modelo...", total=None)
                        subprocess.check_call(download_command)
                        progress.update(task, completed=True)
                else:
                    # Descarga con Python
                    import urllib.request
                    
                    # Función para mostrar progreso
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        percent = min(100, downloaded * 100 / total_size)
                        console.print(f"Progreso: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end="\r")
                    
                    with console.status("[yellow]Descargando modelo... (esto puede tomar tiempo)[/yellow]"):
                        urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
                
                if os.path.exists(model_path):
                    console.print(f"[green]✓ Modelo descargado exitosamente en:[/green] {model_path}")
                    return True
                else:
                    console.print("[red]✗ La descarga parece haber fallado.[/red]")
            except Exception as e:
                console.print(f"[red]✗ Error durante la descarga: {str(e)}[/red]")
            
            console.print("\n[yellow]Recomendaciones:[/yellow]")
            console.print("1. Descarga manualmente el modelo desde: https://huggingface.co/tensorblock/Meta-Llama-3-8B-Instruct-function-calling-GGUF/blob/main/Meta-Llama-3-8B-Instruct-function-calling-Q4_K_M.gguf")
            console.print(f"2. Coloca el archivo en: {model_path}")
            console.print("3. Ejecuta nuevamente este script")
            
            if Confirm.ask("¿Intentar especificar otra ruta?"):
                return check_and_download_model(config)
            else:
                return False
        else:
            console.print("[yellow]Descarga cancelada.[/yellow]")
            return False
            
    else:  # choice == "3"
        console.print("\n[yellow]Instrucciones para obtener el modelo manualmente:[/yellow]")
        console.print("1. Descarga el modelo desde: https://huggingface.co/tensorblock/Meta-Llama-3-8B-Instruct-function-calling-GGUF/blob/main/Meta-Llama-3-8B-Instruct-function-calling-Q4_K_M.gguf")
        console.print(f"2. Coloca el archivo en: {model_path}")
        console.print("3. Ejecuta nuevamente este script")
        return False
    
    return False

def configure_base_directory(config):
    """Permite al usuario configurar el directorio base donde operará el asistente."""
    console.print(Panel.fit(
        "[bold blue]Configuración del Directorio Base[/bold blue]\n"
        "Aquí puedes especificar el directorio donde el asistente realizará las operaciones.",
        border_style="blue"
    ))
    
    # Verificar si ya existe un directorio configurado
    current_dir = config.get("BASE_DIR")
    if current_dir and os.path.exists(current_dir):
        console.print(f"[yellow]Directorio actual:[/yellow] {current_dir}")
        if Confirm.ask("¿Deseas mantener este directorio?"):
            return current_dir
    
    # Opciones de configuración
    console.print("\n[bold]Opciones disponibles:[/bold]")
    console.print("1) Usar el directorio actual")
    console.print("2) Especificar una ruta absoluta")
    console.print("3) Crear un nuevo directorio")
    
    choice = Prompt.ask("Selecciona una opción", choices=["1", "2", "3"], default="1")
    
    if choice == "1":
        base_dir = os.path.abspath(os.getcwd())
    elif choice == "2":
        while True:
            path = Prompt.ask("Introduce la ruta absoluta")
            base_dir = os.path.abspath(path)
            if os.path.exists(base_dir):
                if not Confirm.ask(f"El directorio '{base_dir}' existe. ¿Deseas usarlo?"):
                    continue
                break
            else:
                if Confirm.ask(f"El directorio '{base_dir}' no existe. ¿Deseas crearlo?"):
                    try:
                        os.makedirs(base_dir, exist_ok=True)
                        break
                    except Exception as e:
                        console.print(f"[red]Error al crear el directorio: {str(e)}[/red]")
    else:  # choice == "3"
        parent_dir = Prompt.ask("Directorio padre", default=os.getcwd())
        folder_name = Prompt.ask("Nombre del nuevo directorio")
        base_dir = os.path.join(os.path.abspath(parent_dir), folder_name)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            console.print(f"[red]Error al crear el directorio: {str(e)}[/red]")
            return configure_base_directory(config)
    
    console.print(f"[green]✓ Directorio base configurado:[/green] {base_dir}")
    
    # Guardar configuración para futuras ejecuciones
    if Confirm.ask("¿Guardar esta configuración para futuras ejecuciones?"):
        config_dir = config.get("CONFIG_DIR")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.json")
        
        try:
            # Cargar configuración existente si existe
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
            else:
                saved_config = {}
            
            # Actualizar directorio base
            saved_config["BASE_DIR"] = base_dir
            
            # Guardar configuración
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(saved_config, f, indent=2)
                
            console.print("[green]Configuración guardada exitosamente.[/green]")
        except Exception as e:
            console.print(f"[yellow]No se pudo guardar la configuración: {str(e)}[/yellow]")
    
    return base_dir

def parse_arguments():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="FileSystem MCP - Asistente inteligente para operaciones del sistema de archivos")
    
    parser.add_argument("--base-dir", type=str, help="Directorio base para operaciones")
    parser.add_argument("--model", type=str, help="Ruta al modelo Llama")
    parser.add_argument("--permission-level", type=str, choices=[level.name for level in PermissionLevel], 
                       help="Nivel de permisos para operaciones")
    parser.add_argument("--confirm-all", action="store_true", help="Confirmar todas las operaciones")
    
    return parser.parse_args()

def main():
    """Función principal para ejecutar el asistente MCP."""
    try:
        # Mostrar banner de inicio
        console.print(Panel.fit(
            "[bold green]FileSystem MCP - Asistente Inteligente[/bold green]\n"
            "[yellow]Sistema para operaciones de archivos mediante lenguaje natural[/yellow]",
            title="¡Bienvenido!",
            border_style="blue"
        ))
        
        # Cargar configuración inicial
        config = load_config()
        
        # Procesar argumentos de línea de comandos
        args = parse_arguments()
        
        # Actualizar configuración con argumentos
        if args.base_dir:
            config["BASE_DIR"] = os.path.abspath(args.base_dir)
        if args.model:
            config["MODEL_PATH"] = args.model
        if args.confirm_all:
            config["SECURITY"]["confirm_all"] = True
        
        # Verificar y configurar el directorio base
        # Siempre configurar en la primera ejecución o si se proporciona un directorio inválido
        if config["IS_FIRST_RUN"] or not config.get("BASE_DIR") or not os.path.exists(config.get("BASE_DIR", "")):
            console.print("[bold blue]Primera ejecución detectada o directorio base no configurado.[/bold blue]")
            config["BASE_DIR"] = configure_base_directory(config)
        
        # Verificar e instalar dependencias
        if not check_and_install_dependencies():
            console.print("[yellow]Advertencia: No todas las dependencias están instaladas.[/yellow]")
            if not Confirm.ask("¿Continuar de todos modos?"):
                console.print("[red]Programa finalizado.[/red]")
                return 1
        
        # Verificar y descargar el modelo si es necesario
        if not check_and_download_model(config):
            console.print("[red]No se pudo continuar sin el modelo de lenguaje.[/red]")
            return 1
        
        # Inicializar el asistente
        assistant = FileSystemMCP(config)
        
        # Inicializar sistema de comandos
        assistant.chat_commands = ChatCommands(assistant)
        
        # Establecer nivel de permisos si se especificó
        if args.permission_level:
            assistant.permissions.set_permission_level(PermissionLevel[args.permission_level])
        
        # Iniciar el bucle de chat
        assistant.run_chat_loop()
        
    except KeyboardInterrupt:
        console.print("\n\n[bold green]¡Hasta luego![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error crítico: {str(e)}[/bold red]")
        console.print_exception()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
