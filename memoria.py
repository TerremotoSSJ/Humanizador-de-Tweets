import json
from pathlib import Path
from datetime import datetime
from typing import Union
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

def guardar_historial(nuevo_mensaje: Union[SystemMessage, HumanMessage, AIMessage, ToolMessage], archivo: Path, thread_id: str) -> None:
    """Quiero guardarlo de manera {
        id (siendo id el thread del usuario):{
            numero_mensajes: int,
            mensajes: list[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]   
        }
    }"""
    if isinstance(nuevo_mensaje, ToolMessage):
        mensaje_dict = {
        "type": type(nuevo_mensaje).__name__,
        "content": nuevo_mensaje.content,
        "tool_call_id": nuevo_mensaje.tool_call_id  # ← Guardar también esto
    }
    else:
        mensaje_dict = {
        "type": type(nuevo_mensaje).__name__,
        "content": nuevo_mensaje.content
        }
    if archivo.exists():
        with open(archivo, "r", encoding="utf-8") as f:
            historial=json.load(f)
    else:
        historial={}

    if thread_id not in historial:
        historial[thread_id] = {
            "numero_mensajes": 0,
            "mensajes": []
        }

    historial[thread_id]["mensajes"].append(mensaje_dict)
    historial[thread_id]["numero_mensajes"] += 1
    if(len(historial[thread_id]["mensajes"]) > 20): # Limitar a los últimos 20 mensajes por thread para no ocupar demasiado espacio
        historial[thread_id]["mensajes"] = historial[thread_id]["mensajes"][-20:]
        historial[thread_id]["numero_mensajes"] = 20

    

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)

def cargar_historial(archivo: Path, thread_id: str) -> list[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]:
    """Cargar el historial de mensajes de un thread específico, devolviendo una lista de objetos de mensaje en el formato original"""
    if not archivo.exists():
        # Crear un archivo vacío si no existe
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return []
    
    with open(archivo, "r", encoding="utf-8") as f:
        historial=json.load(f)

    if thread_id not in historial:
        return []

    mensajes = []
    for mensaje_dict in historial[thread_id]["mensajes"]:
        tipo = mensaje_dict["type"]
        contenido = mensaje_dict["content"]
        if tipo == "SystemMessage":
            mensajes.append(SystemMessage(content=contenido))
        elif tipo == "HumanMessage":
            mensajes.append(HumanMessage(content=contenido))
        elif tipo == "AIMessage":
            mensajes.append(AIMessage(content=contenido))
        elif tipo == "ToolMessage":
            mensajes.append(ToolMessage(content=contenido, tool_call_id=mensaje_dict["tool_call_id"]))

    return mensajes
