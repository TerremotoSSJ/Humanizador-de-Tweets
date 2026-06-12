import time
import os
from openai import OpenAI
from openai.types.batch import Batch
from openai.types import FileObject
import json


# Parametros
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
model="gpt-5-nano" 
file="tweets_filtrados.jsonl" # Archivo de entrada con los tweets a traducir
file_output="tweets_traducidos.jsonl"

prompt="""Eres un asistente que reescribe tweets a un tono formal y profesional.

REGLAS:
1. Mantén el significado original exacto
2. No añadas ni elimines menciones (@usuario), hashtags (#) o enlaces
3. No uses emojis, abreviaturas ni expresiones coloquiales (ej: "xD", "jajaja", "tío", "genial")
4. Si el tweet es ofensivo, reescribelo de forma neutral sin perder el mensaje principal

EJEMPLOS:
Tweet original: "jajaja esto es increíble tío! 🔥"
Tweet formal: "Esto es increíble."

Tweet original: "@usuario eres un inútil"
Tweet formal: "@usuario los resultados no son satisfactorios."

Tweet original: "q pasa gente? hoy me toca madrugar 😴"
Tweet formal: "Buenos días. Hoy debo madrugar."

"""

def crear_cuerpo_peticion(tweet: str) -> dict:
    return {
        "model":model,
        "messages":[
            {"role":"system","content":prompt},
            {"role":"user","content":f"TWEET ORIGINAL: {tweet.strip()}"}
        ]
    }

def crear_peticion(tweet: str, tweet_id: str) -> dict:
    """Seguimos el formato de peticiones por lotes de OpenAI, con un ID único para cada tweet"""
    return {
        "custom_id": tweet_id,
        "method":"POST",
        "url":"/v1/chat/completions",
        "body": crear_cuerpo_peticion(tweet)
    }

def crear_archivo_peticiones(file: str,batch_id: str) -> str:
    """Creamos un nuevo archivo de peticiones a partir del archivo de tweets filtrados, con un ID de lote único"""
    file_output=f"peticiones_{batch_id}.jsonl"
    with open(file, "r", encoding="utf-8") as f:
        with open(file_output, "w", encoding="utf-8") as f_out:
            for idx,line in enumerate(f):
                if line.strip(): 
                    peticion=crear_peticion(json.loads(line)["tweet"], f"{batch_id}_{idx}") # Generamos un ID único para cada petición
                    json.dump(peticion, f_out, ensure_ascii=False)
                    f_out.write("\n")
    return file_output # Devolvemos el nombre del archivo generado


def subir_archivo_peticiones(batch_id: str) -> FileObject:
    """Subimos el archivo de peticiones a OpenAI y devolvemos el objeto File resultante"""
    batch_input_file=client.files.create(
        file=open(f"peticiones_{batch_id}.jsonl", "rb"),
        purpose="batch"
    )
    print(f"Archivo de peticiones subido con ID: {batch_input_file.id}")
    return batch_input_file

def crear_batch(batch_id: str) -> Batch:
    """Subimos el archivo de peticiones y creamos un Batch para procesarlas"""
    batch=subir_archivo_peticiones(batch_id)
    new=client.batches.create(
        input_file_id=batch.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return new # Devolvemos el objeto Batch creado

def estado_batch(batch_id: str) -> str:
    """Consulta el estado del Batch usando su ID"""
    batch=client.batches.retrieve(batch_id)
    return batch.status

def descargar_resultados(output_file_id: str) -> list[str]:
    """Descarga el archivo de resultados usando su ID y extrae las traducciones"""
    respuestas={}
    file_response=client.files.content(output_file_id) # Descargamos el archivo de resultados usando su ID
    lineas=file_response.text
    
    #Cada linea es un archivo json con la respuesta de una petición individual
    for linea in lineas.split("\n"):
        if not linea.strip():
            continue
        
        #Parseamos el json
        respuesta=json.loads(linea)

        #Extraemos la traduccion de la respuesta
        traduccion=respuesta["response"]["body"]["choices"][0]["message"]["content"].strip()
        respuestas[respuesta["custom_id"]] = traduccion
    return respuestas

def main() -> None:
    batch_id="lote1" # ID único para este lote de peticiones
    crear_archivo_peticiones(file, batch_id) # Creamos el archivo de peticiones a partir del archivo de tweets
    batch=crear_batch(batch_id) # Subimos el archivo y creamos el Batch
    print(f"Batch creado con ID: {batch.id}. Esperando resultados...")
    try:
        while True:
            batch=client.batches.retrieve(batch.id) # Actualizamos el estado del Batch
            status=estado_batch(batch.id) # Consultamos el estado del Batch periódicamente hasta que se complete
            print(f"Estado del batch: {status}")
            if status=="completed":

                print("Batch completado. Descargando resultados...")
                respuestas=descargar_resultados(batch.output_file_id) # Descargamos las respuestas usando el ID del archivo de salida del Batch
                 
                # Leer tweets originales para juntar con las traducciones, usando el mismo orden que el archivo de peticiones
                tweets_originales = []
                with open(file, "r", encoding="utf-8") as f_in:
                    for linea in f_in:
                        if linea.strip():
                            tweets_originales.append(json.loads(linea)["tweet"])

                # Guardar traducciones en el formato deseado, como no aseguran orden, usamos el custom_id para mapear cada traduccion con su tweet original
                with open(file_output, "w", encoding="utf-8") as f_out:
                    for i, original in enumerate(tweets_originales):
                        nombre_id = f"{batch_id}_{i}"
                        traduccion = respuestas.get(nombre_id, "Error: No se encontró traducción") # Si no se encuentra traducción para este ID, ponemos un mensaje de error
                        resultado = {
                            "original": original,
                            "traduccion": traduccion
                        }
                        f_out.write(json.dumps(resultado, ensure_ascii=False) + "\n")
                
                print(f"✅ {len(respuestas)} traducciones guardadas en {file_output}")
                break
            
            #Si falla
            elif status in ["failed", "error","expired"]:
                raise Exception(f"Batch terminó con estado: {status}")
            

            print(f"""Peticiones completadas: {batch.request_counts.completed}\n
                  Peticiones con error: {batch.request_counts.failed}\n
                  Peticiones totales: {batch.request_counts.total}\n
                  Esperando 30 segundos antes de volver a consultar el estado...\n\n""")
            time.sleep(30) # Esperamos 30 segundos antes de volver a consultar el estado
    except Exception as e:
        print(f"Ha sucedido un error, en el estado: {e}")

if __name__ == "__main__":    main()