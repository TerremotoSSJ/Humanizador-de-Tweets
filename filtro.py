# paso_1_filtrar_tweets.py
import os
import sys

# Evita un crash al finalizar Python por hilos de descarga (hf_transfer).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from datasets import load_dataset
import re
import json
from tqdm import tqdm
from pathlib import Path

#Parametros
tweets_maximos=10000 # Número máximo de tweets a guardar (ajustable)
porcentaje_con_menciones=0.3 # Porcentaje de tweets que pueden contener menciones (ajustable)
archivo_entrada="HateCorpus.txt" # Archivo de entrada con los tweets originales
archivo_salida="tweets_filtrados.jsonl" # Archivo de salida para los tweets filtrados
fila_empezar=0 # Fila desde la cual empezar a procesar (ajustable)
mencion_regex = re.compile(r'@\w+')
url_regex = re.compile(r'http\S+')

# 1. Cargar dataset (empieza con una parte)
print("Cargando dataset")



def tiene_menciones(texto):
    """Devuelve True si el texto tiene menciones (@usuario)"""
    return mencion_regex.search(texto) is not None

def limpiar_tweet(texto):
    """Limpia URLs"""
    texto = url_regex.sub('', texto)
    return texto.strip()

def leer_tweet(tweet) -> str:
    """Extrae el texto de la linea del dataset"""
    texto = tweet.split(";")[2] # El texto del tweet está en la tercera parte
    return limpiar_tweet(texto)

def primer_filtro(max_tweets,archivo_entrada, archivo_salida):
    if fila_empezar > 0:
        dataset=dataset.skip(fila_empezar) # Saltamos las primeras filas ya procesadas
    max_tweets_con_menciones = int(max_tweets * porcentaje_con_menciones)
    max_tweets_sin_menciones = max_tweets - max_tweets_con_menciones
    tweets_con_menciones = 0
    tweets_sin_menciones = 0
    with open(archivo_entrada, 'r', encoding='utf-8') as f_in:
        with open(archivo_salida, 'w', encoding='utf-8') as f_out:
            for tweet in tqdm(f_in, desc="Filtrando tweets"):
                texto = leer_tweet(tweet)
                es_mencion=tiene_menciones(texto)
                escribir=False
                if es_mencion:
                    if tweets_con_menciones < max_tweets_con_menciones:
                        escribir=True
                        tweets_con_menciones += 1
                else:
                    if tweets_sin_menciones < max_tweets_sin_menciones:
                        escribir=True
                        tweets_sin_menciones += 1
                if escribir:
                    json.dump({"tweet": texto}, f_out, ensure_ascii=False)
                    f_out.write('\n')
                if tweets_con_menciones >= max_tweets_con_menciones and tweets_sin_menciones >= max_tweets_sin_menciones:
                    break
    print(f"Filtrado completo. Tweets con menciones: {tweets_con_menciones}, sin menciones: {tweets_sin_menciones}. Total: {tweets_con_menciones + tweets_sin_menciones}")



    




if __name__ == "__main__":
    if os.path.exists(archivo_salida):
        os.remove(archivo_salida)
    primer_filtro(tweets_maximos, archivo_entrada, archivo_salida)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

