import os
from openai import OpenAI
import json

# Parametros
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
model="gpt-5-nano" 
batch_size=30
max_tokens=2048 # Maximo de tokens por lote
max_retries=4 # Número máximo de reintentos para obtener una salida válida
file="tweets_filtrados.jsonl" # Archivo de entrada con los tweets a traducir
file_output="tweets_traducidos.jsonl"
prompt="Rescribe el siguiente tweet en un estilo formal, breve y educado. Devuelve ÚNICAMENTE un array JSON con los tweets reescritos, en el mismo orden y cantidad:\n"

def tweet_amount(file: str) -> int:
    """
    file: path to the file to count tweets from
    returns the number of tweets in the file
    """
    count = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "tweet" in data:
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def obtain_tweets(row: int, amount: int) -> list[str]:
    """
    row: number of the row to start from
    amount: number of tweets to extract
    returns a list of tweets
    """
    file_path = os.path.join(os.getcwd(), file)
    tweets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < row:
                continue
            if len(tweets) >= amount:
                break
            try:
                data = json.loads(line)
                if "tweet" in data:
                    tweets.append(data["tweet"])
            except json.JSONDecodeError:
                continue
    return tweets

def normalize_output(output: list, tweets: list[str]) -> list[str]:
    """
    Garantiza misma longitud y orden que los tweets de entrada.
    Si faltan elementos o vienen vacíos, usa el tweet original como respaldo.
    """
    expected = len(tweets)
    normalized: list[str] = []

    for i in range(min(len(output), expected)):
        value = output[i]
        if isinstance(value, str) and value.strip():
            normalized.append(value)
        else:
            normalized.append(tweets[i])

    while len(normalized) < expected:
        normalized.append(tweets[len(normalized)])

    return normalized


def tweets_translation(tweets: list[str], prompt: str,model: str,max_tokens: int) -> list[str]:
    """
    tweets: list of tweets to translate
    prompt: the prompt to use for translation
    model: the model to use for translation
    max_tokens: the maximum number of tokens to generate
    returns a list of translated tweets
    """
    expected = len(tweets)
    for attempt in range(1, max_retries + 1):
        strong_prompt = (
            prompt
            + f"\nIMPORTANTE: Debes devolver EXACTAMENTE {expected} elementos en el array JSON. "
            + "No agregues explicación, no omitas elementos, no cambies el orden."
        )
        response=client.responses.create(
            model=model,
            input=strong_prompt +"\n"+ json.dumps(tweets, ensure_ascii=False)
        )
        try:
            output = json.loads(response.output_text)
            if isinstance(output, list) and len(output) == expected:
                return normalize_output(output, tweets)

            size = len(output) if isinstance(output, list) else "formato inválido"
            print(
                f"⚠️ Intento {attempt}/{max_retries}: salida inválida ({size}), "
                + f"esperados {expected}. Reintentando..."
            )
        except json.JSONDecodeError:
            print(f"⚠️ Intento {attempt}/{max_retries}: JSON inválido. Reintentando...")

    print(
        f"⚠️ Se alcanzó el máximo de reintentos. "
        + f"Se usarán los tweets originales para completar hasta {expected}."
    )
    return normalize_output([], tweets)
    
    
def main_logic(index: int = 0) -> None:
    max_tweets = tweet_amount(file)
    for i in range(index, max_tweets, batch_size):
        print(f"Procesando tweets {i} a {min(i + batch_size, max_tweets)} de {max_tweets}")
        tweets = obtain_tweets(i, batch_size)
        translated_tweets = tweets_translation(tweets, prompt, model, max_tokens)
        if len(translated_tweets) != len(tweets):
            print(
                f"⚠️ Ajuste de seguridad: salida {len(translated_tweets)}, "
                + f"entrada {len(tweets)}. Corrigiendo con respaldo local."
            )
            translated_tweets = normalize_output(translated_tweets, tweets)
        with open(file_output, "a", encoding="utf-8") as f_out:
            for cont in range(len(translated_tweets)):
                json.dump({"tweet": translated_tweets[cont],"original_tweet": tweets[cont]}, f_out, ensure_ascii=False)
                f_out.write("\n")

if __name__ == "__main__":
    input_index = input("Ingrese el índice de inicio (0 para comenzar desde el principio): ")
    try:
        index = int(input_index)
        if index < 0:
            raise ValueError
    except ValueError:
        print("Índice inválido. Se usará 0 por defecto.")
        index = 4171
    main_logic(index)
