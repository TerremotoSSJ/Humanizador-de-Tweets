from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "./modelo_entrenado"


def build_prompt(tweet: str) -> str:
    # Debe coincidir con la plantilla usada en entrenamiento.py
    return (
        "Reescribe el siguiente tweet de futbol en un estilo natural no formal.\n"
        f"Tweet: {tweet.strip()}\n"
        "Respuesta:\n"
    )


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Igual que en entrenamiento: cuantizacion 4-bit
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, tweet: str, max_new_tokens: int = 80) -> str:
    prompt = build_prompt(tweet)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7, #
            top_p=0.9, 
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extrae solo la parte generada después de "Respuesta:"
    if "Respuesta:\n" in text:
        return text.split("Respuesta:\n", 1)[1].strip()
    return text.strip()


def main():
    model, tokenizer = load_model_and_tokenizer()

    test_tweets = [
        "No me gusto para nada la respuesta de mi amigo, es un maleducado",
        "Ayer fui al cine y la película estuvo increíble, me encantó cada minuto",
        "La verdad es que no tengo ni la más mínima idea de cómo resolver este problema, estoy perdido",
        "No voy a ir a la fiesta, no me apetece nada y además estoy cansado",
    ]

    print("=== TEST RAPIDO MODELO FINE-TUNED ===")
    for i, tweet in enumerate(test_tweets, start=1):
        output = generate(model, tokenizer, tweet)
        print(f"\n[{i}] Input : {tweet}")
        print(f"[{i}] Output: {output}")


if __name__ == "__main__":
    main()