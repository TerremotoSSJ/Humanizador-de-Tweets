from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

dataset=load_dataset("json",data_files="tweets_traducidos.jsonl")["train"]


# Aplicamos cuantizacion de 4 bits usando doble cuantizacion para reducir el tamaño del modelo
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                  bnb_4bit_use_double_quant=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)
training_args = SFTConfig(
    output_dir="./modelo_entrenado",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    bf16 = torch.cuda.is_bf16_supported(),
    fp16=False,
    logging_steps=10, 
    save_steps=50, 
    report_to="none",
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_length=128,
    save_total_limit=2,
    packing=True, # Permite agrupar múltiples ejemplos en un solo lote para aprovechar mejor la capacidad de tokens del modelo
)
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", quantization_config=quant_config, device_map="auto")


def normalize_text(text: str) -> str:
    # Mantenemos un formato natural, pero eliminamos espacios extra y normalizamos saltos de línea
    return " ".join(str(text).strip().split())

# Cambiamos al formato prompt completion para SFTTrainer
def preprocess_function(example):
    tweet = normalize_text(example["tweet"])
    original_tweet = normalize_text(example["original_tweet"]) 

    prompt = (
        "Reescribe el siguiente tweet en un estilo natural no formal.\n"
        f"Tweet: {tweet}\n"
        "Respuesta:\n"
    )
    completion = f"{original_tweet}\n"
    return {"prompt": prompt, "completion": completion}


dataset=dataset.map(preprocess_function,remove_columns=dataset.column_names) # Aplicamos el formato conversacional de SFTTrainer
dataset=dataset.train_test_split(test_size=0.1) # 90% train, 10% test

tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Asegura que el padding se añada al final de la secuencia

trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer, # En warnings me pide un tokenizer
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=training_args
)
trainer.train()
trainer.save_model("./modelo_entrenado")