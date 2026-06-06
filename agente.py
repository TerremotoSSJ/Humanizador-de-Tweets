from functools import lru_cache
from operator import itemgetter
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
from langchain.tools import tool
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



#Modelo y manejo del humanizador de tweets y memoria
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = Path("./modelo_entrenado")

#Memoria
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

store={}

class ChatMessageHistory:
	def __init__(self, user_id: str):
		if user_id not in store:
			store[user_id] = []
		self.user_id = user_id
		self.history = store[self.user_id]

	def add_user_message(self, message: str):
		store[self.user_id].append(HumanMessage(content=message))

	def add_ai_message(self, message: str):
		store[self.user_id].append(AIMessage(content=message))

	def clear(self):
		store[self.user_id] = []
	
	def get_context(self, max_messages: int = 10):
		return store[self.user_id][-max_messages:]

def obtener_guardar_historia(dic: dict) -> tuple[str, str]:
	history = ChatMessageHistory(user_id=dic["user_id"])
	context = history.get_context()
	history.add_user_message(dic["input"])
	return {"history": context, "input": dic["input"], "user_id": dic["user_id"]}

#Guardar pregunta -> creamos prompt con contexto -> respuesta del modelo grande -> humanizar respuesta -> respuesta final



prompt_principal = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente que escribe una repuesta breve a un tweet de manera informal"),
		MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)


# Para evitar cargar el modelo cada vez que se llame la herramienta
@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
	"""Carga modelo + adaptador una sola vez para evitar latencia por llamada."""
	tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

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

	model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
	model.eval()
	return model, tokenizer


def _build_prompt(tweet: str) -> str:
	# Mantener coherencia con la plantilla usada en entrenamiento
	return (
		"Reescribe el siguiente tweet en un estilo natural no formal.\n"
		f"Tweet: {tweet.strip()}\n"
		"Respuesta:\n"
	)



@tool("humanizar_tweet")
def humanizar_tweet(tweet: str) -> str:
	"""Reescribe un tweet en estilo natural no formal usando Qwen+LoRA local."""
	if not tweet or not tweet.strip():
		return "El tweet esta vacio."

	if not ADAPTER_PATH.exists():
		return f"No se encontro el adaptador en: {ADAPTER_PATH.resolve()}"

	model, tokenizer = _load_model_and_tokenizer()
	prompt = _build_prompt(tweet)

	inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
	device = next(model.parameters()).device
	inputs = {k: v.to(device) for k, v in inputs.items()}

	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=80,
			do_sample=True,
			temperature=0.4,
			repetition_penalty=1.1,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if "Respuesta:\n" in text:
		return text.split("Respuesta:\n", 1)[1].strip()
	return text.strip()


# Modelo grande para tarea principal 
gemini_api_key =os.environ.get("GOOGLE_API_KEY")
modelo_grande=ChatGoogleGenerativeAI(
	model="gemini-2.5-flash",
	temperature=0.7
)

def guardar_respuesta(data: dict)-> str:
	user_id = data.get("user_id")
	response = data.get("respuesta")
	humanized_response = humanizar_tweet.func(response)

	# Guardamos la respuesta humanizada en la memoria del usuario
	history = ChatMessageHistory(user_id=user_id)
	history.add_ai_message(humanized_response)

	return humanized_response

# Cadena de herramientas para el agente

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



cadena = (
    RunnableLambda(obtener_guardar_historia)
    | {
        "prompt_messages": prompt_principal,
        "user_id": itemgetter("user_id")  # ← Extrae user_id para pasarlo
    }
    | RunnablePassthrough.assign(
        respuesta=lambda x: modelo_grande.invoke(x["prompt_messages"]).content
    )
    | RunnableLambda(guardar_respuesta)
)



if __name__ == "__main__":
    # Ejemplo de uso
    entrada = {"input": "¿Qué opinas sobre la inteligencia artificial?", "user_id": "user1"}
    salida = cadena.invoke(entrada)  # ← Usa la variable, no repitas el dict
    print("Respuesta humanizada:", salida)
    
    salida2 = cadena.invoke({"input": "De que estabamos hablando?", "user_id": "user1"})
    print("Contexto actualizado:", salida2)
    
    salida3 = cadena.invoke({"input": "De que estabamos hablando?", "user_id": "user2"})
    print("Contexto nuevo usuario:", salida3)