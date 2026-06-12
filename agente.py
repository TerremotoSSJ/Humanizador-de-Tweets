from functools import lru_cache

import os
from pathlib import Path
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tavily import TavilyClient
from langgraph.checkpoint.sqlite import SqliteSaver
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = Path("./modelo_entrenado")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

from langgraph.checkpoint.memory import InMemorySaver

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



@tool("humanizar_tweet", return_direct=True)
def humanizar_tweet(tweet: str) -> str:
	"""Reescribe un tweet en estilo natural no formal usando Qwen+LoRA local."""
	if not tweet or not tweet.strip():
		return "El tweet esta vacio."

	if not ADAPTER_PATH.exists():
		return f"No se encontro el adaptador en: {ADAPTER_PATH.resolve()}"

	model, tokenizer = _load_model_and_tokenizer()
	prompt = (
        "Reescribe el siguiente tweet en un estilo natural no formal.\n"
		f"Tweet: {tweet.strip()}\n"
		"Respuesta:\n"
    )

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

@tool("buscar_informacion", return_direct=True)
def buscar_informacion(query: str) -> str:
    """Busca información relevante para el tweet usando Tavily."""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    contexto = tavily_client.get_search_context(
        query=query,
        search_depth="basic"
    )
    return contexto if contexto else "No se encontró información relevante."

# Agente

model="google_genai:gemini-2.5-flash"

agente=create_agent(model=model, 
					tools=[humanizar_tweet, buscar_informacion],
					system_prompt="""
Eres un asistente que tiene que responder a las preguntas y respuestas de los usuarios de Twitter de forma humanizada

REGLAS:
1. Si el usuario hace una pregunta o comentario sobre un tema específico, primero busca información relevante usando la herramienta de búsqueda y luego responde de forma natural usando la información encontrada.
2. Si el usuario comparte un tweet que quiere humanizar, usa la herramienta de humanización para reescribirlo en un estilo natural no formal.
3. Siempre mantén un tono amigable y cercano, como si estuvieras hablando con un amigo. Evita sonar como un robot o una empresa.
4. Si no entiendes algo o no tienes suficiente información, responde de forma honesta y empática, ofreciendo ayuda adicional si es posible.

Herramientas disponibles:
- humanizar_tweet(tweet: str) -> str: Reescribe un tweet en estilo natural no formal usando un modelo local de Qwen+LoRA.
- buscar_informacion(query: str) -> str: Busca información relevante para el tweet usando Tavily.
""",
checkpointer=InMemorySaver()         
)



def llamar_agente(thread_id: str, tweet: str):
	"""Función para llamar al agente con un thread_id específico, lo que permite mantener el contexto entre llamadas."""
	config={"configurable":{"thread_id": thread_id}}
	result = agente.invoke({"messages": tweet}, config=config)
	messages = result.get("messages", [])

	for msg in reversed(messages):
		if (hasattr(msg, "content") and msg.content):
			print("Agente:", msg.content)
	print("Resultado del agente:", result)
	
if __name__ == "__main__":
	thread_id=str("holi")
	llamar_agente(thread_id, "Que opinas sobre el nuevo iPhone?")
	llamar_agente(thread_id, "De que estabamos hablando?")