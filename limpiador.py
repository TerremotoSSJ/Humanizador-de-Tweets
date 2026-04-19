import json
import re
from pathlib import Path
import random

INPUT_FILE = Path("tweets_futbol_completa.jsonl")
OUTPUT_FILE = Path("tweets_futbol_limpio.jsonl")

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION_PATTERN = re.compile(r"@\w+")

def quitar_enlaces(texto: str) -> str:
	"""Quita solamente enlaces de un texto y deja el resto igual."""
	return URL_PATTERN.sub("", texto)

def quitar_menciones(texto: str) -> str:
	"""Quita solamente menciones de un texto y deja el resto igual."""
	return MENTION_PATTERN.sub("", texto)

def main(enlaces: bool, menciones: bool, probabilidad_menciones: float) -> None:
	if not INPUT_FILE.exists():
		print(f"❌ No existe el archivo: {INPUT_FILE}")
		return

	with INPUT_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
		for line in fin:
			line = line.strip()
			if not line:
				continue

			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				continue

			tweet = row.get("tweet")
			if isinstance(tweet, str):
				if enlaces:
					row["tweet"] = quitar_enlaces(tweet)
				if menciones and random.random() < probabilidad_menciones:  
					row["tweet"] = quitar_menciones(row["tweet"])

			json.dump(row, fout, ensure_ascii=False)
			fout.write("\n")

	print(f"✅ Archivo generado: {OUTPUT_FILE}")


if __name__ == "__main__":
	enlaces=input("¿Quieres quitar enlaces de los tweets? (s/n): ").strip().lower() == "s"
	menciones=input("¿Quieres quitar menciones de los tweets? (s/n): ").strip().lower() == "s"
	if menciones:
		probabilidad_menciones = float(input("¿Con qué probabilidad quieres quitar menciones? (0-1): ").strip())
	else:
		probabilidad_menciones = 0.0
	main(enlaces, menciones, probabilidad_menciones)
