import json
import re
from pathlib import Path


INPUT_FILE = Path("tweets_futbol_para_chatgpt_palabra_completa.jsonl")
OUTPUT_FILE = Path("tweets_futbol_sin_enlaces.jsonl")

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


def quitar_enlaces(texto: str) -> str:
	"""Quita solamente enlaces de un texto y deja el resto igual."""
	return URL_PATTERN.sub("", texto)


def main() -> None:
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
				row["tweet"] = quitar_enlaces(tweet)

			json.dump(row, fout, ensure_ascii=False)
			fout.write("\n")

	print(f"✅ Archivo generado: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
