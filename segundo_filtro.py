import json
import re
from pathlib import Path

from tqdm import tqdm


INPUT_FILE = Path("tweets_futbol_para_chatgpt.jsonl")
OUTPUT_FILE = Path("tweets_futbol_para_chatgpt_palabra_completa.jsonl")


def build_full_match_pattern(trigger: str) -> re.Pattern[str]:
	"""Create a regex that matches the trigger as a full word/phrase, not substring."""
	escaped = re.escape(trigger.strip())
	escaped = escaped.replace(r"\ ", r"\s+")
	return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def is_full_trigger_match(tweet: str, trigger: str) -> bool:
	if not tweet or not trigger:
		return False
	pattern = build_full_match_pattern(trigger)
	return pattern.search(tweet) is not None


def main() -> None:
	if not INPUT_FILE.exists():
		print(f"❌ No existe el archivo: {INPUT_FILE}")
		return

	total_lines = sum(1 for _ in INPUT_FILE.open("r", encoding="utf-8"))
	kept = 0
	skipped = 0
	invalid = 0
	accepted_rows = []

	with INPUT_FILE.open("r", encoding="utf-8") as fin:
		for line in tqdm(fin, total=total_lines, desc="Filtrando palabra completa", unit="linea"):
			line = line.strip()
			if not line:
				continue

			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				invalid += 1
				continue

			tweet = str(row.get("tweet", ""))
			trigger = str(row.get("palabra_disparadora", ""))

			if is_full_trigger_match(tweet, trigger):
				accepted_rows.append(row)
				kept += 1
			else:
				skipped += 1

	accepted_rows.sort(key=lambda r: str(r.get("palabra_disparadora", "")).lower())

	with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
		for row in accepted_rows:
			json.dump(row, fout, ensure_ascii=False)
			fout.write("\n")

	print(f"✅ Filtrado completo. Aceptados: {kept}")
	print(f"🚫 Rechazados por subcadena/no match completo: {skipped}")
	if invalid:
		print(f"⚠️ Líneas inválidas ignoradas: {invalid}")
	print(f"💾 Archivo generado: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
