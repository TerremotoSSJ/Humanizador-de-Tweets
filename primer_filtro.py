# paso_1_filtrar_tweets.py
import os
import sys

# Evita un crash al finalizar Python por hilos de descarga (hf_transfer).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from datasets import load_dataset
import re
import json
from itertools import islice
from tqdm import tqdm
from pathlib import Path



outputFile=Path("tweets_futbol.jsonl")
# 1. Cargar dataset (empieza con una parte)
print("Cargando dataset")
# Usamos streaming ya que hay muchos tweets.
dataset = load_dataset("pysentimiento/spanish-tweets", split="train", streaming=True)

# 2. Filtrar tweets de fútbol

terminos_evadir = [
    # ============================================
    # POLÍTICA GENERAL
    # ============================================
    "política", "gobierno", "elecciones", "corrupción", "economía", "inflación",
    "presidente", "congreso", "senado", "diputados", "parlamento", "ministerio",
    "votar", "voto", "campaña", "partido político", "izquierda", "derecha",
    
    # ============================================
    # POLÍTICA MÉXICO
    # ============================================
    "amlo", "lopez", "morena", "gatell", "elcacas", "amloveteya", "lopezveteya",
    "lopezalpenal", "lopezasesino", "morenagolpista", "golpista", "lopezcorrupto",
    "amlofracaso", "amloasesino", "yeidckol", "ackerman", "gatellcriminal",
    "alfaro", "lopezhaciendonarco", "morenaterrorista", "morenacuervo", "frenaaa",
    "jalisco", "jalisconoesta", "alfaropresidente", "morenafuera",
    
    # ============================================
    # POLÍTICA COLOMBIA
    # ============================================
    "uribe", "alvaro uribe", "duque", "petro", "fajardo", "paramilitares",
    "farc", "eln", "uribismo", "uribista", "paronacional", "paro nacional",
    "esmad", "manifestacion", "colombia", "golpe de estado",
    
    # ============================================
    # POLÍTICA OTROS PAÍSES
    # ============================================
    "bolivia", "evo", "venezuela", "maduro", "cuba", "castro", "chavismo",
    "psuv", "oea", "almagro", "derechos humanos", "dictadura", "autoritarismo",
    "totalitarismo", "represión", "censura", "opresión", "tiranía", "despotismo",
    "autocracia", "regímenes autoritarios", "militar", "fascismo",
    
    # ============================================
    # K-POP / BTS (masivos)
    # ============================================
    "bts", "army", "jungkook", "jimin", "v", "taehyung", "jhope", "suga", "rm",
    "jin", "bangtan", "henecia", "hyunjoong", "ss501", "khj", "jikook", "vmin",
    "golden", "euphoria", "staygold", "pcas", "poptimeawards", "grupodoano",
    "bangtansonyeondan", "btswins", "btswin", "mtv", "btsarmy", "bts_is_coming",
    "kpop", "k-pop", "blackpink", "twice", "exo", "nct", "seventeen", "stray kids",
    "lady gaga", "dua lipa", "selenagomez", "arianagrande", "justinbieber", "shawnmendes",
    "aguilera", "theweeknd", "weeknd", "ed sheeran", "dua lipa", "halsey", "billie eilish",
    
    # ============================================
    # LUCHA LIBRE (AEW, WWE)
    # ============================================
    "aew", "aewdynamite", "aewfull", "aewontnt", "aewrevolution", "kenny omega",
    "moxley", "jericho", "young bucks", "wwe", "triplemania", "lucha libre",
    "aewrampage", "aewcollision", "cm punk", "bryan danielson", "jon moxley",
    "chris jericho", "the elite", "bullet club", "lucha", "luchador", "ring",
    "wrestling", "raw", "smackdown", "nxt", "royal rumble", "wrestlemania",
    
    # ============================================
    # SALUD Y PANDEMIA
    # ============================================
    "covid", "pandemia", "vacuna", "salud pública", "hospitales", "cuarentena",
    "coronavirus", "contagio", "aislamiento", "mascarilla", "cubrebocas",
    
    # ============================================
    # EDUCACIÓN
    # ============================================
    "educación", "escuela", "universidad", "colegio", "maestro", "profesor",
    "estudiante", "clases", "examen", "tarea", "materia", "semestre",
    
    # ============================================
    # SEGURIDAD Y JUSTICIA
    # ============================================
    "seguridad", "justicia", "inseguridad", "violencia", "delincuencia", "crimen",
    "narcotráfico", "narco", "cártel", "asesinato", "homicidio", "robos", "criminalidad",
    "violaciones", "feminicidio", "abuso", "corrupción policial", "policía", "prisión", "carcel", "juzgado",
    "criminal", "criminales", "delincuentes", "violentos", "violenta", "violento",
    
    # ============================================
    # MEDIO AMBIENTE
    # ============================================
    "medio ambiente", "cambio climático", "ecología", "contaminación", "reciclaje",
    "sostenibilidad", "clima", "calentamiento global", "deforestación",
    
    # ============================================
    # ECONOMÍA Y TRABAJO
    # ============================================
    "economía", "inflación", "desempleo", "paro", "salario", "pobreza", "empleo",
    "trabajo", "empresa", "negocio", "emprendimiento", "pensiones", "jubilación",
    
    # ============================================
    # TECNOLOGÍA
    # ============================================
    "tecnología", "inteligencia artificial", "redes sociales", "internet",
    "ciberseguridad", "criptomonedas", "blockchain", "bitcoin", "ethereum",
    "software", "hardware", "computadora", "celular", "smartphone", "app",
    
    # ============================================
    # ENTRETENIMIENTO GENERAL
    # ============================================
    "entretenimiento", "cine", "música", "series", "celebridades", "moda",
    "videojuegos", "anime", "comics", "netflix", "disney", "marvel", "dc",
    "película", "pelicula", "serie", "capítulo", "temporada", "personaje",
    "streamer", "twitch", "youtube", "youtuber", "influencer",
    
    # ============================================
    # VIDEOJUEGOS ESPECÍFICOS
    # ============================================
    "roblox", "minecraft", "fortnite", "lol", "league of legends", "valorant",
    "call of duty", "gaming", "gamer", "playstation", "xbox", "nintendo",
    "pokemon", "zelda", "final fantasy", "gta", "grand theft auto",
    
    # ============================================
    # TEMAS SOCIALES
    # ============================================
    "lgtbi", "lgbt", "lgbtq", "orgullo", "diversidad", "feminismo", "machismo",
    "género", "igualdad", "discriminación", "racismo", "xenofobia", "homofobia",
    "migración", "inmigración", "refugiados", "fronteras",
    
    # ============================================
    # PALABRAS POLISÉMICAS (contexto NO fútbol)
    # ============================================
    # Alaba (verbo alabar, no el jugador)
     "alabar", "alabando", "alaban",
    
    # Amarilla (color, no tarjeta)
    "amarillas", "camiseta amarilla", "luz amarilla",
    
    # Arsenal (armas, no club)
    "arsenal de armas", "arsenal nuclear", "arsenal militar",
    
    # Asistencia (ayuda social, no deportiva)
    "asistencia social", "asistencia médica", "asistencia sanitaria", "asistencia humanitaria",
    "tomar asistencia", "pasar asistencia", "asistencia perfecta", "falta de asistencia",
    
    # Bestia (insulto genérico, no elogio deportivo)
    "bestia", "bestia humana", "la bestia", "animal bestia",
    
    # Blancos (color, política)
     "zapatos blancos", "calcetines blancos", "pantalones blancos",
    "supremacistas blancos", "hombres blancos", "vino blanco",
    
    # Bloqueó (redes sociales)
     "bloquear", "me bloqueó", "te bloqueó",
    
    # Cadiz (ciudad)
     "playa de cadiz", "ciudad de cadiz",
    
    # Central (ubicación, no club)
    "zona central", "estación central", "banco central", "parque central",
    "mercado central", "comedy central", "sede central",
    
    # Centró (verbo concentrar)
    "centró", "centrarse", "se centró", "centrado en",
    
    # Chilena (selección chilena, no gol)
     "chilena sub", "roja chilena",
    
    # Copa (copa de vino)
    "copa de vino", "copa de licor", "copa de whisky", "copa de champán", "copa de cerveza",
    
    # Corner (esquina)
    "corner de", "corner la", "corner del",
    
    
    # Defensa (defensa personal, ministerio)
    "defensa personal", "ministerio de defensa", "defensa propia", "legítima defensa",
    
    # Extremo (adjetivo)
     "caso extremo", "situación extrema", "extremo peligro", "extremo frío", "extremo calor",
    
    # Falta (expresión sentimental)
    "me hace falta", "te hago falta", "hace falta", "falta que me haces",
    "falta un beso", "falta de cariño",
    
    # Fichaje (fichaje laboral)
    "fichaje de personal", "fichaje laboral", "fichaje de empleados",
    
    # Final (genérico)
    "final feliz", "final de película", "final de serie", "llegó a su final",
    
    # Goles (golpes)
    "golpes", "a golpes", "recibir golpes", "dar golpes", "golpear", "golpeó",
    
    # Gremio (sindicato)
    "gremio de taxistas", "gremio docente", "gremio de trabajadores",
    
    # Inter (interés, no club)
    "interés", "interesante", "interés en", "demostrar interés", "falta de interés",
    
    # Liga (romance)
    "ligar", "ligue", "ligar con alguien", "quiero ligar",
    
    # Mano (parte del cuerpo)
     "manos", "a mano", "de la mano", "dar la mano", "mano derecha",
    
    # Milan (ciudad)
    "ciudad de milán", "viajar a milán",
    
    # Muerto (fallecimiento)
    "muerto", "muerta", "falleció", "murió", "está muerto",
    
    # Mundial (adjetivo)
    "a nivel mundial", "éxito mundial", "fama mundial", "problema mundial",
    
    # Palmas (aplausos)
    "palmas", "aplausos", "a palmas", "palmas en",
    
    # Paredes (pared)
    "paredes", "pared", "entre paredes", "paredes de",
    
    # Partido (político)
    "partido político", "partido de izquierda", "partido de derecha",
    
    # Pase (paseo, tiempo)
    "paseo", "dar un paseo", "la pase bien", "que la pases",
        
    # Perro (animal)
    "perro", "perros", "mascota", "animal perro",
    
    # Portero (portero de edificio)
    "portero de edificio", "portero automático",
    
    # Poste (poste de luz)
    "poste de luz", "poste eléctrico", "poste de teléfono",
    
    # Ramos (flores)
    "ramo de flores", "ramo de rosas", "ramo",
    
    # Roma (ciudad)
    "roma", "ciudad de roma", "viajar a roma",
    
    # Santos (religión)
    "santos", "los santos", "día de los santos", "santo",
    
    # Técnico (técnico de algo)
    "técnico de", "técnico en", "técnico informático",
    
    # Título (académico)
    "título universitario", "título académico", "obtener título",
    
    # Var (variable)
    "var", "variable", "var en", "var de",
    
    # Victoria (nombre)
     "victoria nombre", "victoria beckham",
    
    # ============================================
    # EXPRESIONES COMUNES NO FUTBOLERAS
    # ============================================
    "te quiero", "te amo", "feliz cumpleaños", "happy birthday",
    "recuerdos", "nostalgia", "amor", "corazón", "beso",
    "familia", "amigo", "amiga", "relación", "novio", "novia",
    "éramos", "somos", "eres", "conocer", "conocerte",
    
    # ============================================
    # FRASES COMUNES EN REDES
    # ============================================
    "lo logramos", "lo logre", "logramos", "superamos", "meta",
    "weverse", "dm", "privado", "instagram", "facebook", "tiktok",

    "ciudad de", "municipio", "ayuntamiento", "calle", "plaza",
    "pueblo", "localidad", "barrio", "distrito",
    "turismo", "viajar", "visitar", "hotel", "alojamiento",
    "universidad", "campus", "facultad",
    "fiesta", "feria", "tradición", "procesión",
    "colegio", "instituto", "escuela", "educación",
    "estat espanyol", "estado español"
]

terminos_evadir = [t.lower() for t in terminos_evadir]

terminos_futbol_seguros = [
    # ============================================
    # EQUIPOS ESPAÑOLES (nombres completos)
    # ============================================
    "real madrid", "fc barcelona", "atlético madrid", "atletico madrid",
    "sevilla", "valencia", "real betis", "villareal", "real sociedad",
    "athletic bilbao", "celta de vigo", "osasuna", "getafe",
    "rayo vallecano", "alaves", "las palmas",
     "levante", "sporting de gijon",
    
    # ============================================
    # EQUIPOS INTERNACIONALES
    # ============================================
    "river plate", "boca juniors", "racing", "san lorenzo",
    "flamengo", "palmeiras",
    "milan", "inter", "juventus", "napoli", "roma", "lazio", "atalanta",
    "bayern munich", "borussia dortmund", "rb leipzig", "bayer leverkusen",
    "psg", "marseille", "monaco",
    "manchester united", "manchester city", "liverpool", "chelsea",
    "tottenham", "newcastle", "aston villa", "everton", "leicester",
    
    # ============================================
    # JUGADORES (ampliado)
    # ============================================
    "messi", "cristiano ronaldo", "cristiano", "ronaldo", "neymar", "benzema",
    "lewandowski", "vinicius jr", "vinicius", "rodrygo", "mbappé", "kylian",
    "haaland", "bellingham", "griezmann", "luis suárez", "aguero",
    "iniesta", "xavi", "ramos", "sergio ramos", "piqué", "modric", "kroos",
    "casemiro", "valverde", "camavinga", "tchouameni", "rudiger", "alaba",
    "courtois", "ter stegen", "oblak", "dibu", "emiliano martinez",
    "di maría", "enzo fernández", "julián álvarez", "álvarez",
    "dybala", "lautaro", "cavani", "darwin núñez", "núñez",
    "maldini", "zlatan", "ibrahimovic", "robert lewandowski", "lewandowski",
    "pogba", "kante", "kante n'golo", "n'golo kante", "de bruyne", "kevin de bruyne",
    "son heung-min", "heung-min", "harry kane", "kane", "bruno fernandes", "fernandes", "jack grealish", "grealish",
    "iker casillas", "casillas", "xavi hernández", "sergio busquets", "busquets",
     "mohamed salah", "diego maradona", "maradona", "zico", "ronaldinho",
    "higuaín", "gonzalo higuaín", "roberto carlos", "santi cazorla", "dani alves",
    "thiago alcántara", "thiago silva", "rodrigo de paul","marcos llorente",

    
    # ============================================
    # COMPETICIONES
    # ============================================
    "champions league", "europa league", "conference league",
    "copa libertadores", "copa sudamericana", "copa del rey",
    "la liga", "premier league", "serie a", "bundesliga", "ligue 1",
    "primera división", "segunda división", "supercopa", "eurocopa", "copa américa", "finalissima",
    
    # ============================================
    # TÉRMINOS FUTBOLEROS (ampliados)
    # ============================================
    "gol", "goles", "golazo", "goleada", "goleador", "goleadores",
    "penalti", "penalty", "tiro libre", "falta directa",
    "tarjeta roja", "roja directa", "tarjeta amarilla",
    "fuera de juego", "offside", "var", "videoarbitraje",
    "hat trick", "triplete", "doblete", "volea", "regate", "dribbling", "saque de esquina", "corner", "saca de esquina",
    "portería", "larguero", "travesaño", "mano penal", "amonestación",
    "prórroga", "tiempo extra", "penales","partido del equipo", "partido del club", "partido de la selección", "partido de fútbol",
    
    # ============================================
    # VERBOS FUTBOLEROS
    # ============================================
     "marcó un gol", "anotó un gol",
    "empató", "remontó", "goleó",
    "atajó", "amonestaron",
    "regateó", "dribló", "centró", "cabeceó", "remató", "disparó",
    "fichó", "fichaje", "renovó",
    
    # ============================================
    # JERGA FUTBOLERA (hinchas)
    # ============================================
    "partidazo", "derbi", "remontada", "goleada",
     "al larguero", "al travesaño",
    "fue gol", "no fue gol", "qué jugador",
    "pecho frío",
    
    # ============================================
    # PUESTOS Y ROLES
    # ============================================
    "delantero", "delantera", "centrodelantero",
    "mediapunta", "extremo derecho", "extremo izquierdo", "interior",
    "centrocampista", "mediocentro", "pivote", "lateral", "carrilero",
    "portero", "arquero", "guardameta", "dt", "director técnico", "míster",
    "árbitro", "referí", "juez de línea", "cuarto árbitro",
    
    ### ============================================
    # TÉRMINOS NEGATIVOS
    ### ============================================

    "arbitro ladrón", "arbitro robó", "arbitro robando","arbitro corrupto", "partido robado", 
    "negreira","arbitro corrupto", "partido comprado", "liga comprada","liga amañada", 
    "partido regalado", "partido amañado", "arbitro comprado", 


    # ============================================
    # CLUBES POR APODOS (útil para filtrar)
    # ============================================
     "culers", "cules", "colchoneros",
    "rojiblancos", "verdiblancos","txuri urdin"
    , "xeneizes", "bosteros","culerdos"




]

terminos_futbol_seguros = [t.lower() for t in terminos_futbol_seguros]


def palabra_disparadora(texto, palabras_futbol=terminos_futbol_seguros):
    texto_limpio = texto.lower()
    for p in palabras_futbol:
        if p in texto_limpio:
            return p
    return None

def limpiar_tweet(texto):
    """Limpia menciones, URLs, hashtags"""
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'#', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def primer_filtro():
    print("Filtrando tweets")
    tweets_futbol = []
    max_tweets_a_revisar = 30000
    for tweet in tqdm(
        islice(dataset, max_tweets_a_revisar),
        total=max_tweets_a_revisar,
        desc="Revisando tweets",
        unit="tweet",
    ):
        texto = tweet['text']

        # Si aparece un término negativo, se rechaza directamente.
        palabra_negativa = palabra_disparadora(texto, terminos_evadir)
        if palabra_negativa:
            continue

        # Solo si no hubo negativos, se valida por terminos seguros de fútbol.
        palabra_match = palabra_disparadora(texto, terminos_futbol_seguros)
        if palabra_match:
            tweets_futbol.append({
                "tweet_original": texto,
                "tweet_limpio": limpiar_tweet(texto),
                "palabra_disparadora": palabra_match,
            })

        if len(tweets_futbol) >= 50000:  # empieza con 10k
            break

    print(f"✅ {len(tweets_futbol)} tweets de fútbol")

    # Guardamos en un archivo JSONL
    with open("tweets_futbol.jsonl", "w", encoding="utf-8") as f:
        for t in tweets_futbol:
            json.dump({
                "palabra_disparadora": t['palabra_disparadora'],
                "tweet": t['tweet_original'],
            }, f, ensure_ascii=False)
            f.write("\n")

    print("Guardados en futbol_tweets.jsonl")




if __name__ == "__main__":
    primer_filtro()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

