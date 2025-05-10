import praw
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ciudades-trends-script")

# Autenticación
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Leer dataset
df_ciudades = pd.read_csv("../data/processed/ciudades_con_clima.csv")
popularidad_reddit = []

# Buscar menciones por ciudad
for idx, row in df_ciudades.iterrows():
    ciudad = row['Ciudad']
    query = f"travel {ciudad}"
    print(f"Buscando Reddit posts para: {query}")

    try:
        count = sum(1 for _ in reddit.subreddit("all").search(query, limit=100, sort='new'))
        popularidad_reddit.append(count)
    except Exception as e:
        print(f"Error en {ciudad}: {str(e)}")
        popularidad_reddit.append(None)

    sleep(2)

# Añadir columna y guardar
df_ciudades["popularidad_reddit"] = popularidad_reddit
df_ciudades.to_csv("../data/processed/ciudades_con_reddit.csv", index=False)

print("¡Popularidad Reddit añadida!")