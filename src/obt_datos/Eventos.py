import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Leer dataset de ciudades
df_ciudades = pd.read_csv("../data/processed/ciudades_con_clima.csv")

# Cargar API KEY de entorno
load_dotenv()
API_KEY = os.getenv("TICKETMASTER_API_KEY")

if not API_KEY:
    raise ValueError("Falta la clave de API de Ticketmaster")

# URL base
url = "https://app.ticketmaster.com/discovery/v2/events.json"

# Lista para guardar todos los eventos como filas
eventos_lista = []

for _, row in df_ciudades.iterrows():
    ciudad = row['Ciudad']
    lat = row['Latitud']
    lon = row['Longitud']

    fecha_inicio = datetime.utcnow().replace(second=0, microsecond=0).isoformat() + "Z"
    fecha_fin = (datetime.utcnow() + timedelta(days=300)).replace(second=0, microsecond=0).isoformat() + "Z"

    print(f"Buscando eventos para {ciudad}...")

    params = {
        "apikey": API_KEY,
        "latlong": f"{lat},{lon}",
        "radius": 50,
        "startDateTime": fecha_inicio,
        "endDateTime": fecha_fin,
        "size": 50,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        print(f"Requesting events for {ciudad} with URL: {response.url}")

        if response.status_code == 200:
            data = response.json()
            events = data.get("_embedded", {}).get("events", [])

            for event in events:
                nombre = event.get("name", "Sin nombre")
                fecha = event.get("dates", {}).get("start", {}).get("localDate", "Sin fecha")
                descripcion = event.get("info", "") or event.get("pleaseNote", "Sin descripción")
                categoria = event.get("classifications", [{}])[0].get("segment", {}).get("name", "Desconocido")

                eventos_lista.append({
                    "Ciudad": ciudad,
                    "Latitud": lat,
                    "Longitud": lon,
                    "Nombre_Evento": nombre,
                    "Fecha_Evento": fecha,
                    "Descripción": descripcion,
                    "Categoría": categoria
                })
        else:
            print(f"Error en {ciudad}: {response.status_code} {response.text}")

    except Exception as e:
        print(f"Error buscando eventos en {ciudad}: {str(e)}")

    sleep(1)

# Crear DataFrame y guardar
df_eventos = pd.DataFrame(eventos_lista)
df_eventos.to_csv("../data/processed/ciudades_con_eventos.csv", index=False)

print("¡Eventos guardados por fila correctamente!")