import pandas as pd
import requests
from time import sleep
from datetime import datetime
from geopy.geocoders import Nominatim

# Cargar ciudades
df_ciudades = pd.read_csv("../data/processed/ciudades_con_clima.csv")

# Inicializar geolocalizador de Nominatim
geolocator = Nominatim(user_agent="clima_agencia_viajes")

datos_clima = []

start_date = "2020-01-01"
end_date = "2024-12-31"

for idx, row in df_ciudades.iterrows():
    ciudad = row['Ciudad']
    latitud = row['Latitud']
    longitud = row['Longitud']

    print(f"Procesando: {ciudad}")

    try:
        location = geolocator.reverse((latitud, longitud), language='es', timeout=10)
        if location:
            address = location.raw.get('address', {})


        else:
            pais = region = 'Desconocido'
    except Exception as e:
        print(f"Error al obtener datos de ubicación para {ciudad}: {str(e)}")
        pais = region = 'Desconocido'

    # Construir la URL de la API de Open-Meteo
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitud}&longitude={longitud}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=30)  # Timeout de 30 segundos
        if response.status_code == 200:
            data = response.json()
            fechas = data['daily']['time']
            temp_max = data['daily']['temperature_2m_max']
            temp_min = data['daily']['temperature_2m_min']
            precipitacion = data['daily']['precipitation_sum']

            # Añadir los datos meteorológicos con información adicional
            for i in range(len(fechas)):
                datos_clima.append({
                    'Ciudad': ciudad,
                    'Latitud': latitud,
                    'Longitud': longitud,
                    'Fecha': fechas[i],
                    'Temp_Max': temp_max[i],
                    'Temp_Min': temp_min[i],
                    'Precipitacion': precipitacion[i]
                })
        else:
            print(f"Error para {ciudad}: {response.status_code}")
    except Exception as e:
        print(f"Error para {ciudad}: {str(e)}")

    sleep(10)  # Dormir 1 segundo entre requests

# Crear DataFrame final
df_clima = pd.DataFrame(datos_clima)

# Guardar el archivo CSV con los datos adicionales
df_clima.to_csv("../data/processed/Datos_climaticos_completos_hist.csv", index=False)

print("¡Proceso completado!")