import requests
import os
from dotenv import load_dotenv
import pandas as pd

# Cargar las variables de entorno del archivo .env
load_dotenv()

# API key de WeatherAPI
api_key = os.getenv("WeatherAPI_KEY")

# Verificar si la API key está correctamente cargada
if not api_key:
    raise ValueError("API key no encontrada en las variables de entorno")

# Función para obtener el clima de una ciudad
def obtener_clima(ciudad, lat, lon):
    try:
        # URL de la API de WeatherAPI
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&lang=es"

        # Hacer la solicitud GET a la API
        response = requests.get(url)
        
        # Si la respuesta es exitosa
        if response.status_code == 200:
            data = response.json()
            
            # Extraer la información del clima
            temp = data['current']['temp_c']  # Temperatura en grados Celsius
            humidity = data['current']['humidity']  # Humedad en porcentaje
            description = data['current']['condition']['text']  # Descripción del clima
            return temp, humidity, description
        else:
            # Loguear el error en caso de una respuesta no exitosa
            print(f"Error al obtener datos para {ciudad} - Código de estado: {response.status_code}")
            return None, None, None
    except requests.exceptions.RequestException as e:
        # Loguear si ocurre una excepción durante la solicitud
        print(f"Error al solicitar datos para {ciudad}: {e}")
        return None, None, None

# Agregar las columnas de clima al DataFrame de ciudades
def obtener_datos_clima():
    ciudades_df['Temperatura'], ciudades_df['Humedad'], ciudades_df['Descripcion'] = zip(*ciudades_df.apply(
        lambda row: obtener_clima(row['Ciudad'], row['Latitud'], row['Longitud']), axis=1))
    
    return ciudades_df

# Cargar el CSV con los datos de las ciudades
csv_path = "../Modelo_recomendaci-n_Agencia_de_viajes/data/raw/listado_destinos_coordenadas.csv"
ciudades_df = pd.read_csv(csv_path)

# Obtener los datos de clima
ciudades_df = obtener_datos_clima()

# Verificar los resultados
print(ciudades_df.head())

# Guardar el DataFrame actualizado con la información del clima
ciudades_df.to_csv('../data/processed/ciudades_con_clima.csv', index=False)