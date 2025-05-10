import pandas as pd
import requests
import json  # Importa la biblioteca json para manejar errores de decodificación

# --- CONFIGURACIÓN ---
RAPIDAPI_KEY = "badd9d87a3msh268e94afba1b1c8p1f8621jsnbd311e2c0c3f"  # Reemplaza con tu clave de RapidAPI
API_HOST = "flights-sky.p.rapidapi.com" # Reemplaza con el host de la API (lo verás en RapidAPI)
FLIGHTS_ENDPOINT = "https://flights-sky.p.rapidapi.com/flights/price-calendar?fromEntityId=CDG" # ¡DEBES VERIFICAR Y ACTUALIZAR ESTE ENDPOINT EN LA DOCUMENTACIÓN DE LA API!

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": API_HOST
}

# --- INPUT DEL USUARIO ---
origen_ciudad = input("Introduce la ciudad de origen: ").title()
fecha_salida = input("Introduce la fecha de salida (YYYY-MM-DD): ")
fecha_regreso = input("Introduce la fecha de regreso (YYYY-MM-DD, deja en blanco para vuelos de ida): ")

# --- CARGAR DESTINOS DESDE CSV CON CÓDIGOS IATA ---
csv_path = "/Users/joanafernandes/Data Science Bootcamp/Predicccion_agencia_de_viajes_proyecto/data/processed/Codigos_aeropuestos.csv"
df_destinos = pd.read_csv(csv_path)

# --- OBTENER CÓDIGO IATA DEL ORIGEN DESDE EL DATAFRAME ---
origen_row = df_destinos[df_destinos['Ciudad'] == origen_ciudad]

if not origen_row.empty:
    origen_iata = origen_row['COD_IATA'].iloc[0]
    print(f"Código IATA de origen: {origen_iata}")
else:
    print(f"No se encontró el código IATA para la ciudad de origen: {origen_ciudad}. El programa finalizará.")
    exit()

# --- FUNCIÓN PARA OBTENER PRECIOS DE VUELOS ---
def obtener_precio_vuelo(origen, destino_iata, fecha_salida, fecha_regreso=None, moneda="EUR"):
    params = {
        "origin": origen,
        "destination": destino_iata,
        "departureDate": fecha_salida,
        "currency": moneda
    }
    if fecha_regreso:
        params["returnDate"] = fecha_regreso

    try:
        response = requests.get(FLIGHTS_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()  # Lanza una excepción para códigos de estado HTTP erróneos (4xx o 5xx)
        data = response.json()
        # --- PROCESAR LA RESPUESTA PARA EXTRAER EL PRECIO ---
        if data and 'data' in data and len(data['data']) > 0:
            precios = [flight['price'] for flight in data['data']]
            if precios:
                precio_minimo = min(precios)
                return precio_minimo
            else:
                return None
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error al buscar vuelo de {origen} a {destino_iata}: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP al buscar vuelo de {origen} a {destino_iata}: {e.response.status_code}")
        try:
            print(f"Contenido del error: {e.response.text}")
        except:
            print("No se pudo decodificar el contenido del error.")
        return None
    except KeyError:
        print(f"Error al procesar la respuesta para vuelo de {origen} a {destino_iata}: Estructura JSON inesperada.")
        return None
    except json.JSONDecodeError:
        print(f"Error al decodificar la respuesta JSON para vuelo de {origen} a {destino_iata}.")
        print(f"Contenido de la respuesta: {response.text}")
        return None

# --- ITERAR SOBRE LAS CIUDADES DE DESTINO ---
resultados_vuelos = []
for index, row in df_destinos.iterrows():
    ciudad_destino = row['Ciudad']
    destino_iata = row['COD_IATA']

    if pd.notna(destino_iata):
        precio = obtener_precio_vuelo(origen_iata, destino_iata, fecha_salida, fecha_regreso)
        if precio is not None:
            resultados_vuelos.append({
                "Origen": origen_ciudad,
                "Destino": ciudad_destino,
                "IATA Destino": destino_iata,
                "Fecha Salida": fecha_salida,
                "Fecha Regreso": fecha_regreso if fecha_regreso else "Ida",
                "Precio (EUR)": precio
            })
        else:
            print(f"No se encontraron precios para {ciudad_destino} ({destino_iata}) desde {origen_ciudad} en las fechas indicadas.")
    else:
        print(f"No se encontró el código IATA para {ciudad_destino}. Omitiendo.")

# --- MOSTRAR RESULTADOS ---
df_resultados_vuelos = pd.DataFrame(resultados_vuelos)
print("\n--- Resultados de búsqueda de vuelos ---")
print(df_resultados_vuelos)

# --- GUARDAR RESULTADOS EN CSV (Opcional) ---
df_resultados_vuelos.to_csv("data/processed/precios_vuelos.csv", index=False)

print("\n✅ Los precios de los vuelos se han guardado en 'data/processed/precios_vuelos.csv'.")