import pandas as pd
import requests
import json
from tqdm import tqdm

# --- CONFIGURACIÓN ---
RAPIDAPI_KEY = "e1beb37a81mshafcf57072ce609dp13cd52jsn849a9e1726af"
API_HOST = "google-flights2.p.rapidapi.com"
FLIGHTS_ENDPOINT = "https://priceline-com2.p.rapidapi.com/flights/details"  # Verificar si sigue vigente

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": API_HOST
}

# --- INPUT DEL USUARIO ---
origen_ciudad = input("Introduce la ciudad de origen: ").title().strip()
fecha_salida = input("Introduce la fecha de salida (YYYY-MM-DD): ")
fecha_regreso = input("Introduce la fecha de regreso (YYYY-MM-DD, deja en blanco para vuelos de ida): ")
class_of_service = input("Introduce la clase (ECONOMY, PREMIUM_ECONOMY, BUSSINES, FIRST): ").upper().strip()
# orden_de_clasificacion = input("Introduce el criterio de orden (ML_BEST_VALUE, DURATION, PRICE): ").upper().strip()

# --- CARGAR DESTINOS ---
csv_path = r"C:\Users\mafer\OneDrive\Escritorio\Predicccion_agencia_de_viajes_proyecto\data\processed\Codigos_aeropuestos.csv"
df_destinos = pd.read_csv(csv_path)
df_destinos['Ciudad'] = df_destinos['Ciudad'].str.strip().str.title()

# --- OBTENER CÓDIGO IATA DE ORIGEN ---
origen_row = df_destinos[df_destinos['Ciudad'] == origen_ciudad]
if not origen_row.empty:
    origen_iata = origen_row['COD_IATA'].iloc[0]
    print(f"Código IATA de origen: {origen_iata}")
else:
    print(f"No se encontró el código IATA para la ciudad de origen: {origen_ciudad}. El programa finalizará.")
    exit()

# --- FUNCIÓN PARA OBTENER PRECIO DE VUELO ---
def obtener_precio_vuelo(origen, destino_iata, fecha_salida, fecha_regreso=None, clase="ECONOMY", orden="PRICE"):
    legs = [{"sourceAirportCode": origen, "destinationAirportCode": destino_iata, "date": fecha_salida}]
    
    if fecha_regreso:
        legs.append({
            "sourceAirportCode": destino_iata,
            "destinationAirportCode": origen,
            "date": fecha_regreso
        })
    
    payload = {
        "legs": legs,
        "classOfService": clase,
        "sortOrder": orden
    }

    try:
        response = requests.post(FLIGHTS_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"❌ Error al decodificar JSON para {origen} → {destino_iata}.")
            print("Contenido recibido:", response.text)
            return None

        # ⚠️ Ajustar esta parte según la estructura real de la respuesta:
        itinerarios = data.get("data", {}).get("flights", [])
        precios = []

        for vuelo in itinerarios:
            try:
                precio = vuelo.get("purchaseLinks", [])[0].get("totalPrice", {}).get("formatted")
                if precio:
                    precio_numerico = float(precio.replace("€", "").replace(",", "").strip())
                    precios.append(precio_numerico)
            except Exception as e:
                continue

        return min(precios) if precios else None

    except requests.exceptions.RequestException as err:
        print(f"❌ Error de solicitud para {origen} → {destino_iata}: {err}")
        return None

# --- ITERAR DESTINOS ---
# --- ITERAR DESTINOS ---
resultados_vuelos = []
for _, row in tqdm(df_destinos.iterrows(), total=len(df_destinos)):
    ciudad_destino = row['Ciudad']
    destino_iata = row['COD_IATA']

    if pd.notna(destino_iata) and destino_iata != origen_iata:
        # 🔽 AQUÍ SE USA LA VERSIÓN NUEVA DE LA FUNCIÓN
        precio = obtener_precio_vuelo(
            origen_iata,
            destino_iata,
            fecha_salida,
            fecha_regreso,
            class_of_service,
        )

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
            print(f"No se encontraron precios para {ciudad_destino} ({destino_iata}) desde {origen_ciudad}.")
    else:
        print(f"⚠️ Código IATA no válido o igual al origen para {ciudad_destino}.")


# --- GUARDAR RESULTADOS ---
if resultados_vuelos:
    df_resultados = pd.DataFrame(resultados_vuelos)
    print("\n--- Resultados ---")
    print(df_resultados)

    df_resultados.to_csv(
        r"C:\Users\mafer\OneDrive\Escritorio\Predicccion_agencia_de_viajes_proyecto\data\processed\precios_vuelos.csv",
        index=False
    )
    print("✅ Resultados guardados correctamente.")
else:
    print("⚠️ No se encontraron resultados. No se guardó ningún archivo.")
