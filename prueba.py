import os
import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime

# Configura la carpeta de salida
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("🏨 Búsqueda de Hoteles por Ubicación")
st.markdown("Encuentra hoteles cerca de una ubicación específica usando TripAdvisor.")

# Configuración de la API
API_URL = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchHotelsByLocation"
HEADERS = {
    "X-RapidAPI-Key": "TU_RAPIDAPI_KEY",  # Reemplaza con tu clave de RapidAPI
    "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com"
}

# Configuración del formulario
latitud = st.text_input("📍 Latitud", value="40.730610")
longitud = st.text_input("📍 Longitud", value="-73.935242")
moneda = st.selectbox("💲 Moneda", ["USD", "EUR", "GBP", "MXN", "COP", "ARS"], index=0)
pagina = st.number_input("📄 Número de página", min_value=1, value=1, step=1)

# Directorio para guardar los archivos procesados
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio si no existe

# Botón de búsqueda
if st.button("🔍 Buscar Hoteles"):
    try:
        # Parámetros de la API
        params = {
            "latitude": latitud,
            "longitude": longitud,
            "pageNumber": pagina,
            "currencyCode": moneda
        }

        # Llamada a la API
        response = requests.get(API_URL, headers=HEADERS, params=params)

        # Verifica si la respuesta es exitosa
        if response.status_code == 200:
            hoteles = response.json().get("data", [])

            # Muestra los hoteles en una tabla
            if hoteles:
                df_hoteles = pd.DataFrame(hoteles)
                
                # Guarda los datos en un archivo CSV
                output_file = os.path.join(output_dir, "hoteles.csv")
                df_hoteles.to_csv(output_file, index=False)
                
                st.success(f"✅ Archivo guardado exitosamente en: {output_file}")
                st.dataframe(df_hoteles[["name", "address", "rating", "price", "ranking"]])
            else:
                st.warning("⚠️ No se encontraron hoteles para esta ubicación.")

        else:
            st.error(f"Error en la API: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
