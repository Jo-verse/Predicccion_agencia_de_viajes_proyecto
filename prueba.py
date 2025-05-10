import requests
import pandas as pd

# ✅ Parámetros configurables
url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchHotelsByLocation"
querystring = {
    "latitude": "40.730610",
    "longitude": "-73.935242",
    "pageNumber": "1",
    "currencyCode": "USD"
}
headers = {
    "x-rapidapi-host": "tripadvisor16.p.rapidapi.com",
    "x-rapidapi-key": "9657827be4mshc6f461c74460ae9p1c1d9ejsn1f282702e67f",  # Clave de API proporcionada
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ✅ Llamada a la API
response = requests.get(url, headers=headers, params=querystring)

# ✅ Validar respuesta
if response.status_code == 200:
    try:
        data = response.json()
        if 'data' in data and data['data'] is not None:
            hoteles = []
            for hotel in data['data']:
                hoteles.append({
                    "Nombre": hotel.get("name"),
                    "Dirección": hotel.get("address"),
                    "Teléfono": hotel.get("phone"),
                    "Rating": hotel.get("rating"),
                    "Número de Opiniones": hotel.get("num_reviews"),
                    "Precio": hotel.get("price_level"),
                    "Categoría": hotel.get("category"),
                    "Imagen": hotel.get("photo")
                })
            df = pd.DataFrame(hoteles)
            df.to_csv("data/processed/hoteles_tripadvisor.csv", index=False)
            print(f"✅ Datos guardados en data/processed/hoteles_tripadvisor.csv ({len(df)} registros)")
        else:
            print("❌ La respuesta no contiene la clave 'data' o es None.")
    except Exception as e:
        print(f"❌ Error al procesar los datos: {e}")
else:
    print(f"❌ Error al obtener los datos de la API: {response.status_code}")
