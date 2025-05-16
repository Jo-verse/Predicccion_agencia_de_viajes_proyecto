import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ========================
# 1. Cargar modelo y utilidades
# ========================
model = joblib.load("../models/model_completo.pkl")
le = joblib.load("../models/label_encoder.pkl")
columnas_modelo = pd.read_csv("../data/processed/x_train_columns.csv", header=None).squeeze().tolist()
full_df = pd.read_csv("../data/processed/total_data_240k.csv")

with open("../data/processed/Json/ciudad_transformation_rules.json") as f:
    ciudad_mapping = json.load(f)
id_to_ciudad = {str(v): k for k, v in ciudad_mapping.items()}

# ========================
# 2. Función de construcción de input
# ========================
def construir_input_usuario(valores_dict, columnas_modelo):
    df = pd.DataFrame(columns=columnas_modelo)
    df.loc[0] = 0
    for clave, valor in valores_dict.items():
        col = f"{clave}_{valor}"
        if col in df.columns:
            df.at[0, col] = 1
    for col in ["estimated_price_eur_x", "estimated_price_eur_y", "distance_to_city_center_km"]:
        if col in valores_dict:
            df.at[0, col] = valores_dict[col]
    return df

# ========================
# 3. Funciones enriquecimiento
# ========================
def get_clima_estimado(ciudad, temporada):
    clima = full_df[(full_df["ciudad"] == ciudad) & (full_df["temporada"] == temporada)]
    if clima.empty:
        return None
    return clima[["temp_max", "temp_min", "precipitacion"]].mean().round(1).to_dict()

def get_eventos(ciudad, temporada):
    eventos = full_df[(full_df["ciudad"] == ciudad) & (full_df["temporada"] == temporada)]
    eventos = eventos.dropna(subset=["evento_nombre", "evento_categoria", "evento_desc", "fecha"]).drop_duplicates()
    eventos = eventos[eventos["evento_nombre"] != "sin_evento"].head(3)
    if eventos.empty:
        return None
    return eventos.to_dict(orient="records")

def get_precio_vuelo(origen, destino):
    vuelos = full_df[(full_df["origin_city"] == origen) & (full_df["ciudad"] == destino)]
    return round(vuelos["flight_price"].mean(), 2) if not vuelos.empty else None

def get_hotel(ciudad):
    hoteles = full_df[full_df["ciudad"] == ciudad][["hotel_name", "estimated_price_eur_y", "hotel_type", "distance_to_city_center_km"]]
    hotel = hoteles.dropna().sort_values("estimated_price_eur_y").head(1)
    return hotel.to_dict(orient="records")[0] if not hotel.empty else None

# ========================
# 4. Interfaz de usuario
# ========================
st.title("🌍 Descubre tu próximo destino ideal")

perfil = st.selectbox("🧳 ¿Cómo te defines como viajero?", sorted(full_df["perfil_viajero"].dropna().unique()))
entorno = st.selectbox("🌄 ¿Qué tipo de entorno prefieres?", sorted(full_df["entornos"].dropna().unique()))
clasificacion = st.selectbox("🎯 ¿Qué tipo de experiencia buscas?", sorted(full_df["clasificacion_destino"].dropna().unique()))
temporada = st.selectbox("📆 ¿Temporada del viaje?", sorted(full_df["temporada"].dropna().unique()))
clase = st.selectbox("💺 Clase del vuelo", sorted(full_df["class"].dropna().unique()))
origen = st.selectbox("🛫 Ciudad de origen", sorted(full_df["origin_city"].dropna().unique()))

precio_x = st.slider("💰 Precio estimado del vuelo (€)", 50, 1000, 150)
precio_y = st.slider("🏨 Precio estimado del hotel (€)", 20, 500, 100)
distancia = st.slider("📍 Distancia al centro (km)", 0, 20, 2)

# ========================
# 5. Predicción personalizada
# ========================
if st.button("🔍 Recomiéndame destinos"):
    input_dict = {
        "perfil_viajero_n": perfil,
        "entornos_n": entorno,
        "clasificacion_destino_n": clasificacion,
        "temporada_n": temporada,
        "origin_city": origen,
        "class_n": clase,
        "estimated_price_eur_x": precio_x,
        "estimated_price_eur_y": precio_y,
        "distance_to_city_center_km": distancia
    }

    X_user = construir_input_usuario(input_dict, columnas_modelo)
    probs = model.predict_proba(X_user)[0]
    top_indices = np.argsort(probs)[::-1]

    ciudades_mostradas = 0
    sin_eventos_mostrados = 0
    st.subheader("🎯 Resultados:")

    for idx in top_indices:
        if ciudades_mostradas == 5:
            break

        ciudad_id = model.classes_[idx]
        ciudad_nombre = id_to_ciudad[str(ciudad_id)]

        clima = get_clima_estimado(ciudad_nombre, temporada)
        eventos = get_eventos(ciudad_nombre, temporada)
        vuelo = get_precio_vuelo(origen, ciudad_nombre)
        hotel = get_hotel(ciudad_nombre)

        if None in (clima, vuelo, hotel):
            continue

        if eventos is None:
            if sin_eventos_mostrados >= 1:
                continue
            sin_eventos_mostrados += 1

        ciudades_mostradas += 1
        st.markdown(f"## {ciudades_mostradas}. {ciudad_nombre}")
        st.write("☁️ Clima estimado:", clima)

        if eventos:
            st.write("🎫 Eventos:")
            for e in eventos:
                st.markdown(f"- **{e['evento_nombre']}** ({e['evento_categoria']}): {e['evento_desc']} - `{e['fecha']}`")
        else:
            st.info("🎫 Sin eventos disponibles.")

        st.write("✈️ Vuelo:", vuelo)
        st.write("🏨 Hotel:", hotel)

    if ciudades_mostradas == 0:
        st.warning("⚠️ No se encontraron ciudades con información suficiente.")