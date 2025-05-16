import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json
import os

# ========================
# Configuración de rutas
# ========================
@st.cache_data(show_spinner=False)
def cargar_archivos():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data', 'processed')
    json_dir = os.path.join(data_dir, 'Json')
    
    try:
        model = joblib.load(os.path.join(model_dir, 'model_completo.pkl'))
        le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        columnas_modelo = pd.read_csv(os.path.join(data_dir, 'x_train_columns.csv'), header=None).squeeze().tolist()
        full_df = pd.read_csv(os.path.join(data_dir, 'total_data_240k.csv'))
        with open(os.path.join(json_dir, 'ciudad_transformation_rules.json')) as f:
            ciudad_mapping = json.load(f)
        id_to_ciudad = {str(v): k for k, v in ciudad_mapping.items()}
    except Exception as e:
        st.error(f"Error cargando archivos necesarios: {e}")
        st.stop()

    return model, le, columnas_modelo, full_df, id_to_ciudad

model, le, columnas_modelo, full_df, id_to_ciudad = cargar_archivos()

# ========================
# Función para construir input del usuario
# ========================
def construir_input_usuario(valores_dict, columnas_modelo):
    df = pd.DataFrame(columns=columnas_modelo)
    df.loc[0] = 0  # Inicializa con ceros
    for clave, valor in valores_dict.items():
        col = f"{clave}_{valor}"
        if col in df.columns:
            df.at[0, col] = 1
    for col in ['estimated_price_eur_x', 'estimated_price_eur_y', 'distance_to_city_center_km']:
        if col in valores_dict:
            df.at[0, col] = valores_dict[col]
    return df

# ========================
# Funciones de enriquecimiento
# ========================
def get_clima_estimado(ciudad, temporada):
    clima = full_df[(full_df["ciudad"] == ciudad) & (full_df["temporada"] == temporada)]
    if clima.empty:
        return None
    datos = clima[['temp_max', 'temp_min', 'precipitacion', 'humedad_actual']].mean(numeric_only=True).round(1).to_dict()
    datos['desc_actual'] = clima['desc_actual'].mode()[0] if 'desc_actual' in clima.columns and not clima['desc_actual'].mode().empty else "N/A"
    return datos

def get_eventos(ciudad, temporada):
    eventos = full_df[(full_df['ciudad'] == ciudad) & (full_df['temporada'] == temporada)].dropna(subset=['evento_nombre', 'evento_categoria', 'evento_desc', 'fecha']).drop_duplicates()
    if eventos.empty:
        return None
    return eventos.head(3).to_dict(orient='records')

def get_precio_vuelo(origen, destino):
    vuelos = full_df[(full_df['origin_city'] == origen) & (full_df['ciudad'] == destino)]
    if vuelos.empty:
        return None
    info = vuelos[['flight_price', 'flight_duration_hr']].dropna().mean(numeric_only=True).round(1).to_dict()
    info['airline'] = vuelos['airline'].mode()[0] if 'airline' in vuelos and not vuelos['airline'].mode().empty else "N/A"
    info['stops'] = vuelos['stops'].mode()[0] if 'stops' in vuelos and not vuelos['stops'].mode().empty else "N/A"
    info['class'] = vuelos['class'].mode()[0] if 'class' in vuelos and not vuelos['class'].mode().empty else "N/A"
    return info

def get_hotel(ciudad):
    hoteles = full_df[full_df['ciudad'] == ciudad].dropna(subset=['hotel_name', 'estimated_price_eur_y', 'hotel_type', 'category', 'hotel_type_1', 'distance_to_city_center_km'])
    if hoteles.empty:
        return None
    hotel = hoteles.sort_values('estimated_price_eur_y').head(1)
    return hotel[['hotel_name', 'estimated_price_eur_y', 'hotel_type', 'category', 'hotel_type_1', 'distance_to_city_center_km']].to_dict(orient='records')[0]

# ========================
# Interfaz de usuario
# ========================
st.title("🌍 Encuentra tu próximo destino ideal")
perfil = st.selectbox("🧳 ¿Perfil de viajero?", sorted(full_df["perfil_viajero"].dropna().unique()))
entorno = st.selectbox("🌄 Tipo de entorno", sorted(full_df["entornos"].dropna().unique()))
clasificacion = st.selectbox("🎯 Tipo de experiencia", sorted(full_df["clasificacion_destino"].dropna().unique()))
temporada = st.selectbox("📆 Temporada", sorted(full_df["temporada"].dropna().unique()))
clase = st.selectbox("💺 Clase del vuelo", sorted(full_df["class"].dropna().unique()))
origen = st.selectbox("🛫 Ciudad de origen", sorted(full_df["origin_city"].dropna().unique()))
precio_x = st.slider("💰 Precio estimado vuelo (€)", 50, 1000, 150)
precio_y = st.slider("🏨 Precio estimado hotel (€)", 20, 500, 100)
distancia = st.slider("📍 Distancia al centro (km)", 0, 20, 2)

# Botón de predicción
if st.button("🔍 Recomiéndame destinos"):
    input_dict = {
        'perfil_viajero_n': perfil,
        'entornos_n': entorno,
        'clasificacion_destino_n': clasificacion,
        'temporada_n': temporada,
        'origin_city': origen,
        'class_n': clase,
        'estimated_price_eur_x': precio_x,
        'estimated_price_eur_y': precio_y,
        'distance_to_city_center_km': distancia
    }
    X_user = construir_input_usuario(input_dict, columnas_modelo)
    try:
        probs = model.predict_proba(X_user)[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.stop()
    st.success("🔍 Recomendaciones generadas con éxito")