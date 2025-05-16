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
# Interfaz de usuario
# ========================
st.title("✨ Encuentra tu Próximo Destino Ideal ✈️")
st.write("Explora, sueña y planea tu próxima aventura con nuestras recomendaciones personalizadas.")

st.sidebar.header("🎯 Filtros de Búsqueda")
perfil = st.sidebar.selectbox("🧳 Perfil de Viajero", sorted(full_df["perfil_viajero"].dropna().unique()))
entorno = st.sidebar.selectbox("🌄 Tipo de Entorno", sorted(full_df["entornos"].dropna().unique()))
clasificacion = st.sidebar.selectbox("🎯 Tipo de Experiencia", sorted(full_df["clasificacion_destino"].dropna().unique()))
temporada = st.sidebar.selectbox("📆 Temporada", sorted(full_df["temporada"].dropna().unique()))
clase = st.sidebar.selectbox("💺 Clase del Vuelo", sorted(full_df["class"].dropna().unique()))
origen = st.sidebar.selectbox("🛫 Ciudad de Origen", sorted(full_df["origin_city"].dropna().unique()))
precio_x = st.sidebar.slider("💰 Precio Estimado Vuelo (€)", 50, 1000, 150)
precio_y = st.sidebar.slider("🏨 Precio Estimado Hotel (€)", 20, 500, 100)
distancia = st.sidebar.slider("📍 Distancia al Centro (km)", 0, 20, 2)

if st.sidebar.button("🔍 Recomiéndame Destinos"):
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
        destinos_recomendados = np.argsort(probs)[-5:][::-1]
        st.success("🔍 Recomendaciones generadas con éxito")
        for destino_id in destinos_recomendados:
            destino = id_to_ciudad[str(destino_id)]
            st.subheader(f"🌟 {destino}")
            st.write(f"📍 Ubicación: {destino}")
            st.write(f"💸 Precio medio del vuelo: {precio_x}€")
            st.write(f"🏨 Precio medio del hotel: {precio_y}€")
            st.write(f"📏 Distancia al centro: {distancia} km")
            st.divider()
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
