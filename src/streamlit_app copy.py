import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
# ========================
# Estilo de la app
# ========================
st.set_page_config(page_title="Destino Ideal", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #0F1117;
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ========================
# Cargar archivos
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
        st.error(f":x: Error cargando archivos: {e}")
        st.stop()
    return model, le, columnas_modelo, full_df, id_to_ciudad
model, le, columnas_modelo, full_df, id_to_ciudad = cargar_archivos()
# ========================
# Construir input del usuario
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
st.title(":destellos: Encuentra tu Próximo Destino Ideal :avión:")
st.write("Explora, sueña y planea tu próxima aventura con nuestras recomendaciones personalizadas.")
st.sidebar.header(":dardo: Filtros de Búsqueda")
perfil = st.sidebar.selectbox(":equipaje: Perfil de Viajero", sorted(full_df["perfil_viajero"].dropna().unique())).strip().title()
entorno = st.sidebar.selectbox(":amanecer_sobre_las_montañas: Tipo de Entorno", sorted(full_df["entornos"].dropna().unique())).strip().title()
clasificacion = st.sidebar.selectbox(":etiqueta: Tipo de Experiencia", sorted(full_df["clasificacion_destino"].dropna().unique())).strip().title()
temporada = st.sidebar.selectbox(":calendario: Temporada", sorted(full_df["temporada"].dropna().unique())).strip().title()
clase = st.sidebar.selectbox(":asiento: Clase del Vuelo", sorted(full_df["class"].dropna().unique())).strip().title()
origen = st.sidebar.selectbox(":avión_despegando: Ciudad de Origen", sorted(full_df["origin_city"].dropna().unique())).strip().title()
precio_x = st.sidebar.slider(":bolsa_de_dinero: Precio Estimado Vuelo (€)", 50, 1000, 150)
precio_y = st.sidebar.slider(":hotel: Precio Estimado Hotel (€)", 20, 500, 100)
distancia = st.sidebar.slider(":tachuela_redonda: Distancia al Centro (km)", 0, 20, 2)
if st.sidebar.button(":lupa: Recomiéndame Destinos"):
    input_dict = {
        'perfil_viajero_n': perfil,
        'entornos_n': entorno,
        'clasificacion_destino_n': clasificacion,
        'temporada_n': temporada,
        'origin_city_n': origen,
        'class_n': clase,
        'estimated_price_eur_x': precio_x,
        'estimated_price_eur_y': precio_y,
        'distance_to_city_center_km': distancia
    }
    X_user = construir_input_usuario(input_dict, columnas_modelo)
    try:
        probs = model.predict_proba(X_user)[0]
        destinos_recomendados = np.argsort(probs)[-5:][::-1]
        st.success(":lupa: Recomendaciones generadas con éxito")
        for destino_id in destinos_recomendados:
            nombre = id_to_ciudad[str(destino_id)].capitalize()
            st.subheader(f":estrella: {nombre}")
            st.write(f":tachuela_redonda: Ubicación: {nombre}")
            st.write(f":dinero_con_alas: Precio medio del vuelo: {precio_x}€")
            st.write(f":hotel: Precio medio del hotel: {precio_y}€")
            st.write(f":regla: Distancia al centro: {distancia} km")
            st.divider()
    except Exception as e:
        st.error(f":advertencia: Error en la predicción: {e}")