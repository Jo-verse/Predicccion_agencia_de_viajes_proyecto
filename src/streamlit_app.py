import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ========================
# Configuración y carga
# ========================
@st.cache_data(show_spinner=False)
def cargar_datos():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model = joblib.load(os.path.join(base_dir, 'models', 'model_completo.pkl'))
    le = joblib.load(os.path.join(base_dir, 'models', 'label_encoder.pkl'))
    columnas = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'x_train_columns.csv'), header=None).squeeze().tolist()
    full_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'total_data_240k.csv'))
    with open(os.path.join(base_dir, 'data', 'processed', 'Json', 'ciudad_transformation_rules.json')) as f:
        mapping = json.load(f)
    id_to_ciudad = {str(v): k for k, v in mapping.items()}
    return model, le, columnas, full_df, id_to_ciudad

model, le, columnas_modelo, full_df, id_to_ciudad = cargar_datos()

# ========================
# Funciones auxiliares
# ========================
def construir_input_usuario(valores_dict, columnas_modelo):
    df = pd.DataFrame(columns=columnas_modelo)
    df.loc[0] = 0
    for clave, valor in valores_dict.items():
        col = f"{clave}_{valor}"
        if col in df.columns:
            df.at[0, col] = 1
    for col in ['estimated_price_eur_x', 'estimated_price_eur_y', 'distance_to_city_center_km']:
        if col in valores_dict:
            df.at[0, col] = valores_dict[col]
    return df

def get_clima_estimado(ciudad, temporada):
    clima = full_df[(full_df["ciudad"] == ciudad) & (full_df["temporada"] == temporada)]
    if clima.empty: return None
    datos = clima[["temp_max", "temp_min", "precipitacion", "humedad_actual"]].mean().round(1).to_dict()
    desc = clima["desc_actual"].mode()
    datos["desc_actual"] = desc[0] if not desc.empty else "N/A"
    return datos

def get_eventos(ciudad, temporada):
    eventos = full_df[(full_df["ciudad"] == ciudad) & (full_df["temporada"] == temporada)]
    eventos = eventos.dropna(subset=["evento_nombre", "evento_categoria", "evento_desc", "fecha"]).drop_duplicates()
    eventos = eventos[eventos["evento_nombre"] != "sin_evento"].head(3)
    return eventos.to_dict("records") if not eventos.empty else None

def get_precio_vuelo(origen, destino):
    vuelos = full_df[(full_df["origin_city"] == origen) & (full_df["ciudad"] == destino)]
    if vuelos.empty: return None
    info = vuelos[["flight_price", "flight_duration_hr"]].mean().round(1).to_dict()
    info["airline"] = vuelos["airline"].mode()[0] if "airline" in vuelos else "N/A"
    info["stops"] = vuelos["stops"].mode()[0] if "stops" in vuelos else "N/A"
    info["class"] = vuelos["class"].mode()[0] if "class" in vuelos else "N/A"
    return info

def get_hotel(ciudad):
    hoteles = full_df[full_df["ciudad"] == ciudad].dropna(subset=["hotel_name", "estimated_price_eur_y", "hotel_type", "category", "hotel_type_1", "distance_to_city_center_km"])
    if hoteles.empty: return None
    hotel = hoteles.sort_values("estimated_price_eur_y").head(1)
    return hotel[["hotel_name", "estimated_price_eur_y", "hotel_type", "category", "hotel_type_1", "distance_to_city_center_km"]].to_dict("records")[0]

# ========================
# Interfaz Streamlit
# ========================
st.set_page_config(page_title="Destino Ideal", layout="wide")
st.title("🌍 Encuentra tu Próximo Destino Ideal")
st.write("Explora, sueña y planea tu próxima aventura con nuestras recomendaciones personalizadas.")

# Sidebar
st.sidebar.header("🔍 Filtros de Búsqueda")
perfil = st.sidebar.selectbox("🧳 Perfil de Viajero", sorted(full_df["perfil_viajero"].dropna().unique()))
entorno = st.sidebar.selectbox("🌄 Tipo de Entorno", sorted(full_df["entornos"].dropna().unique()))
clasificacion = st.sidebar.selectbox("🎯 Tipo de Experiencia", sorted(full_df["clasificacion_destino"].dropna().unique()))
temporada = st.sidebar.selectbox("📆 Temporada", sorted(full_df["temporada"].dropna().unique()))
clase = st.sidebar.selectbox("💺 Clase del Vuelo", sorted(full_df["class"].dropna().unique()))
origen = st.sidebar.selectbox("🛫 Ciudad de Origen", sorted(full_df["origin_city"].dropna().unique()))
precio_x = st.sidebar.slider("💸 Precio Estimado Vuelo (€)", 50, 1000, 150)
precio_y = st.sidebar.slider("🏨 Precio Estimado Hotel (€)", 20, 500, 100)
distancia = st.sidebar.slider("📍 Distancia al Centro (km)", 0, 20, 2)

# Recomendaciones
if st.sidebar.button("🔎 Recomiéndame Destinos"):
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

    st.subheader("🎯 Resultados Personalizados:")
    mostradas = 0
    sin_eventos_mostrado = False

    for idx in top_indices:
        if mostradas == 5:
            break

        ciudad_id = model.classes_[idx]
        ciudad_raw = id_to_ciudad[str(ciudad_id)]
        ciudad = ciudad_raw.title()

        clima = get_clima_estimado(ciudad_raw, temporada)
        eventos = get_eventos(ciudad_raw, temporada)
        vuelo = get_precio_vuelo(origen, ciudad_raw)
        hotel = get_hotel(ciudad_raw)

        if None in (clima, vuelo, hotel):
            continue
        if eventos is None and sin_eventos_mostrado:
            continue

        mostradas += 1

        # ✅ Expander con TÍTULO visible cerrado
        with st.expander(f"Recomendación #{mostradas}", expanded=True):
            st.markdown(f"""
                <h2 style='margin-bottom: 0.2rem; font-size: 1.8rem; color: #1f4e79;'>
                    🏙️ {ciudad}</span>
                </h2>
                <hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>
            """, unsafe_allow_html=True)

            # Clima
            st.markdown("### ☁️ Clima Estimado")
            st.markdown(
                f"""
                🔶 **Descripción:** {clima.get('desc_actual', 'N/A')}  
                🌡️ **Máx:** {clima.get('temp_max', 'N/A')} °C — 🌡️ **Mín:** {clima.get('temp_min', 'N/A')} °C  
                🌧️ **Precipitación:** {clima.get('precipitacion', 'N/A')} mm  
                💧 **Humedad:** {clima.get('humedad_actual', 'N/A')}%
                """
            )

            # Eventos
            st.markdown("### 🎫 Eventos")
            if eventos:
                for e in eventos:
                    st.markdown(
                        f"""- 🗓️ **{e['evento_nombre']}** ({e['evento_categoria']})  
                        _{e['evento_desc']}_ — `{e['fecha']}`"""
                    )
            else:
                sin_eventos_mostrado = True
                st.info("ℹ️ No hay eventos disponibles para esta ciudad.")

            # Vuelo y Hotel en columnas
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✈️ Vuelo")
                st.markdown(
                    f"""
                    💶 **Precio:** {vuelo.get('flight_price', 'N/A')} €  
                    ⏱️ **Duración:** {vuelo.get('flight_duration_hr', 'N/A')} h  
                    🛫 **Aerolínea:** {vuelo.get('airline', 'N/A')}  
                    🪑 **Clase:** {vuelo.get('class', 'N/A')}  
                    🔁 **Escalas:** {vuelo.get('stops', 'N/A')}
                    """
                )

            with col2:
                st.markdown("### 🏨 Hotel")
                st.markdown(
                    f"""
                    🏠 **Nombre:** {hotel.get('hotel_name', 'N/A')}  
                    💰 **Precio por noche:** {hotel.get('estimated_price_eur_y', 'N/A')} €  
                    🏷️ **Tipo:** {hotel.get('hotel_type', 'N/A')}  
                    🏅 **Categoría:** {hotel.get('category', 'N/A')}  
                    📝 **Descripción:** {hotel.get('hotel_type_1', 'N/A')}  
                    📍 **Distancia al centro:** {hotel.get('distance_to_city_center_km', 'N/A')} km
                    """
                )

    if mostradas == 0:
        st.warning("⚠️ No se encontraron ciudades con información suficiente.")