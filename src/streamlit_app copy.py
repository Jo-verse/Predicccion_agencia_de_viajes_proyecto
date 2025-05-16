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
st.title(":tierra_áfrica: Encuentra tu Próximo Destino Ideal")
st.write("Explora, sueña y planea tu próxima aventura con nuestras recomendaciones personalizadas.")
st.sidebar.header(":lupa: Filtros de Búsqueda")
perfil = st.sidebar.selectbox(":equipaje: Perfil de Viajero", sorted(full_df["perfil_viajero"].dropna().unique()))
entorno = st.sidebar.selectbox(":amanecer_sobre_las_montañas: Tipo de Entorno", sorted(full_df["entornos"].dropna().unique()))
clasificacion = st.sidebar.selectbox(":dardo: Tipo de Experiencia", sorted(full_df["clasificacion_destino"].dropna().unique()))
temporada = st.sidebar.selectbox(":calendario: Temporada", sorted(full_df["temporada"].dropna().unique()))
clase = st.sidebar.selectbox(":asiento: Clase del Vuelo", sorted(full_df["class"].dropna().unique()))
origen = st.sidebar.selectbox(":avión_despegando: Ciudad de Origen", sorted(full_df["origin_city"].dropna().unique()))
precio_x = st.sidebar.slider(":dinero_con_alas: Precio Estimado Vuelo (€)", 50, 1000, 150)
precio_y = st.sidebar.slider(":hotel: Precio Estimado Hotel (€)", 20, 500, 100)
distancia = st.sidebar.slider(":tachuela_redonda: Distancia al Centro (km)", 0, 20, 2)
if st.sidebar.button(":lupa_derecha: Recomiéndame Destinos"):
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
    st.subheader(":dardo: Resultados Personalizados:")
    mostradas = 0
    sin_eventos_mostrado = False
    for idx in top_indices:
        if mostradas == 5:
            break
        ciudad_id = model.classes_[idx]
        ciudad = id_to_ciudad[str(ciudad_id)]
        clima = get_clima_estimado(ciudad, temporada)
        eventos = get_eventos(ciudad, temporada)
        vuelo = get_precio_vuelo(origen, ciudad)
        hotel = get_hotel(ciudad)
        if None in (clima, vuelo, hotel):
            continue
        if eventos is None and sin_eventos_mostrado:
            continue
        mostradas += 1
        st.markdown(f"---\n## :puesta_de_sol_urbana: {ciudad} — Recomendación #{mostradas}")
        st.markdown("### :nube: Clima Estimado")
        st.markdown(f"- **Descripción:** {clima.get('desc_actual', 'N/A')}")
        st.markdown(f"- **Temp Máx:** {clima.get('temp_max', 'N/A')} °C")
        st.markdown(f"- **Temp Mín:** {clima.get('temp_min', 'N/A')} °C")
        st.markdown(f"- **Precipitación:** {clima.get('precipitacion', 'N/A')} mm")
        st.markdown(f"- **Humedad:** {clima.get('humedad_actual', 'N/A')}%")
        st.markdown("### :tique: Eventos")
        if eventos:
            for e in eventos:
                st.markdown(f"- **{e['evento_nombre']}** ({e['evento_categoria']}): {e['evento_desc']} — `{e['fecha']}`")
        else:
            sin_eventos_mostrado = True
            st.info(":fuente_de_información: No hay eventos disponibles para esta ciudad.")
        st.markdown("### :avión: Detalles del Vuelo")
        st.markdown(f"- **Precio medio:** {vuelo.get('flight_price', 'N/A')} €")
        st.markdown(f"- **Duración estimada:** {vuelo.get('flight_duration_hr', 'N/A')} h")
        st.markdown(f"- **Aerolínea:** {vuelo.get('airline', 'N/A')}")
        st.markdown(f"- **Clase:** {vuelo.get('class', 'N/A')}")
        st.markdown(f"- **Escalas:** {vuelo.get('stops', 'N/A')}")
        st.markdown("### :hotel: Hotel Recomendado")
        st.markdown(f"- **Nombre:** {hotel.get('hotel_name', 'N/A')}")
        st.markdown(f"- **Precio por noche:** {hotel.get('estimated_price_eur_y', 'N/A')} €")
        st.markdown(f"- **Tipo:** {hotel.get('hotel_type', 'N/A')}")
        st.markdown(f"- **Categoría:** {hotel.get('category', 'N/A')}")
        st.markdown(f"- **Descripción:** {hotel.get('hotel_type_1', 'N/A')}")
        st.markdown(f"- **Distancia al centro:** {hotel.get('distance_to_city_center_km', 'N/A')} km")
    if mostradas == 0:
        st.warning(":advertencia: No se encontraron ciudades con información suficiente.")