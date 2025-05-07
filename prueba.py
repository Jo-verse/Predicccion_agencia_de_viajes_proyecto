import streamlit as st
import pandas as pd
import pickle

# Carga del modelo
model = pickle.load(open('models/spam_detector.sav', 'rb'))  # Reemplaza con la ruta de tu modelo

st.title("🌆 Predicción de Clima por Ciudad")
st.markdown("Predice condiciones climáticas basadas en datos de ciudades.")

# Carga de datos
@st.cache
def load_data():
    data = pd.read_csv('ciudades_con_clima.csv')  # Asegúrate de que la ruta sea correcta
    return data

data = load_data()

# Selección de ciudad
ciudad = st.selectbox("Selecciona una ciudad", data['ciudad'].unique())

# Filtrado de datos para la ciudad seleccionada
datos_ciudad = data[data['ciudad'] == ciudad]

# Mostrar datos de la ciudad
st.write(f"Datos actuales para {ciudad}:")
st.dataframe(datos_ciudad)

# Preparación de datos para el modelo
# Asegúrate de que las columnas coincidan con las que espera tu modelo
input_data = datos_ciudad.drop(columns=['ciudad'])  # Ajusta según sea necesario

# Predicción
if st.button("Predecir"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Condiciones favorables (probabilidad: {prob:.2f})")
    else:
        st.error(f"❌ Condiciones desfavorables (probabilidad: {prob:.2f})")
