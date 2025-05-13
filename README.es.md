## 🧳 Descripción del caso de uso: 🧭 Predicción Inteligente de Destinos de Viaje

En el competitivo sector del turismo, comprender y anticiparse a las necesidades de los clientes es clave para mejorar la experiencia del usuario, optimizar las campañas de marketing y aumentar la conversión de ventas. Este proyecto tiene como objetivo construir una herramienta web para agencias de viajes que recomienda destinos personalizados utilizando aprendizaje automático con un modelo XGBoost y un formulario interactivo.


### 🎯 Objetivo del proyecto

El objetivo principal es desarrollar un modelo de machine learning capaz de **predecir si un cliente completará una reserva de viaje** basándose en variables como:

- Edad, género y nacionalidad del cliente.
- Canal por el que realiza la reserva (web, agencia física, teléfono...).
- Número de acompañantes, destino preferido, tipo de paquete turístico.
- Historial de interacciones anteriores.
- Fechas y duración del viaje, etc.

Este tipo de predicción puede ayudar a la agencia a:

- **Identificar clientes potenciales** con mayor probabilidad de conversión.
- **Diseñar ofertas personalizadas** según el perfil del cliente.
- **Optimizar recursos de atención al cliente**, priorizando a los leads más cualificados.
- **Reducir costes en campañas publicitarias** segmentando mejor el público objetivo.

### 🧠 Beneficio para el negocio

Al integrar este modelo en los procesos de la agencia, se consigue una **mayor eficiencia operativa y una mejora en la experiencia del cliente**, lo que en última instancia se traduce en **más reservas confirmadas** y **mayores ingresos**. Además, permite a la agencia ser más proactiva y estratégica en sus decisiones de negocio.


### 🚀 Tecnologías Utilizadas

- **Python** (Pandas, NumPy, XGBoost, Scikit-learn)
- **SQL** (para consultas de datos estructurados)
- **Streamlit** (interfaz web)
- **Jupyter Notebook** (exploración, análisis y modelado)
- **Git & GitHub**

### 🤖 Modelo de Predicción

El motor de recomendación está basado en un modelo **XGBoost Classifier**, elegido por su alta precisión y rendimiento en tareas de clasificación complejas. El modelo fue entrenado con características como:

- Edad y presupuesto del cliente
- Intereses (naturaleza, cultura, gastronomía, etc.)
- Tipo de viaje preferido (aventura, relax, familiar, lujo)
- Historial de destinos visitados
- Temporada del año

El objetivo es predecir el destino más adecuado para cada perfil de usuario.


### ⚙️ Instalación y Uso

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
streamlit run app/main.py
