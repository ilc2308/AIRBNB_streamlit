import joblib
import pandas as pd
import streamlit as st

st.title('Predicción del precio de Airbnb en Nueva York')
st.write('Ingresa las características del alojamiento para estimar su precio por noche (datos de 2019).')

# Cargar el modelo final (pipeline con la preparación de datos incluida)
modelo = joblib.load('modelo_final_airbnb.pkl')

st.header('Ubicación')
distrito = st.selectbox('Distrito', ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
latitud = st.number_input('Latitud', min_value=40.49, max_value=40.92, value=40.7580, format='%.4f')
longitud = st.number_input('Longitud', min_value=-74.26, max_value=-73.70, value=-73.9855, format='%.4f')
st.map(pd.DataFrame({'lat': [latitud], 'lon': [longitud]}))

st.header('Alojamiento y anfitrión')
tipo = st.selectbox('Tipo de alojamiento', ['Entire home/apt', 'Private room', 'Shared room'])
noches_minimas = st.number_input('Noches mínimas', min_value=1, max_value=365, value=2)
disponibilidad = st.number_input('Días disponibles al año', min_value=0, max_value=365, value=120)
num_resenas = st.number_input('Número de reseñas', min_value=0, value=10)
resenas_mes = st.number_input('Reseñas por mes', min_value=0.0, value=0.5)
num_anuncios = st.number_input('Número de anuncios del anfitrión', min_value=1, value=1)

if st.button('Estimar precio'):
    datos = pd.DataFrame({
        'neighbourhood_group': [distrito],
        'latitude': [latitud],
        'longitude': [longitud],
        'room_type': [tipo],
        'minimum_nights': [noches_minimas],
        'number_of_reviews': [num_resenas],
        'reviews_per_month': [resenas_mes],
        'calculated_host_listings_count': [num_anuncios],
        'availability_365': [disponibilidad],
    })
    precio = modelo.predict(datos)[0]
    st.success(f'Precio estimado: USD {precio:.0f} por noche')
    st.write('El error típico del modelo es de unos 23 USD; úsalo como precio de referencia.')
