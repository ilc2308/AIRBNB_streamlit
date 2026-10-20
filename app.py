import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.title('Predicción del precio de Airbnb en Nueva York')
st.write('Ingresa las características del alojamiento para estimar su precio por noche (datos de 2019).')


# Cargar el modelo una sola vez (el caché evita recargarlo en cada clic)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_final_airbnb.pkl')


modelo = cargar_modelo()

# ---------------- Ubicación: se elige con un clic en el mapa
st.header('Ubicación')
st.write('Haz clic en el mapa para marcar dónde está el alojamiento.')

# Ubicación inicial: Times Square
if 'ubicacion' not in st.session_state:
    st.session_state['ubicacion'] = [40.7580, -73.9855]

mapa = folium.Map(location=st.session_state['ubicacion'], zoom_start=12)
folium.Marker(st.session_state['ubicacion'], tooltip='Alojamiento').add_to(mapa)
clic = st_folium(mapa, height=450, use_container_width=True, returned_objects=['last_clicked'])

# Si el usuario hizo clic en un punto nuevo, se guarda y se vuelve a dibujar el mapa con el marcador
if clic and clic['last_clicked']:
    nueva = [clic['last_clicked']['lat'], clic['last_clicked']['lng']]
    if nueva != st.session_state['ubicacion']:
        st.session_state['ubicacion'] = nueva
        st.rerun()

latitud, longitud = st.session_state['ubicacion']
st.write(f'Ubicación seleccionada: latitud {latitud:.4f}, longitud {longitud:.4f}')

fuera_de_nyc = not (40.49 <= latitud <= 40.92 and -74.26 <= longitud <= -73.70)
if fuera_de_nyc:
    st.warning('La ubicación está fuera de la ciudad de Nueva York. Selecciona un punto dentro de la ciudad.')

distrito = st.selectbox('Distrito', ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])

# ---------------- Alojamiento y anfitrión
st.header('Alojamiento y anfitrión')
tipo = st.selectbox('Tipo de alojamiento', ['Entire home/apt', 'Private room', 'Shared room'])
noches_minimas = st.number_input('Noches mínimas', min_value=1, max_value=365, value=2)
disponibilidad = st.number_input('Días disponibles al año', min_value=0, max_value=365, value=120)
num_resenas = st.number_input('Número de reseñas', min_value=0, value=10)
resenas_mes = st.number_input('Reseñas por mes', min_value=0.0, value=0.5)
num_anuncios = st.number_input('Número de anuncios del anfitrión', min_value=1, value=1)

if st.button('Estimar precio', disabled=fuera_de_nyc):
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
