import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

# Compatibilidad multiplataforma (Linux -> Windows) para deserialización de XGBoost
if sys.platform == 'win32':
    try:
        import xgboost.core
        orig_setstate = xgboost.core.Booster.__setstate__

        def _win_setstate(self, state):
            buf = state.get("handle")
            if buf is not None and isinstance(buf, (bytearray, bytes)):
                b = bytes(buf)
                marker = b'rng_stateSL'
                idx = b.find(marker)
                if idx != -1:
                    str_len = int.from_bytes(b[idx + len(marker):idx + len(marker) + 8], 'big')
                    str_start = idx + len(marker) + 8
                    old_str = b[str_start:str_start + str_len].decode('latin1')
                    parts = old_str.strip().split()
                    if len(parts) >= 624 and all(p.isdigit() for p in parts[:10]):
                        import xgboost as xgb
                        reg = xgb.XGBRegressor(n_estimators=1, random_state=42)
                        reg.fit(np.array([[1], [2]]), np.array([1, 2]))
                        win_buf = bytes(reg.get_booster().__getstate__()['handle'])
                        win_idx = win_buf.find(marker)
                        win_len = int.from_bytes(win_buf[win_idx + len(marker):win_idx + len(marker) + 8], 'big')
                        win_str_start = win_idx + len(marker) + 8
                        win_str_bytes = win_buf[win_str_start:win_str_start + win_len]
                        new_buf = b[:idx + len(marker)] + win_len.to_bytes(8, 'big') + win_str_bytes + b[str_start + str_len:]
                        state["handle"] = bytearray(new_buf)
            return orig_setstate(self, state)

        xgboost.core.Booster.__setstate__ = _win_setstate
    except Exception:
        pass

# Importar utilidades de features para consistencia con el pipeline
try:
    from features import haversine_vec, PALABRAS_LUJO, PALABRAS_PRESUPUESTO, PALABRAS_TRANSPORTE
except Exception:
    def haversine_vec(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

    PALABRAS_LUJO = ['luxury', 'lux', 'penthouse', 'exclusive', 'stunning', 'spectacular', 'skyline', 'terrace', 'balcony', 'duplex', 'pool', 'high-end', 'gorgeous', 'views', 'view']
    PALABRAS_PRESUPUESTO = ['cheap', 'budget', 'cozy', 'small', 'simple', 'tiny', 'economical', 'basic']
    PALABRAS_TRANSPORTE = ['subway', 'train', 'metro', 'station', 'steps to', 'min to', 'minutes to', 'walk to']

st.set_page_config(page_title="Predicción de precio Airbnb NYC", page_icon="🏠", layout="wide")
st.title('Predicción del precio de Airbnb en Nueva York')
st.write('Estimación con modelo enriquecido: geolocalización con múltiples POIs, micro-barrio (Target Encoding), análisis semántico del título y ratios de negocio.')


# Cargar el modelo una sola vez
@st.cache_resource
def cargar_modelo():
    ruta = 'models/modelo_final_airbnb.pkl' if os.path.exists('models/modelo_final_airbnb.pkl') else 'modelo_final_airbnb.pkl'
    return joblib.load(ruta)


# Coordenadas y distrito/barrio de los alojamientos del dataset
@st.cache_data
def cargar_ubicaciones():
    ruta = 'data/ubicaciones_distritos.csv' if os.path.exists('data/ubicaciones_distritos.csv') else 'ubicaciones_distritos.csv'
    df = pd.read_csv(ruta)
    if 'neighbourhood' not in df.columns:
        df['neighbourhood'] = 'Unknown'
    return df


# Métricas del modelo final (error mediano y rango de referencia calculados con el 30 % de prueba)
@st.cache_data
def cargar_metricas():
    ruta = 'models/metricas_modelo.json' if os.path.exists('models/metricas_modelo.json') else 'metricas_modelo.json'
    try:
        with open(ruta, encoding='utf-8') as f:
            metricas = json.load(f)
        nombre = metricas['mejor_modelo']['nombre']
        fila = next(r for r in metricas['test_modelos_ajustados'] if r['modelo'] == nombre)
        return {'medae': float(fila['MedAE']), 'mae': float(fila['MAE']),
                'q25': float(metricas['rango_referencia_razon']['q25']),
                'q75': float(metricas['rango_referencia_razon']['q75']),
                'modelo': nombre}
    except Exception:
        return {'medae': 23.0, 'mae': 56.0, 'q25': 0.8, 'q75': 1.25, 'modelo': 'XGBoost'}


modelo = cargar_modelo()
ubicaciones = cargar_ubicaciones()
metricas_modelo = cargar_metricas()

# ---------------- Ubicación: se elige con un clic en el mapa
st.header('1) Ubicación')
st.write('Haz clic en el mapa para marcar dónde está el alojamiento. El modelo utiliza el **micro-barrio** y las **distancias a POIs** para refinar la predicción.')

# Ubicación inicial: Times Square
if 'ubicacion' not in st.session_state:
    st.session_state['ubicacion'] = [40.7580, -73.9855]

mapa = folium.Map(location=st.session_state['ubicacion'], zoom_start=12)
folium.Marker(st.session_state['ubicacion'], tooltip='Alojamiento').add_to(mapa)
clic = st_folium(mapa, height=450, use_container_width=True, returned_objects=['last_clicked'])

if clic and clic['last_clicked']:
    nueva = [clic['last_clicked']['lat'], clic['last_clicked']['lng']]
    if nueva != st.session_state['ubicacion']:
        st.session_state['ubicacion'] = nueva
        st.rerun()

latitud, longitud = st.session_state['ubicacion']
col1, col2 = st.columns(2)
with col1:
    st.metric('Latitud', f'{latitud:.5f}')
with col2:
    st.metric('Longitud', f'{longitud:.5f}')

fuera_de_nyc = not (40.49 <= latitud <= 40.92 and -74.26 <= longitud <= -73.70)
if fuera_de_nyc:
    st.warning('La ubicación está fuera de la ciudad de Nueva York. Selecciona un punto dentro de la ciudad para obtener una predicción fiable.')

# Determinar distrito y micro-barrio más cercano
try:
    dists = haversine_vec(latitud, longitud, ubicaciones['latitude'].values, ubicaciones['longitude'].values)
    idx_cercano = int(np.argmin(dists))
    distrito = str(ubicaciones.iloc[idx_cercano]['neighbourhood_group'])
    barrio = str(ubicaciones.iloc[idx_cercano]['neighbourhood'])
    dist_cercana_km = float(dists[idx_cercano])
    st.info(f'**Neighbourhood Group:** {distrito} | **Neighbourhood (micro-barrio):** {barrio} | Distancia al punto de referencia más cercano: {dist_cercana_km:.2f} km')
except Exception:
    distancias = (ubicaciones['latitude'] - latitud) ** 2 + (ubicaciones['longitude'] - longitud) ** 2
    idx_cercano = int(distancias.idxmin())
    distrito = str(ubicaciones.iloc[idx_cercano]['neighbourhood_group'])
    barrio = str(ubicaciones.iloc[idx_cercano].get('neighbourhood', 'Unknown'))
    dist_cercana_km = float(np.sqrt(distancias.iloc[idx_cercano]) * 111.0)
    st.info(f'**Neighbourhood Group:** {distrito} | **Neighbourhood (micro-barrio):** {barrio}')

# ---------------- Anuncio y alojamiento
st.header('2) Anuncio, alojamiento y anfitrión')
c_a1, c_a2 = st.columns(2)
with c_a1:
    nombre_anuncio = st.text_input('Título del anuncio', value='Cozy apartment near subway with great location', help='El modelo analiza palabras clave (lujo, transporte, espacio) para ajustar la estimación.')
    tipo = st.selectbox('Tipo de alojamiento', ['Entire home/apt', 'Private room', 'Shared room'])
with c_a2:
    noches_minimas = st.number_input('Noches mínimas', min_value=1, max_value=365, value=2)
    disponibilidad = st.number_input('Días disponibles al año', min_value=0, max_value=365, value=120)
    num_resenas = st.number_input('Número de reseñas', min_value=0, value=10)
    resenas_mes = st.number_input('Reseñas por mes', min_value=0.0, value=0.5)
    num_anuncios = st.number_input('Número de anuncios del anfitrión', min_value=1, value=1)

st.caption('⚠️ **Aviso regulatorio (NYC):** Desde septiembre de 2023, la *Local Law 18* de Nueva York restringe los alquileres completos de corta estancia (< 30 días) sin que el anfitrión esté presente. El dataset utilizado es de 2019, por lo que los precios pueden no reflejar la normativa actual.')

# ---------------- Predicción
st.header('3) Estimación')
if st.button('Estimar precio', type='primary', disabled=fuera_de_nyc):
    datos = pd.DataFrame({
        'name': [nombre_anuncio],
        'neighbourhood_group': [distrito],
        'neighbourhood': [barrio],
        'latitude': [latitud],
        'longitude': [longitud],
        'room_type': [tipo],
        'minimum_nights': [noches_minimas],
        'number_of_reviews': [num_resenas],
        'reviews_per_month': [resenas_mes],
        'calculated_host_listings_count': [num_anuncios],
        'availability_365': [disponibilidad],
    })

    try:
        precio_pred = float(modelo.predict(datos)[0])
        precio_pred = max(precio_pred, 10.0)

        # Rango en el que estuvo el precio real de la mitad central de los alojamientos de prueba
        medae_ref = metricas_modelo['medae']
        p_low = max(10.0, precio_pred * metricas_modelo['q25'])
        p_high = precio_pred * metricas_modelo['q75']

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric('Precio sugerido (referencia)', f'USD {precio_pred:.0f} / noche')
        with m2:
            st.metric('Rango razonable', f'USD {p_low:.0f} – USD {p_high:.0f}')
        with m3:
            st.metric('Error mediano típico (referencia)', f'~USD {medae_ref:.0f}')

        st.success(f'**Precio sugerido: USD {precio_pred:.0f} por noche** — *Rango razonable: USD {p_low:.0f} – USD {p_high:.0f}*')

        # Explicabilidad básica
        st.subheader('¿Por qué este precio?')
        razones = []
        if tipo == 'Entire home/apt':
            razones.append('- **Apartamento/casa completa:** Suele tener la mayor prima respecto a una habitación.')
        elif tipo == 'Private room':
            razones.append('- **Habitación privada:** Precio intermedio, generalmente menor que un alojamiento completo.')
        else:
            razones.append('- **Habitación compartida:** Suele ser la opción más económica.')

        if distrito == 'Manhattan':
            razones.append('- **Manhattan:** Uno de los distritos con mayor demanda en NYC.')
        elif distrito == 'Brooklyn':
            razones.append('- **Brooklyn:** Muy demandado con gran variación según el micro-barrio.')
        elif distrito == 'Queens':
            razones.append('- **Queens:** Generalmente más accesible, con buenas conexiones de transporte.')
        else:
            razones.append(f'- **{distrito}:** Influye según ubicación y demanda del barrio.')

        titulo_low = nombre_anuncio.lower()
        if any(p in titulo_low for p in PALABRAS_LUJO):
            razones.append('- **Atributos de lujo detectados en el título:** Puede aumentar la prima (+según mercado).')
        if any(p in titulo_low for p in PALABRAS_TRANSPORTE):
            razones.append('- **Cercanía/transporte mencionado:** Suele ser valorado positivamente en NYC.')
        if any(p in titulo_low for p in PALABRAS_PRESUPUESTO):
            razones.append('- **Términos económicos/cozy:** Pueden orientar hacia un precio más competitivo.')

        if dist_cercana_km < 2.0:
            razones.append('- **Muy cerca de un punto de referencia:** Mayor accesibilidad -> suele subir el precio.')
        elif dist_cercana_km > 6.0:
            razones.append('- **Más alejado del centro:** Suele presionar el precio a la baja.')

        if noches_minimas >= 7:
            razones.append('- **Estadía mínima mayor (>=7 noches):** Puede ajustar la tarifa por tipo de estancia.')
        if disponibilidad >= 300:
            razones.append('- **Alta disponibilidad anual:** Puede reflejar mayor rotación o enfoque en alquileres de corta estancia.')

        for r in razones:
            st.write(r)

        st.caption('Nota: Esta explicación es heurística para apoyar la decisión. El modelo combina todas las variables de forma no lineal. '
                   f'El rango razonable corresponde a la mitad central de los errores del modelo ({metricas_modelo["modelo"]}) '
                   'en el 30 % de prueba: en la mitad de los alojamientos de prueba, el precio real quedó dentro de ese rango.')
    except Exception as e:
        st.error(f'Error al predecir: {e}')
        st.info('Verifica que el archivo models/modelo_final_airbnb.pkl sea compatible con la versión actual de features.py y las dependencias.')
