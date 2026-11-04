from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NoReturn

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from compat import aplicar_compatibilidad_xgboost
from features import PALABRAS_LUJO, PALABRAS_PRESUPUESTO, PALABRAS_TRANSPORTE, haversine_vec

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / 'models' / 'modelo_final_airbnb.pkl'
RUTA_METRICAS = RAIZ / 'models' / 'metricas_modelo.json'
RUTA_UBICACIONES = RAIZ / 'data' / 'ubicaciones_distritos.csv'

st.set_page_config(page_title='Predicción de precio Airbnb NYC', page_icon='🏠', layout='wide')


def _detener(mensaje: str, exc: Exception) -> NoReturn:
    """Muestra un error explícito en la app y detiene la ejecución."""
    logger.exception(mensaje)
    st.error(mensaje)
    with st.expander('Detalle técnico'):
        st.exception(exc)
    st.stop()
    raise RuntimeError(mensaje)


@st.cache_resource(show_spinner=False)
def _cargar_modelo(ruta: str):
    aplicar_compatibilidad_xgboost()
    return joblib.load(ruta)


def cargar_modelo():
    try:
        return _cargar_modelo(str(RUTA_MODELO))
    except FileNotFoundError as exc:
        _detener(f'No se encontró el modelo en `{RUTA_MODELO}`. Ejecuta `python train_model.py` para generarlo.', exc)
    except Exception as exc:
        _detener(
            'No se pudo cargar el modelo. Revisa que las versiones de '
            '`scikit-learn`, `xgboost` y Python coincidan con las del entrenamiento.',
            exc,
        )


@st.cache_data(show_spinner=False)
def _cargar_ubicaciones(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    requeridas = {'latitude', 'longitude', 'neighbourhood_group', 'neighbourhood'}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas en {ruta}: {", ".join(sorted(faltantes))}')
    return df


@st.cache_data(show_spinner=False)
def _cargar_metricas(ruta: str) -> dict:
    with open(ruta, encoding='utf-8') as f:
        metricas = json.load(f)
    nombre = metricas['mejor_modelo']['nombre']
    fila = next(r for r in metricas['test_modelos_ajustados'] if r['modelo'] == nombre)
    return {
        'medae': float(fila['MedAE']),
        'mae': float(fila['MAE']),
        'q25': float(metricas['rango_referencia_razon']['q25']),
        'q75': float(metricas['rango_referencia_razon']['q75']),
        'modelo': nombre,
    }


def cargar_ubicaciones() -> pd.DataFrame:
    try:
        return _cargar_ubicaciones(str(RUTA_UBICACIONES))
    except Exception as exc:
        _detener(f'No se pudo cargar el archivo de ubicaciones `{RUTA_UBICACIONES}`.', exc)


def cargar_metricas() -> dict:
    try:
        return _cargar_metricas(str(RUTA_METRICAS))
    except Exception as exc:
        _detener(
            f'No se pudo cargar el archivo de métricas `{RUTA_METRICAS}`. '
            'Revisa que exista y tenga el formato esperado.',
            exc,
        )


modelo = cargar_modelo()
ubicaciones = cargar_ubicaciones()
metricas_modelo = cargar_metricas()

st.title('Predicción del precio de Airbnb en Nueva York')
st.write(
    'Estimación con modelo enriquecido: geolocalización con múltiples POIs, '
    'micro-barrio (Target Encoding), análisis semántico del título y ratios de negocio.'
)

# ---------------- Ubicación: mapa, coordenadas manuales o dirección
st.header('Ubicación')
st.write(
    'Haz clic en el mapa para marcar dónde está el alojamiento. El modelo utiliza el '
    '**micro-barrio** y las **distancias a POIs** para refinar la predicción.'
)

if 'ubicacion' not in st.session_state:
    st.session_state['ubicacion'] = [40.7580, -73.9855]
if 'lat_manual' not in st.session_state:
    st.session_state['lat_manual'] = float(st.session_state['ubicacion'][0])
if 'lon_manual' not in st.session_state:
    st.session_state['lon_manual'] = float(st.session_state['ubicacion'][1])

mapa = folium.Map(location=st.session_state['ubicacion'], zoom_start=12)
folium.Marker(st.session_state['ubicacion'], tooltip='Alojamiento').add_to(mapa)
clic = st_folium(mapa, height=450, use_container_width=True, returned_objects=['last_clicked'])

if clic and clic['last_clicked']:
    nueva = [clic['last_clicked']['lat'], clic['last_clicked']['lng']]
    if nueva != st.session_state['ubicacion']:
        st.session_state['ubicacion'] = nueva
        st.session_state['lat_manual'] = float(nueva[0])
        st.session_state['lon_manual'] = float(nueva[1])
        st.rerun()

with st.expander('¿El mapa no carga? Introduce la ubicación manualmente'):
    with st.form('form_coordenadas'):
        c_lat, c_lon = st.columns(2)
        with c_lat:
            st.number_input('Latitud', min_value=-90.0, max_value=90.0, step=0.00001, format='%.5f', key='lat_manual')
        with c_lon:
            st.number_input(
                'Longitud', min_value=-180.0, max_value=180.0, step=0.00001, format='%.5f', key='lon_manual'
            )
        aplicar_coordenadas = st.form_submit_button('Usar estas coordenadas')

    if aplicar_coordenadas:
        st.session_state['ubicacion'] = [float(st.session_state['lat_manual']), float(st.session_state['lon_manual'])]
        st.rerun()

with st.expander('Buscar una dirección en Nueva York'):
    with st.form('form_direccion'):
        direccion = st.text_input(
            'Dirección o lugar',
            placeholder='Times Square, New York',
            help='Se usa el servicio público Nominatim (OpenStreetMap).',
        )
        buscar_direccion = st.form_submit_button('Buscar dirección')

    if buscar_direccion and direccion.strip():
        try:
            from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
            from geopy.geocoders import Nominatim

            geolocalizador = Nominatim(user_agent='airbnb-precio-nyc')
            resultado = geolocalizador.geocode(f'{direccion}, New York, NY', timeout=10)
        except (GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable, OSError):
            logger.warning('No se pudo geocodificar la dirección.', exc_info=True)
            st.warning(
                'No se pudo consultar el servicio de direcciones. '
                'Puedes usar el mapa o introducir las coordenadas manualmente.'
            )
        else:
            if resultado is None:
                st.warning('No se encontró esa dirección. Prueba con una más específica.')
            else:
                st.session_state['ubicacion'] = [resultado.latitude, resultado.longitude]
                st.session_state['lat_manual'] = float(resultado.latitude)
                st.session_state['lon_manual'] = float(resultado.longitude)
                st.rerun()

latitud, longitud = st.session_state['ubicacion']
col1, col2 = st.columns(2)
with col1:
    st.metric('Latitud', f'{latitud:.5f}')
with col2:
    st.metric('Longitud', f'{longitud:.5f}')

fuera_de_nyc = not (40.49 <= latitud <= 40.92 and -74.26 <= longitud <= -73.70)
if fuera_de_nyc:
    st.warning(
        'La ubicación está fuera de la ciudad de Nueva York. '
        'Selecciona un punto dentro de la ciudad para obtener una predicción fiable.'
    )

# El micro-barrio se estima con el centroide del barrio más cercano.
dists = haversine_vec(latitud, longitud, ubicaciones['latitude'].to_numpy(), ubicaciones['longitude'].to_numpy())
idx_cercano = int(np.argmin(dists))
distrito_auto = str(ubicaciones.iloc[idx_cercano]['neighbourhood_group'])
barrio_auto = str(ubicaciones.iloc[idx_cercano]['neighbourhood'])
dist_barrio_km = float(dists[idx_cercano])

st.info(
    f'**Distrito estimado:** {distrito_auto} | '
    f'**Micro-barrio estimado:** {barrio_auto} | '
    f'Distancia al centroide del barrio: {dist_barrio_km:.2f} km'
)
st.caption(
    'La ubicación se estima con el centroide del micro-barrio más cercano, no con un polígono '
    'oficial. Si la estimación no es correcta, puedes ajustar distrito y micro-barrio abajo.'
)

with st.expander('Ajustar distrito y micro-barrio manualmente'):
    usar_ajuste_manual = st.checkbox('Usar distrito y micro-barrio manuales')
    if usar_ajuste_manual:
        grupos = sorted(ubicaciones['neighbourhood_group'].dropna().astype(str).unique())
        idx_grupo = grupos.index(distrito_auto) if distrito_auto in grupos else 0
        distrito_manual = st.selectbox('Distrito', grupos, index=idx_grupo)
        barrios = sorted(
            ubicaciones.loc[ubicaciones['neighbourhood_group'] == distrito_manual, 'neighbourhood']
            .dropna()
            .astype(str)
            .unique()
        )
        idx_barrio = barrios.index(barrio_auto) if barrio_auto in barrios else 0
        barrio_manual = st.selectbox('Micro-barrio', barrios, index=idx_barrio)
        distrito, barrio = distrito_manual, barrio_manual
    else:
        distrito, barrio = distrito_auto, barrio_auto

# ---------------- Anuncio y alojamiento
st.header('Anuncio, alojamiento y anfitrión')

with st.form('form_anuncio'):
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        nombre_anuncio = st.text_input(
            'Título del anuncio',
            value='Cozy apartment near subway with great location',
            help='El modelo analiza palabras clave (lujo, transporte, espacio) para ajustar la estimación.',
        )
        tipo = st.selectbox('Tipo de alojamiento', ['Entire home/apt', 'Private room', 'Shared room'])
        noches_minimas = st.number_input('Noches mínimas', min_value=1, max_value=365, value=2)
        disponibilidad = st.number_input('Días disponibles al año', min_value=0, max_value=365, value=120)
    with c_a2:
        num_resenas = st.number_input('Número de reseñas', min_value=0, value=10)
        resenas_mes = st.number_input('Reseñas por mes', min_value=0.0, value=0.5)
        num_anuncios = st.number_input('Número de anuncios del anfitrión', min_value=1, value=1)
    estimar = st.form_submit_button('Estimar precio', type='primary', disabled=fuera_de_nyc)

st.caption(
    '⚠️ **Aviso regulatorio (NYC):** Desde septiembre de 2023, la *Local Law 18* de Nueva York '
    'restringe los alquileres completos de corta estancia (< 30 días) sin que el anfitrión esté '
    'presente. El dataset utilizado es de 2019, por lo que los precios pueden no reflejar la '
    'normativa actual.'
)

# ---------------- Predicción
st.header('Estimación')
if estimar:
    datos = pd.DataFrame(
        {
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
        }
    )

    try:
        precio_pred = float(modelo.predict(datos)[0])
    except Exception as exc:
        logger.exception('Error al predecir el precio.')
        st.error('No se pudo calcular la estimación con el modelo.')
        with st.expander('Detalle técnico'):
            st.exception(exc)
    else:
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

        st.success(
            f'**Precio sugerido: USD {precio_pred:.0f} por noche** — '
            f'*Rango razonable: USD {p_low:.0f} – USD {p_high:.0f}*'
        )

        # Explicabilidad básica
        st.subheader('¿Por qué este precio?')
        razones = []
        if tipo == 'Entire home/apt':
            razones.append('- **Apartamento/casa completa:** Suele tener la mayor prima respecto a una habitación.')
        elif tipo == 'Private room':
            razones.append(
                '- **Habitación privada:** Precio intermedio, generalmente menor que un alojamiento completo.'
            )
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

        if dist_barrio_km < 2.0:
            razones.append('- **Muy cerca del centroide del barrio:** Mayor accesibilidad -> suele subir el precio.')
        elif dist_barrio_km > 6.0:
            razones.append('- **Lejos del centroide del barrio:** Puede presionar el precio a la baja.')

        if noches_minimas >= 7:
            razones.append('- **Estadía mínima mayor (>=7 noches):** Puede ajustar la tarifa por tipo de estancia.')
        if disponibilidad >= 300:
            razones.append(
                '- **Alta disponibilidad anual:** Puede reflejar mayor rotación o enfoque en alquileres de corta estancia.'
            )

        for razon in razones:
            st.write(razon)

        st.caption(
            'Nota: Esta explicación es heurística para apoyar la decisión. El modelo combina todas '
            'las variables de forma no lineal. El rango razonable corresponde a la mitad central de '
            f'los errores del modelo ({metricas_modelo["modelo"]}) en el 30 % de prueba: en la mitad '
            'de los alojamientos de prueba, el precio real quedó dentro de ese rango.'
        )
