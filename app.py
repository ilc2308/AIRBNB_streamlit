"""App de Streamlit: precio de referencia por noche de un alojamiento de Airbnb en Nueva York."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NoReturn

import altair as alt
import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from compat import aplicar_compatibilidad_xgboost
from features import PALABRAS_ESPACIO, PALABRAS_LUJO, PALABRAS_PRESUPUESTO, PALABRAS_TRANSPORTE, POIS, haversine_vec

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / 'models' / 'modelo_final_airbnb.pkl'
RUTA_METRICAS = RAIZ / 'models' / 'metricas_modelo.json'
RUTA_UBICACIONES = RAIZ / 'data' / 'ubicaciones_distritos.csv'
RUTA_LIMPIO = RAIZ / 'data' / 'Airbnb_NYC_2019_limpio.csv'
RUTA_TITULOS = RAIZ / 'data' / 'titulos_ejemplo.csv'
TITULO_DEFECTO = 'Cozy apartment near subway with great location'

NOMBRES_POI = {
    'times_square': 'Times Square',
    'central_park': 'Central Park',
    'fidi': 'Financial District',
    'brooklyn_bridge': 'Brooklyn Bridge',
    'jfk': 'Aeropuerto JFK',
    'lga': 'Aeropuerto LaGuardia',
}
TIPOS = {
    'Entire home/apt': 'Casa o apto. completo',
    'Private room': 'Habitación privada',
    'Shared room': 'Habitación compartida',
}
FAMILIAS = [
    ('Lujo', PALABRAS_LUJO, 'sube'),
    ('Espacio', PALABRAS_ESPACIO, 'sube'),
    ('Transporte', PALABRAS_TRANSPORTE, 'neutro'),
    ('Economía', PALABRAS_PRESUPUESTO, 'baja'),
]

st.set_page_config(page_title='Precio Airbnb NYC', page_icon='🗽', layout='wide')

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, input, textarea, button { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .sg { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 4.6rem; max-width: 1240px; }
.hero { background: radial-gradient(1200px 400px at 85% -10%, #E0584A55, transparent 60%), linear-gradient(135deg, #14213D 0%, #1F3260 100%);
        color: #F7F5F0; border-radius: 22px; padding: 34px 38px 30px; margin-bottom: 8px; position: relative; overflow: hidden; }
.hero .eyebrow { color: #F2A58E; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; font-size: .78rem; }
.hero h1 { color: #F7F5F0; font-size: 2.5rem; line-height: 1.08; margin: 8px 0 10px; padding: 0; }
.hero p { color: #C9D1DE; font-size: 1.05rem; max-width: 720px; margin: 0; }
.hero .skyline { position: absolute; right: 26px; bottom: 0; opacity: .18; width: 340px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.chip { background: #ffffff14; border: 1px solid #ffffff26; border-radius: 999px; padding: 6px 14px; font-size: .88rem; color: #F7F5F0; }
.chip b { color: #F2A58E; }
.step { display: flex; align-items: center; gap: 12px; margin: 26px 0 6px; }
.step .num { background: #E0584A; color: #fff; font-family: 'Space Grotesk', 'DM Sans', sans-serif; font-weight: 700; width: 34px; height: 34px; border-radius: 50%;
             display: grid; place-items: center; font-size: 1rem; }
.step h3 { margin: 0; padding: 0; font-size: 1.45rem; }
.step span.sub { color: #6B7280; font-size: .95rem; }
.loc { background: #fff; border: 1px solid #E3DED3; border-radius: 16px; padding: 18px 20px; }
.loc .k { color: #6B7280; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.loc .v { font-family: 'Space Grotesk', 'DM Sans', sans-serif; font-size: 1.35rem; font-weight: 700; color: #14213D; line-height: 1.2; }
.dist { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px dashed #E3DED3; font-size: .95rem; }
.dist:last-child { border-bottom: 0; }
.tag { display: inline-block; border-radius: 999px; padding: 3px 11px; margin: 3px 6px 3px 0; font-size: .85rem; font-weight: 500; }
.tag.sube { background: #FBEDE8; color: #B4432B; } .tag.baja { background: #E4F1EF; color: #1F6F63; } .tag.neutro { background: #EEF1F7; color: #33415C; }
.price { background: linear-gradient(135deg, #14213D, #1F3260); color: #F7F5F0; border-radius: 22px; padding: 28px 32px; }
.price .k { color: #F2A58E; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; font-size: .78rem; }
.price .big { font-family: 'Space Grotesk', 'DM Sans', sans-serif; font-size: 3.6rem; font-weight: 700; line-height: 1; margin: 10px 0 4px; }
.price .big small { font-size: 1.1rem; color: #C9D1DE; font-weight: 500; }
.bar { position: relative; height: 10px; border-radius: 999px; background: #ffffff22; margin: 26px 0 8px; }
.bar .fill { position: absolute; height: 10px; border-radius: 999px; background: linear-gradient(90deg, #F2A58E, #E0584A); }
.bar .dot { position: absolute; top: -6px; width: 22px; height: 22px; border-radius: 50%; background: #fff; border: 4px solid #E0584A; transform: translateX(-50%); }
.bar-labels { display: flex; justify-content: space-between; color: #C9D1DE; font-size: .88rem; }
.card { background: #fff; border: 1px solid #E3DED3; border-radius: 16px; padding: 18px 20px; height: 100%; }
.card .k { color: #6B7280; font-size: .82rem; }
.card .v { font-family: 'Space Grotesk', 'DM Sans', sans-serif; font-size: 1.7rem; font-weight: 700; color: #14213D; }
.card .s { color: #6B7280; font-size: .85rem; }
.up { color: #B4432B; } .down { color: #1F6F63; }
.foot { color: #6B7280; font-size: .82rem; text-align: center; margin: 40px 0 10px; }
div[data-testid="stButton"] button[kind="primary"] { border-radius: 14px; height: 3.2rem; font-size: 1.05rem; font-weight: 700; }
</style>""")


# ---------------- Carga de recursos (una sola vez)
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


@st.cache_data(show_spinner=False)
def _cargar_ubicaciones(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltantes = {'latitude', 'longitude', 'neighbourhood_group', 'neighbourhood'} - set(df.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas en {ruta}: {", ".join(sorted(faltantes))}')
    return df


@st.cache_data(show_spinner=False)
def _cargar_metricas(ruta: str) -> dict:
    with open(ruta, encoding='utf-8') as f:
        m = json.load(f)
    nombre = m['mejor_modelo']['nombre']
    fila = next(r for r in m['test_modelos_ajustados'] if r['modelo'] == nombre)
    return {
        'medae': float(fila['MedAE']),
        'mae': float(fila['MAE']),
        'r2': float(fila['R2_log']),
        'q25': float(m['rango_referencia_razon']['q25']),
        'q75': float(m['rango_referencia_razon']['q75']),
        'modelo': nombre.replace(' - ', ' · '),
    }


@st.cache_data(show_spinner=False)
def _cargar_comparables(ruta: str) -> pd.DataFrame:
    # Precios del conjunto limpio (salida de la preparación de datos), para comparar con el barrio
    try:
        return pd.read_csv(ruta, usecols=['neighbourhood_group', 'neighbourhood', 'room_type', 'price'])
    except (OSError, ValueError):
        logger.warning('No se pudo leer el dataset limpio; la app no mostrará comparables.', exc_info=True)
        return pd.DataFrame(columns=['neighbourhood_group', 'neighbourhood', 'room_type', 'price'])


@st.cache_data(show_spinner=False)
def _cargar_titulos(ruta: str) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta)
    except (OSError, ValueError):
        return pd.DataFrame({'titulo': [TITULO_DEFECTO], 'grupo': ['Neutral']})


try:
    modelo = _cargar_modelo(str(RUTA_MODELO))
except FileNotFoundError as exc:
    _detener(f'No se encontró el modelo en `{RUTA_MODELO}`. Ejecuta `python train_model.py` para generarlo.', exc)
except Exception as exc:
    _detener(
        'No se pudo cargar el modelo. Revisa que las versiones de `scikit-learn`, `xgboost` y Python '
        'coincidan con las del entrenamiento.',
        exc,
    )
try:
    ubicaciones = _cargar_ubicaciones(str(RUTA_UBICACIONES))
except Exception as exc:
    _detener(f'No se pudo cargar el archivo de ubicaciones `{RUTA_UBICACIONES}`.', exc)
try:
    metricas = _cargar_metricas(str(RUTA_METRICAS))
except Exception as exc:
    _detener(f'No se pudo cargar el archivo de métricas `{RUTA_METRICAS}`.', exc)
comparables = _cargar_comparables(str(RUTA_LIMPIO))
titulos = _cargar_titulos(str(RUTA_TITULOS))


def palabras_detectadas(titulo):
    t = titulo.lower()
    return [(familia, p, efecto) for familia, lista, efecto in FAMILIAS for p in lista if p in t]


def predecir(registro):
    return max(float(modelo.predict(pd.DataFrame([registro]))[0]), 10.0)


# ---------------- Barra lateral: el modelo en breve
with st.sidebar:
    st.markdown('### 🗽 Sobre el modelo')
    st.markdown(f'**{metricas["modelo"]}**, elegido entre 7 modelos con validación cruzada, ANOVA y Tukey.')
    s1, s2 = st.columns(2)
    s1.metric('MAE (USD)', f'{metricas["mae"]:.1f}')
    s2.metric('MedAE (USD)', f'{metricas["medae"]:.1f}')
    s1.metric('R² (log)', f'{metricas["r2"]:.3f}')
    s2.metric('Características', '34')
    st.markdown('**Cómo se construyó (CRISP-DM)**')
    st.markdown(
        '1. Preparación: datos limpios sin errores ni nulos\n2. Minería: 34 características (distancias, título, micro-barrio)\n'
        '3. Modelado: 70 % entrenamiento, 30 % prueba\n4. Despliegue: pipeline completo en esta app'
    )
    st.caption(
        '⚠️ **Local Law 18 (NYC):** desde septiembre de 2023 se restringen los alquileres completos de menos de 30 noches '
        'sin el anfitrión presente. Los datos son de 2019: úsalo como referencia, no como precio de mercado actual.'
    )

# ---------------- Encabezado
st.html(f"""
<div class="hero">
  <svg class="skyline" viewBox="0 0 340 120" fill="#F7F5F0"><path d="M0 120V80h18V60h14v20h10V40h16v80h8V70h12V30l8-14 8 14v90h10V58h18v62h8V46h6V24h4V8h4v16h4v22h6v74h10V64h20v56h8V52l12-10 12 10v68h10V76h16v44h12V60h18v60h10V84h14v36z"/></svg>
  <div class="eyebrow">Minería de datos · Airbnb Nueva York</div>
  <h1>¿Cuánto cobrar por noche en Nueva York?</h1>
  <p>Marca el alojamiento en el mapa, describe el anuncio y el modelo estima un precio de referencia con su rango, comparado con los anuncios de tu barrio.</p>
  <div class="chips">
    <span class="chip">Error típico <b>±{metricas['medae']:.0f} USD</b></span>
    <span class="chip">MAE <b>{metricas['mae']:.1f} USD</b> en datos no vistos</span>
    <span class="chip"><b>34</b> características</span>
    <span class="chip"><b>221</b> micro-barrios</span>
  </div>
</div>""")

# ---------------- 1. Ubicación
st.html(
    '<div class="step"><div class="num">1</div><div><h3>Ubicación</h3><span class="sub">Haz clic en el mapa donde está el alojamiento</span></div></div>'
)

if 'ubicacion' not in st.session_state:
    st.session_state['ubicacion'] = [40.7580, -73.9855]
st.session_state.setdefault('lat_manual', float(st.session_state['ubicacion'][0]))
st.session_state.setdefault('lon_manual', float(st.session_state['ubicacion'][1]))


def mover_a(lat: float, lon: float) -> None:
    st.session_state['ubicacion'] = [round(float(lat), 5), round(float(lon), 5)]
    st.session_state['lat_manual'] = float(lat)
    st.session_state['lon_manual'] = float(lon)


latitud, longitud = st.session_state['ubicacion']

# El micro-barrio se estima con el centroide del barrio más cercano
dists = haversine_vec(latitud, longitud, ubicaciones['latitude'].to_numpy(), ubicaciones['longitude'].to_numpy())
idx = int(np.argmin(dists))
distrito_auto = str(ubicaciones.iloc[idx]['neighbourhood_group'])
barrio_auto = str(ubicaciones.iloc[idx]['neighbourhood'])
fuera_de_nyc = not (40.49 <= latitud <= 40.92 and -74.26 <= longitud <= -73.70) or float(dists[idx]) > 5.0
dist_poi = {k: float(haversine_vec(latitud, longitud, la, lo)) for k, (la, lo) in POIS.items()}

col_mapa, col_info = st.columns([3, 2], gap='large')
with col_mapa:
    mapa = folium.Map(location=[latitud, longitud], zoom_start=12, control_scale=True)
    for clave, (la, lo) in POIS.items():
        folium.CircleMarker(
            [la, lo],
            radius=6,
            color='#14213D',
            weight=2,
            fill=True,
            fill_color='#F7F5F0',
            fill_opacity=1,
            tooltip=NOMBRES_POI[clave],
        ).add_to(mapa)
    poi_cercano = min((k for k in POIS if k not in ('jfk', 'lga')), key=lambda k: dist_poi[k])
    folium.PolyLine([[latitud, longitud], list(POIS[poi_cercano])], color='#E0584A', weight=2, dash_array='6 6').add_to(
        mapa
    )
    folium.Marker(
        [latitud, longitud], tooltip='Tu alojamiento', icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(mapa)
    clic = st_folium(mapa, height=430, use_container_width=True, returned_objects=['last_clicked'], key='mapa')
    if clic and clic.get('last_clicked'):
        nueva = [round(clic['last_clicked']['lat'], 5), round(clic['last_clicked']['lng'], 5)]
        if nueva != st.session_state['ubicacion']:
            mover_a(*nueva)
            st.rerun()

with col_info:
    tarjeta = st.empty()

    with st.expander('¿El mapa no carga? Escribe las coordenadas'), st.form('form_coordenadas'):
        c_lat, c_lon = st.columns(2)
        c_lat.number_input('Latitud', min_value=-90.0, max_value=90.0, step=0.00001, format='%.5f', key='lat_manual')
        c_lon.number_input('Longitud', min_value=-180.0, max_value=180.0, step=0.00001, format='%.5f', key='lon_manual')
        if st.form_submit_button('Usar estas coordenadas'):
            st.session_state['ubicacion'] = [
                round(float(st.session_state['lat_manual']), 5),
                round(float(st.session_state['lon_manual']), 5),
            ]
            st.rerun()

    with st.expander('Buscar una dirección en Nueva York'):
        with st.form('form_direccion'):
            direccion = st.text_input(
                'Dirección o lugar',
                placeholder='Times Square, New York',
                help='Se usa el servicio público Nominatim (OpenStreetMap).',
            )
            buscar = st.form_submit_button('Buscar dirección')
        if buscar and direccion.strip():
            try:
                from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
                from geopy.geocoders import Nominatim

                resultado = Nominatim(user_agent='airbnb-precio-nyc').geocode(f'{direccion}, New York, NY', timeout=10)
            except (ImportError, GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable, OSError):
                logger.warning('No se pudo geocodificar la dirección.', exc_info=True)
                st.warning('No se pudo consultar el servicio de direcciones. Usa el mapa o escribe las coordenadas.')
            else:
                if resultado is None:
                    st.warning('No se encontró esa dirección. Prueba con una más específica.')
                else:
                    mover_a(resultado.latitude, resultado.longitude)
                    st.rerun()

    with st.expander('Corregir distrito y micro-barrio'):
        st.caption('Se estiman con el centroide del micro-barrio más cercano, no con un polígono oficial.')
        if st.checkbox('Elegirlos a mano', key='ajuste_manual'):
            grupos = sorted(ubicaciones['neighbourhood_group'].dropna().astype(str).unique())
            distrito = st.selectbox(
                'Distrito',
                grupos,
                index=grupos.index(distrito_auto) if distrito_auto in grupos else 0,
                key='distrito_manual',
            )
            barrios = sorted(
                ubicaciones.loc[ubicaciones['neighbourhood_group'] == distrito, 'neighbourhood']
                .dropna()
                .astype(str)
                .unique()
            )
            barrio = st.selectbox(
                'Micro-barrio',
                barrios,
                index=barrios.index(barrio_auto) if barrio_auto in barrios else 0,
                key='barrio_manual',
            )
        else:
            distrito, barrio = distrito_auto, barrio_auto

    filas_poi = ''.join(
        f'<div class="dist"><span>{NOMBRES_POI[k]}</span><b>{dist_poi[k]:.1f} km</b></div>'
        for k in ['times_square', 'central_park', 'fidi', 'brooklyn_bridge']
    )
    aeropuerto = min(('jfk', 'lga'), key=lambda k: dist_poi[k])
    marca = ' <span style="font-size:.8rem;color:#B4432B">(elegido a mano)</span>' if barrio != barrio_auto else ''
    tarjeta.html(f"""
    <div class="loc">
      <div class="k">Distrito</div><div class="v">{distrito}</div>
      <div class="k" style="margin-top:10px">Micro-barrio</div><div class="v">{barrio}{marca}</div>
      <div class="s" style="color:#6B7280;font-size:.85rem;margin:6px 0 12px">{latitud:.5f}, {longitud:.5f}</div>
      {filas_poi}
      <div class="dist"><span>✈ {NOMBRES_POI[aeropuerto]}</span><b>{dist_poi[aeropuerto]:.1f} km</b></div>
    </div>""")

if fuera_de_nyc:
    st.warning(
        'El punto está fuera de las zonas con anuncios en Nueva York. '
        'Marca un lugar dentro de la ciudad para estimar el precio.'
    )

# ---------------- 2. El anuncio
st.html(
    '<div class="step"><div class="num">2</div><div><h3>El anuncio</h3><span class="sub">El modelo lee el título y las condiciones de la estancia</span></div></div>'
)

with st.container(border=True):
    modo = st.radio(
        'Título del anuncio',
        ['Elegir un ejemplo', 'Escribir el mío'],
        horizontal=True,
        key='modo_titulo',
        help='El modelo busca palabras de lujo, espacio, transporte y economía en el título.',
    )
    if modo == 'Elegir un ejemplo':
        opciones = list(titulos['titulo'])
        grupos_titulo = dict(zip(titulos['titulo'], titulos['grupo'], strict=True))
        defecto = TITULO_DEFECTO
        nombre_anuncio = st.selectbox(
            'Título de ejemplo',
            opciones,
            index=opciones.index(defecto) if defecto in opciones else 0,
            format_func=lambda t: f'{t}  ·  {grupos_titulo.get(t, "")}',
            key='titulo_ejemplo',
            label_visibility='collapsed',
        )
    else:
        nombre_anuncio = st.text_input(
            'Escribe el título (en inglés, como en Airbnb)',
            value='Cozy apartment near subway with great location',
            key='titulo',
            max_chars=120,
        )

    encontradas = palabras_detectadas(nombre_anuncio)
    if encontradas:
        st.html('<div>' + ''.join(f'<span class="tag {e}">{f}: {p}</span>' for f, p, e in encontradas) + '</div>')
    else:
        st.caption('El título no tiene palabras clave: el modelo lo trata como un anuncio neutro.')

    tipo = (
        st.segmented_control(
            'Tipo de alojamiento',
            list(TIPOS),
            default='Entire home/apt',
            format_func=lambda t: TIPOS[t],
            key='tipo',
            selection_mode='single',
        )
        or 'Entire home/apt'
    )

    c1, c2 = st.columns(2, gap='large')
    with c1:
        noches_minimas = st.number_input('Noches mínimas', min_value=1, max_value=365, value=2, key='noches')
        disponibilidad = st.slider(
            'Días disponibles al año', min_value=0, max_value=365, value=120, key='disponibilidad'
        )
    with c2:
        num_resenas = st.number_input('Número de reseñas', min_value=0, value=10, key='resenas')
        resenas_mes = st.number_input(
            'Reseñas por mes', min_value=0.0, value=0.5, step=0.1, format='%.2f', key='resenas_mes'
        )
        num_anuncios = st.number_input(
            'Anuncios del anfitrión',
            min_value=1,
            value=1,
            key='anuncios',
            help='Más de 1 indica un anfitrión profesional.',
        )

# ---------------- 3. Estimación
st.html(
    '<div class="step"><div class="num">3</div><div><h3>Estimación</h3><span class="sub">Precio de referencia por noche, en dólares</span></div></div>'
)

registro = {
    'name': nombre_anuncio,
    'neighbourhood_group': distrito,
    'neighbourhood': barrio,
    'latitude': latitud,
    'longitude': longitud,
    'room_type': tipo,
    'minimum_nights': noches_minimas,
    'number_of_reviews': num_resenas,
    'reviews_per_month': resenas_mes,
    'calculated_host_listings_count': num_anuncios,
    'availability_365': disponibilidad,
}

if st.button('Estimar precio', type='primary', disabled=fuera_de_nyc, width='stretch', key='estimar'):
    try:
        precio = predecir(registro)
        p_low, p_high = max(10.0, precio * metricas['q25']), precio * metricas['q75']

        # Anuncios comparables del conjunto limpio: mismo micro-barrio y tipo (o mismo distrito si son pocos)
        mismo = comparables[(comparables['neighbourhood'] == barrio) & (comparables['room_type'] == tipo)]
        zona = f'{barrio}'
        if len(mismo) < 8:
            mismo = comparables[(comparables['neighbourhood_group'] == distrito) & (comparables['room_type'] == tipo)]
            zona = f'{distrito}'
        mediana_zona = float(mismo['price'].median()) if len(mismo) else float('nan')

        izquierda, derecha = st.columns([5, 4], gap='large')
        with izquierda:
            escala = max(p_high * 1.25, mediana_zona * 1.1 if mediana_zona == mediana_zona else 0)
            st.html(f"""
            <div class="price">
              <div class="k">Precio sugerido</div>
              <div class="big">USD {precio:,.0f} <small>/ noche</small></div>
              <div style="color:#C9D1DE">{TIPOS[tipo]} en {barrio}, {distrito}</div>
              <div class="bar"><div class="fill" style="left:{p_low / escala * 100:.1f}%;width:{(p_high - p_low) / escala * 100:.1f}%"></div>
                   <div class="dot" style="left:{precio / escala * 100:.1f}%"></div></div>
              <div class="bar-labels"><span>Rango de referencia</span><span><b style="color:#F7F5F0">USD {p_low:,.0f} – {p_high:,.0f}</b></span></div>
            </div>""")
            st.session_state['ultima_estimacion'] = {
                'precio': round(precio),
                'rango': (round(p_low), round(p_high)),
                'distrito': distrito,
                'barrio': barrio,
                'mediana_zona': mediana_zona,
            }
        with derecha:
            m1, m2 = st.columns(2)
            if mediana_zona == mediana_zona:
                dif = (precio / mediana_zona - 1) * 100
                m1.metric(
                    f'Mediana en {zona}', f'USD {mediana_zona:,.0f}', f'Tu precio {dif:+.0f} %', delta_color='off'
                )
                m2.metric('Anuncios comparables', f'{len(mismo):,}'.replace(',', ' '))
            m1.metric('Error típico del modelo', f'± USD {metricas["medae"]:.0f}')
            m2.metric('Precio por semana', f'USD {precio * 7:,.0f}')
            if len(mismo) >= 8:
                datos_hist = mismo[mismo['price'] <= mismo['price'].quantile(0.97)]
                base = (
                    alt.Chart(datos_hist)
                    .mark_bar(color='#C9D1DE', cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X('price:Q', bin=alt.Bin(maxbins=25), title=f'Precio por noche en {zona} (USD)'),
                        y=alt.Y('count():Q', title='Anuncios'),
                    )
                )
                regla = alt.Chart(pd.DataFrame({'p': [precio]})).mark_rule(color='#E0584A', size=3).encode(x='p:Q')
                st.altair_chart((base + regla).properties(height=170), width='stretch')

        # ---------------- ¿Qué pasaría si...? Escenarios calculados con el mismo modelo
        st.html(
            '<div class="step"><div class="num">✦</div><div><h3>¿Qué pasaría si…?</h3><span class="sub">El mismo alojamiento con un cambio, calculado por el modelo</span></div></div>'
        )
        escenarios = []
        for otro in TIPOS:
            if otro != tipo:
                escenarios.append((f'Fuera {TIPOS[otro].lower()}', {**registro, 'room_type': otro}))
        if not any(f == 'Lujo' for f, _, _ in encontradas):
            escenarios.append(
                (
                    'El título dijera “luxury … with skyline view”',
                    {**registro, 'name': f'Luxury {nombre_anuncio} with skyline view'},
                )
            )
        else:
            escenarios.append(('El título no tuviera palabras clave', {**registro, 'name': 'Apartment in New York'}))
        if noches_minimas < 30:
            escenarios.append(('Cumpliera la Local Law 18 (mínimo 30 noches)', {**registro, 'minimum_nights': 30}))
        escenarios.append(
            (
                'Tuviera el doble de reseñas',
                {
                    **registro,
                    'number_of_reviews': max(2 * num_resenas, 2),
                    'reviews_per_month': max(2 * resenas_mes, 0.2),
                },
            )
        )

        cols = st.columns(len(escenarios))
        for col, (texto, reg) in zip(cols, escenarios, strict=True):
            nuevo = predecir(reg)
            cambio = (nuevo / precio - 1) * 100
            if abs(cambio) < 1:
                clase, flecha = '', '≈'
            else:
                clase, flecha = ('up', '▲') if cambio > 0 else ('down', '▼')
            col.html(
                f'<div class="card"><div class="k">Si {texto[0].lower() + texto[1:]}</div>'
                f'<div class="v">USD {nuevo:,.0f}</div><div class="s {clase}">{flecha} {abs(cambio):.0f} % frente a tu estimación</div></div>'
            )

        st.caption(
            f'El rango de referencia es la mitad central de los errores del modelo ({metricas["modelo"]}) en el 30 % de prueba: '
            'en la mitad de esos alojamientos, el precio real quedó dentro de él. Los comparables vienen del conjunto de datos limpio de 2019.'
        )
    except Exception as e:
        st.error(f'Error al predecir: {e}')
        st.info(
            'Verifica que models/modelo_final_airbnb.pkl sea compatible con features.py y con las versiones de requirements.txt.'
        )
else:
    st.caption('Ajusta la ubicación y el anuncio, y presiona **Estimar precio**.')

st.html(
    '<div class="foot">Proyecto de Minería de Datos · Metodología CRISP-DM · Datos: Inside Airbnb, Nueva York 2019 · '
    'Precio de referencia, no constituye una tasación.</div>'
)
