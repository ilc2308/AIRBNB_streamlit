# Funciones de preparación de datos e ingeniería de características para el modelo final de Airbnb NYC.
# Este archivo es utilizado por los notebooks de modelado, la suite de pruebas unitarias y la aplicación Streamlit.
import numpy as np

# ---------------- Puntos de Interés Geográficos (POIs) en NYC
POIS = {
    'times_square': (40.7580, -73.9855),
    'central_park': (40.7829, -73.9654),
    'fidi': (40.7075, -74.0090),
    'brooklyn_bridge': (40.7061, -73.9969),
    'jfk': (40.6413, -73.7781),
    'lga': (40.7769, -73.8740),
}

# ---------------- Diccionarios Semánticos de Texto para Títulos de Anuncios
PALABRAS_LUJO = [
    'luxury',
    'lux',
    'penthouse',
    'exclusive',
    'stunning',
    'spectacular',
    'skyline',
    'terrace',
    'balcony',
    'duplex',
    'pool',
    'high-end',
    'gorgeous',
    'views',
    'view',
]
PALABRAS_PRESUPUESTO = ['cheap', 'budget', 'cozy', 'small', 'simple', 'tiny', 'economical', 'basic']
PALABRAS_TRANSPORTE = ['subway', 'train', 'metro', 'station', 'steps to', 'min to', 'minutes to', 'walk to']
PALABRAS_ESPACIO = ['spacious', 'huge', 'large', 'big', 'massive', 'loft']

# ---------------- Definición de Columnas
COLUMNAS_ENTRADA = [
    'name',
    'neighbourhood_group',
    'neighbourhood',
    'latitude',
    'longitude',
    'room_type',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365',
]

# Compatibilidad con pipeline base
COLUMNAS_MODELO = COLUMNAS_ENTRADA

COLUMNAS_SESGADAS = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']

COLUMNAS_NUMERICAS = [
    'latitude',
    'longitude',
    'dist_times_square_km',
    'dist_central_park_km',
    'dist_fidi_km',
    'dist_brooklyn_bridge_km',
    'dist_jfk_km',
    'dist_lga_km',
    'dist_min_aeropuerto_km',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365',
    'feat_luxury_count',
    'feat_has_luxury',
    'feat_budget_count',
    'feat_has_budget',
    'feat_transit_count',
    'feat_spacious_count',
    'name_length',
    'name_word_count',
    'availability_ratio',
    'is_full_year_available',
    'is_host_multi_listing',
]
COLUMNAS_CATEGORICAS_OHE = ['neighbourhood_group', 'room_type']
COLUMNAS_CATEGORICAS_TARGET = ['neighbourhood']


def haversine_vec(lat1, lon1, lat2, lon2):
    """Calcula la distancia esférica (en km) entre coordenadas mediante la fórmula de Haversine."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def distancia_centro_km(lat, lon):
    """Distancia en km hasta Times Square (Midtown Manhattan). Mantiene retrocompatibilidad."""
    return haversine_vec(lat, lon, 40.7580, -73.9855)


def validar_columnas_entrada(datos):
    """Comprueba que el DataFrame tenga todas las columnas que espera el pipeline.

    Falla de forma explícita en lugar de inventar columnas con valores por defecto,
    para que un esquema de entrada incorrecto no pase desapercibido en producción.
    """
    faltantes = [col for col in COLUMNAS_ENTRADA if col not in datos.columns]
    if faltantes:
        raise ValueError('Faltan columnas obligatorias para el modelo: ' + ', '.join(faltantes))


def preparar_datos(datos):
    """Aplica las reglas de limpieza y calidad de datos necesarias antes de transformar variables."""
    validar_columnas_entrada(datos)
    df_out = datos.copy()

    df_out['name'] = df_out['name'].fillna('').astype(str)
    df_out['neighbourhood'] = df_out['neighbourhood'].fillna('Unknown').astype(str)
    df_out['reviews_per_month'] = df_out['reviews_per_month'].fillna(0.0)
    df_out['minimum_nights'] = df_out['minimum_nights'].clip(1, 365)
    df_out['availability_365'] = df_out['availability_365'].clip(0, 365)

    # Filtrar solo columnas de modelo para descartar id, host_name, etc. si vienen en df_out
    return df_out[COLUMNAS_ENTRADA]


def crear_variables(datos):
    """Genera las nuevas variables de minería geoespacial, de texto, ratios y log-transformaciones."""
    df_out = datos.copy()

    # Asegurar name si no viniese
    if 'name' not in df_out.columns:
        df_out['name'] = ''

    # 1. Distancias a múltiples Puntos de Interés (POIs)
    for poi_name, (lat_poi, lon_poi) in POIS.items():
        df_out[f'dist_{poi_name}_km'] = haversine_vec(df_out['latitude'], df_out['longitude'], lat_poi, lon_poi)

    df_out['dist_min_aeropuerto_km'] = df_out[['dist_jfk_km', 'dist_lga_km']].min(axis=1)
    df_out['dist_centro_km'] = df_out['dist_times_square_km']

    # 2. Extracción de señales de texto sobre el título (name)
    name_clean = df_out['name'].astype(str).str.lower()

    df_out['feat_luxury_count'] = name_clean.apply(lambda t: sum(1 for p in PALABRAS_LUJO if p in t))
    df_out['feat_has_luxury'] = (df_out['feat_luxury_count'] > 0).astype(int)

    df_out['feat_budget_count'] = name_clean.apply(lambda t: sum(1 for p in PALABRAS_PRESUPUESTO if p in t))
    df_out['feat_has_budget'] = (df_out['feat_budget_count'] > 0).astype(int)

    df_out['feat_transit_count'] = name_clean.apply(lambda t: sum(1 for p in PALABRAS_TRANSPORTE if p in t))
    df_out['feat_near_transit'] = (df_out['feat_transit_count'] > 0).astype(int)

    df_out['feat_spacious_count'] = name_clean.apply(lambda t: sum(1 for p in PALABRAS_ESPACIO if p in t))

    df_out['name_length'] = df_out['name'].astype(str).apply(len)
    df_out['name_word_count'] = name_clean.apply(lambda t: len(t.split()))

    # 3. Ratios económicos y de anfitrión
    df_out['availability_ratio'] = df_out['availability_365'] / 365.0
    df_out['is_full_year_available'] = (df_out['availability_365'] == 365).astype(int)
    df_out['is_host_multi_listing'] = (df_out['calculated_host_listings_count'] > 1).astype(int)

    # 4. Transformaciones log1p en variables de conteo sesgadas
    for col in COLUMNAS_SESGADAS:
        if col in df_out.columns:
            df_out[col] = np.log1p(df_out[col])

    return df_out


def construir_preprocesamiento():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

    return ColumnTransformer(
        [
            ('num', StandardScaler(), COLUMNAS_NUMERICAS),
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), COLUMNAS_CATEGORICAS_OHE),
            ('cat_te', TargetEncoder(smooth='auto', cv=5, random_state=42), COLUMNAS_CATEGORICAS_TARGET),
        ],
        verbose_feature_names_out=False,
    )


def construir_pipeline(regresor=None):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    if regresor is None:
        from xgboost import XGBRegressor

        regresor = XGBRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42,
        )

    pipe = Pipeline(
        [
            ('preparacion', FunctionTransformer(preparar_datos)),
            ('ingenieria', FunctionTransformer(crear_variables)),
            ('preprocesamiento', construir_preprocesamiento()),
            ('modelo', regresor),
        ]
    )
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
