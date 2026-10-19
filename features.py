# Funciones de preparación de datos e ingeniería de características.
# Este archivo lo usan el notebook de modelado y la aplicación de Streamlit.
import numpy as np

# Variables que usa el modelo
COLUMNAS_MODELO = ['neighbourhood_group', 'latitude', 'longitude', 'room_type',
                   'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                   'calculated_host_listings_count', 'availability_365']

# Variables de conteo con distribución muy sesgada
COLUMNAS_SESGADAS = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                     'calculated_host_listings_count']


def distancia_centro_km(lat, lon):
    # Distancia en km hasta Times Square (fórmula de haversine)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(40.7580), np.radians(-73.9855)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def preparar_datos(df):
    # Mismas operaciones del notebook de preparación de datos
    df = df[COLUMNAS_MODELO].copy()                                # selección de variables
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)    # nulos: sin reseñas -> 0
    df['minimum_nights'] = df['minimum_nights'].clip(1, 365)       # regla de calidad
    df['availability_365'] = df['availability_365'].clip(0, 365)   # regla de calidad
    return df


def crear_variables(df):
    # Ingeniería de características
    df = df.copy()
    df['dist_centro_km'] = distancia_centro_km(df['latitude'], df['longitude'])
    for col in COLUMNAS_SESGADAS:
        df[col] = np.log1p(df[col])
    return df
