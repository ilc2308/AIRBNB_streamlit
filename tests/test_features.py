import numpy as np
import pandas as pd
import pytest

from features import COLUMNAS_MODELO, COLUMNAS_SESGADAS, distancia_centro_km, preparar_datos, crear_variables


def test_distancia_centro_km_times_square():
    # Coordenadas exactas de Times Square
    lat_ts, lon_ts = 40.7580, -73.9855
    dist = distancia_centro_km(lat_ts, lon_ts)
    assert np.isclose(dist, 0.0, atol=1e-3), "La distancia de Times Square a sí mismo debe ser ~0 km"


def test_distancia_centro_km_distinto_punto():
    # Coordenadas de Brooklyn Bridge (~6.5 km de Times Square)
    lat_bb, lon_bb = 40.7061, -73.9969
    dist = distancia_centro_km(lat_bb, lon_bb)
    assert 5.0 < dist < 8.0, f"La distancia esperada a Brooklyn Bridge es ~6-7 km, obtenida: {dist}"


def test_preparar_datos():
    df_prueba = pd.DataFrame({
        'id': [1, 2],
        'name': ['Alojamiento 1', 'Alojamiento 2'],
        'neighbourhood_group': ['Manhattan', 'Brooklyn'],
        'latitude': [40.7580, 40.7000],
        'longitude': [-73.9855, -73.9500],
        'room_type': ['Entire home/apt', 'Private room'],
        'minimum_nights': [0, 500],             # Debe ajustarse a [1, 365]
        'number_of_reviews': [10, 0],
        'reviews_per_month': [np.nan, 2.5],     # NaN debe ser imputado con 0
        'calculated_host_listings_count': [1, 2],
        'availability_365': [-10, 400],         # Debe ajustarse a [0, 365]
    })

    df_resultado = preparar_datos(df_prueba)

    # 1. Columnas extra deben eliminarse
    assert set(df_resultado.columns) == set(COLUMNAS_MODELO)
    assert 'id' not in df_resultado.columns
    assert 'name' not in df_resultado.columns

    # 2. Imputación de nulos
    assert df_resultado['reviews_per_month'].iloc[0] == 0.0

    # 3. Reglas de calidad (clip)
    assert df_resultado['minimum_nights'].iloc[0] == 1
    assert df_resultado['minimum_nights'].iloc[1] == 365
    assert df_resultado['availability_365'].iloc[0] == 0
    assert df_resultado['availability_365'].iloc[1] == 365


def test_crear_variables():
    df_prueba = pd.DataFrame({
        'neighbourhood_group': ['Manhattan'],
        'latitude': [40.7580],
        'longitude': [-73.9855],
        'room_type': ['Entire home/apt'],
        'minimum_nights': [2],
        'number_of_reviews': [10],
        'reviews_per_month': [1.0],
        'calculated_host_listings_count': [1],
        'availability_365': [120],
    })

    df_feat = crear_variables(df_prueba)

    # Variable distancia agregada
    assert 'dist_centro_km' in df_feat.columns
    assert df_feat['dist_centro_km'].iloc[0] >= 0

    # Transformación log1p aplicada en sesgadas
    for col in COLUMNAS_SESGADAS:
        esperado = np.log1p(df_prueba[col].iloc[0])
        assert np.isclose(df_feat[col].iloc[0], esperado)
