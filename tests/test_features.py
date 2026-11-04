import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from features import (
    COLUMNAS_CATEGORICAS_OHE,
    COLUMNAS_CATEGORICAS_TARGET,
    COLUMNAS_MODELO,
    COLUMNAS_NUMERICAS,
    COLUMNAS_SESGADAS,
    POIS,
    construir_pipeline,
    construir_preprocesamiento,
    crear_variables,
    distancia_centro_km,
    preparar_datos,
)


def test_distancia_centro_km_times_square():
    # Coordenadas exactas de Times Square
    lat_ts, lon_ts = 40.7580, -73.9855
    dist = distancia_centro_km(lat_ts, lon_ts)
    assert np.isclose(dist, 0.0, atol=1e-3), 'La distancia de Times Square a sí mismo debe ser ~0 km'


def test_distancia_centro_km_distinto_punto():
    # Coordenadas de Brooklyn Bridge (~6.5 km de Times Square)
    lat_bb, lon_bb = 40.7061, -73.9969
    dist = distancia_centro_km(lat_bb, lon_bb)
    assert 5.0 < dist < 8.0, f'La distancia esperada a Brooklyn Bridge es ~6-7 km, obtenida: {dist}'


def test_preparar_datos():
    df_prueba = pd.DataFrame(
        {
            'id': [1, 2],
            'name': ['Luxury Penthouse in NYC', 'Cozy studio near subway'],
            'neighbourhood_group': ['Manhattan', 'Brooklyn'],
            'neighbourhood': ['Midtown', 'Williamsburg'],
            'latitude': [40.7580, 40.7000],
            'longitude': [-73.9855, -73.9500],
            'room_type': ['Entire home/apt', 'Private room'],
            'minimum_nights': [0, 500],  # Debe ajustarse a [1, 365]
            'number_of_reviews': [10, 0],
            'reviews_per_month': [np.nan, 2.5],  # NaN debe ser imputado con 0
            'calculated_host_listings_count': [1, 2],
            'availability_365': [-10, 400],  # Debe ajustarse a [0, 365]
        }
    )

    df_resultado = preparar_datos(df_prueba)

    # 1. Columnas extra deben eliminarse y columnas de modelo conservarse
    assert set(df_resultado.columns) == set(COLUMNAS_MODELO)
    assert 'id' not in df_resultado.columns
    assert 'name' in df_resultado.columns
    assert 'neighbourhood' in df_resultado.columns

    # 2. Imputación de nulos
    assert df_resultado['reviews_per_month'].iloc[0] == 0.0

    # 3. Reglas de calidad (clip)
    assert df_resultado['minimum_nights'].iloc[0] == 1
    assert df_resultado['minimum_nights'].iloc[1] == 365
    assert df_resultado['availability_365'].iloc[0] == 0
    assert df_resultado['availability_365'].iloc[1] == 365


def test_crear_variables():
    df_prueba = pd.DataFrame(
        {
            'name': ['Luxury Penthouse with Skyline View and Terrace'],
            'neighbourhood_group': ['Manhattan'],
            'neighbourhood': ['Midtown'],
            'latitude': [40.7580],
            'longitude': [-73.9855],
            'room_type': ['Entire home/apt'],
            'minimum_nights': [2],
            'number_of_reviews': [10],
            'reviews_per_month': [1.0],
            'calculated_host_listings_count': [1],
            'availability_365': [120],
        }
    )

    df_feat = crear_variables(df_prueba)

    # Variables de distancia agregadas
    for poi in POIS:
        assert f'dist_{poi}_km' in df_feat.columns
    assert 'dist_min_aeropuerto_km' in df_feat.columns
    assert 'dist_centro_km' in df_feat.columns

    # Banderas semánticas de texto (debe detectar lujo en 'Luxury Penthouse...')
    assert df_feat['feat_has_luxury'].iloc[0] == 1
    assert df_feat['feat_luxury_count'].iloc[0] >= 2
    assert df_feat['name_length'].iloc[0] == len(df_prueba['name'].iloc[0])

    # Ratios económicos
    assert np.isclose(df_feat['availability_ratio'].iloc[0], 120 / 365.0)
    assert df_feat['is_full_year_available'].iloc[0] == 0
    assert df_feat['is_host_multi_listing'].iloc[0] == 0

    # Transformación log1p aplicada en sesgadas
    for col in COLUMNAS_SESGADAS:
        esperado = np.log1p(df_prueba[col].iloc[0])
        assert np.isclose(df_feat[col].iloc[0], esperado)


def test_crear_variables_genera_todas_las_columnas_del_modelo():
    filas = []
    for i, barrio in enumerate(['Midtown', 'Williamsburg']):
        filas.append(
            {
                'name': f'Alojamiento {i}',
                'neighbourhood_group': 'Manhattan' if i == 0 else 'Brooklyn',
                'neighbourhood': barrio,
                'latitude': 40.7580 + i * 0.01,
                'longitude': -73.9855 - i * 0.01,
                'room_type': 'Entire home/apt',
                'minimum_nights': 2,
                'number_of_reviews': 10 + i,
                'reviews_per_month': 1.0,
                'calculated_host_listings_count': 1,
                'availability_365': 120,
            }
        )

    df_feat = crear_variables(preparar_datos(pd.DataFrame(filas)))
    requeridas = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS_OHE + COLUMNAS_CATEGORICAS_TARGET
    faltantes = [col for col in requeridas if col not in df_feat.columns]
    assert not faltantes, f'Faltan columnas requeridas por el modelo: {faltantes}'


def test_construir_pipeline_entrena_y_predice():
    filas = []
    for i in range(30):
        filas.append(
            {
                'name': 'Luxury studio near subway' if i % 2 == 0 else 'Cozy private room',
                'neighbourhood_group': 'Manhattan' if i % 2 == 0 else 'Brooklyn',
                'neighbourhood': 'Midtown' if i % 2 == 0 else 'Williamsburg',
                'latitude': 40.75 + (i % 5) * 0.01,
                'longitude': -73.98 - (i % 5) * 0.01,
                'room_type': 'Entire home/apt' if i % 2 == 0 else 'Private room',
                'minimum_nights': 2,
                'number_of_reviews': 5,
                'reviews_per_month': 0.5,
                'calculated_host_listings_count': 1,
                'availability_365': 120,
            }
        )
    df = pd.DataFrame(filas)
    precios = np.where(df['room_type'] == 'Entire home/apt', 200.0, 80.0)

    pipeline = construir_pipeline(Ridge())
    pipeline.fit(df, precios)
    pred = pipeline.predict(df.head(3))

    assert len(pred) == 3
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)


def test_preparar_datos_falla_si_faltan_columnas_obligatorias():
    with pytest.raises(ValueError, match='latitude'):
        preparar_datos(pd.DataFrame({'name': ['Luxury apartment']}))


def test_preprocesamiento_acepta_columnas_esperadas():
    filas = []
    for i in range(20):
        filas.append(
            {
                'name': 'Luxury apartment' if i % 2 == 0 else 'Cozy room',
                'neighbourhood_group': 'Manhattan' if i % 2 == 0 else 'Brooklyn',
                'neighbourhood': 'Midtown' if i % 2 == 0 else 'Williamsburg',
                'latitude': 40.75 + (i % 5) * 0.01,
                'longitude': -73.98 - (i % 5) * 0.01,
                'room_type': 'Entire home/apt',
                'minimum_nights': 2,
                'number_of_reviews': 10,
                'reviews_per_month': 1.0,
                'calculated_host_listings_count': 1,
                'availability_365': 120,
            }
        )
    df_feat = crear_variables(preparar_datos(pd.DataFrame(filas)))
    y = np.where(df_feat['feat_has_luxury'] == 1, 150.0, 90.0)

    preprocesamiento = construir_preprocesamiento()
    transformado = preprocesamiento.fit_transform(df_feat, y)
    assert transformado.shape[0] == len(df_feat)
    assert transformado.shape[1] > len(COLUMNAS_NUMERICAS)
