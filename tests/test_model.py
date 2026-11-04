from pathlib import Path

import joblib
import pandas as pd

from compat import aplicar_compatibilidad_xgboost

RAIZ = Path(__file__).resolve().parents[1]


def test_modelo_carga_e_inferencia():
    aplicar_compatibilidad_xgboost()

    ruta_modelo = RAIZ / 'models' / 'modelo_final_airbnb.pkl'
    assert ruta_modelo.exists(), f'El archivo del modelo no existe en {ruta_modelo}'

    modelo = joblib.load(ruta_modelo)
    assert modelo is not None

    # Caso de prueba: apartamento entero en Times Square
    ejemplo = pd.DataFrame(
        {
            'name': ['Luxury Midtown Apartment with Skyline View'],
            'neighbourhood_group': ['Manhattan'],
            'neighbourhood': ['Midtown'],
            'latitude': [40.7580],
            'longitude': [-73.9855],
            'room_type': ['Entire home/apt'],
            'minimum_nights': [2],
            'number_of_reviews': [15],
            'reviews_per_month': [1.2],
            'calculated_host_listings_count': [1],
            'availability_365': [150],
        }
    )

    prediccion = modelo.predict(ejemplo)
    assert len(prediccion) == 1
    assert prediccion[0] > 0, 'El precio estimado debe ser positivo'
    # Un apartamento completo en Times Square típicamente cuesta entre $50 y $1000
    assert 50 <= prediccion[0] <= 1000, f'Predicción fuera de rango razonable: {prediccion[0]}'


def test_archivo_ubicaciones():
    ruta_ubicaciones = RAIZ / 'data' / 'ubicaciones_distritos.csv'
    assert ruta_ubicaciones.exists(), f'El archivo de ubicaciones no existe en {ruta_ubicaciones}'

    df_ubicaciones = pd.read_csv(ruta_ubicaciones)
    columnas_esperadas = {'latitude', 'longitude', 'neighbourhood_group', 'neighbourhood'}
    assert columnas_esperadas.issubset(df_ubicaciones.columns)

    distritos_esperados = {'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'}
    assert set(df_ubicaciones['neighbourhood_group'].unique()) == distritos_esperados
    assert df_ubicaciones['neighbourhood'].nunique() >= 200
