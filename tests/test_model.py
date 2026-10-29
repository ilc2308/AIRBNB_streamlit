import os
import sys
import joblib
import pandas as pd
import numpy as np
import pytest

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


def test_modelo_carga_e_inferencia():
    ruta_modelo = 'models/modelo_final_airbnb.pkl' if os.path.exists('models/modelo_final_airbnb.pkl') else 'modelo_final_airbnb.pkl'
    assert os.path.exists(ruta_modelo), f"El archivo del modelo no existe en {ruta_modelo}"

    modelo = joblib.load(ruta_modelo)
    assert modelo is not None

    # Caso de prueba: apartamento entero en Times Square
    ejemplo = pd.DataFrame({
        'neighbourhood_group': ['Manhattan'],
        'latitude': [40.7580],
        'longitude': [-73.9855],
        'room_type': ['Entire home/apt'],
        'minimum_nights': [2],
        'number_of_reviews': [15],
        'reviews_per_month': [1.2],
        'calculated_host_listings_count': [1],
        'availability_365': [150],
    })

    prediccion = modelo.predict(ejemplo)
    assert len(prediccion) == 1
    assert prediccion[0] > 0, "El precio estimado debe ser positivo"
    # Un apartamento completo en Times Square típicamente cuesta entre $50 y $1000
    assert 50 <= prediccion[0] <= 1000, f"Predicción fuera de rango razonable: {prediccion[0]}"


def test_archivo_ubicaciones():
    ruta_ubicaciones = 'data/ubicaciones_distritos.csv' if os.path.exists('data/ubicaciones_distritos.csv') else 'ubicaciones_distritos.csv'
    assert os.path.exists(ruta_ubicaciones), f"El archivo de ubicaciones no existe en {ruta_ubicaciones}"

    df_ubicaciones = pd.read_csv(ruta_ubicaciones)
    columnas_esperadas = {'latitude', 'longitude', 'neighbourhood_group'}
    assert columnas_esperadas.issubset(df_ubicaciones.columns)

    distritos_esperados = {'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'}
    assert set(df_ubicaciones['neighbourhood_group'].unique()) == distritos_esperados
