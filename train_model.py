import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from features import COLUMNAS_SESGADAS, construir_pipeline

warnings.filterwarnings('ignore')

RUTA_DATOS = 'data/AB_NYC_2019.csv'
RUTA_MODELO = 'models/modelo_final_airbnb.pkl'
RUTA_METRICAS = 'models/metricas_modelo.json'
RUTA_IMPORTANCIA = 'models/importancia_caracteristicas.csv'


def cargar_datos():
    df = pd.read_csv(RUTA_DATOS)
    df = df[df['price'] > 0].copy()
    df = df[(df['minimum_nights'] >= 1) & (df['minimum_nights'] <= 365)].copy()
    return df


def evaluar(y_true, y_hat):
    return {
        'R2_log': float(r2_score(np.log1p(y_true), np.log1p(y_hat))),
        'R2_USD': float(r2_score(y_true, y_hat)),
        'MAE': float(mean_absolute_error(y_true, y_hat)),
        'MedAE': float(median_absolute_error(y_true, y_hat)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_hat))),
        'MAPE': float(mean_absolute_percentage_error(y_true, y_hat)),
    }


def preparar_datos_base(datos):
    columnas = [
        'neighbourhood_group', 'latitude', 'longitude', 'room_type',
        'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365',
    ]
    df = datos[columnas].copy()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)
    df['minimum_nights'] = df['minimum_nights'].clip(1, 365)
    df['availability_365'] = df['availability_365'].clip(0, 365)

    lat1, lon1 = np.radians(df['latitude']), np.radians(df['longitude'])
    lat2, lon2 = np.radians(40.7580), np.radians(-73.9855)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df['dist_centro_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))

    for col in COLUMNAS_SESGADAS:
        df[col] = np.log1p(df[col])
    return df


def construir_pipeline_base(regresor):
    columnas_num = [
        'latitude', 'longitude', 'dist_centro_km',
        'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365',
    ]
    preprocesamiento = ColumnTransformer([
        ('num', StandardScaler(), columnas_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['neighbourhood_group', 'room_type']),
    ], verbose_feature_names_out=False)

    pipe = Pipeline([
        ('preparacion', FunctionTransformer(preparar_datos_base)),
        ('preprocesamiento', preprocesamiento),
        ('modelo', regresor),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def modelos_candidatos():
    return {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=350, learning_rate=0.04, max_depth=7,
            subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=42
        ),
    }


def validacion_cruzada(X_train, y_train, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    filas = []
    for nombre, estimador in modelos_candidatos().items():
        inicio = time.time()
        for pliegue, (idx_tr, idx_val) in enumerate(kf.split(X_train), start=1):
            modelo = construir_pipeline(estimador)
            X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
            y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]
            modelo.fit(X_tr, y_tr)
            met = evaluar(y_val, modelo.predict(X_val))
            met.update({'modelo': nombre, 'pliegue': pliegue})
            filas.append(met)
        print(f'{nombre:<18} 5 pliegues en {time.time() - inicio:.1f} s')
    df_cv = pd.DataFrame(filas)
    return df_cv.drop(columns='pliegue').groupby('modelo').mean().sort_values('MAE')


def main():
    os.makedirs('models', exist_ok=True)
    df = cargar_datos()
    X = df.drop(columns='price')
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f'Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}')

    print('\nValidación cruzada del modelo avanzado:')
    tabla_cv = validacion_cruzada(X_train, y_train)
    print(tabla_cv.round(4).to_string())

    inicio = time.time()
    modelo_base = construir_pipeline_base(modelos_candidatos()['XGBoost'])
    modelo_base.fit(X_train, y_train)
    met_base = evaluar(y_test, modelo_base.predict(X_test))
    print(f'Modelo base entrenado en {time.time() - inicio:.1f} s')

    inicio = time.time()
    modelo_avanzado = construir_pipeline(modelos_candidatos()['XGBoost'])
    modelo_avanzado.fit(X_train, y_train)
    met_avanzado = evaluar(y_test, modelo_avanzado.predict(X_test))
    print(f'Modelo avanzado entrenado en {time.time() - inicio:.1f} s')

    comparativa = pd.DataFrame(
        [met_base, met_avanzado],
        index=['Modelo Base (9 variables)', 'Modelo Avanzado (minería de características)'],
    )
    print('\nComparativa en test (30 %):')
    print(comparativa.round(4).to_string())

    reduccion = {
        'MedAE_%': float((1 - met_avanzado['MedAE'] / met_base['MedAE']) * 100),
        'MAE_%': float((1 - met_avanzado['MAE'] / met_base['MAE']) * 100),
        'RMSE_%': float((1 - met_avanzado['RMSE'] / met_base['RMSE']) * 100),
        'R2_log_%': float((met_avanzado['R2_log'] / met_base['R2_log'] - 1) * 100),
    }
    print('\nMejora del modelo avanzado:')
    print(json.dumps(reduccion, indent=2, ensure_ascii=False))

    joblib.dump(modelo_avanzado, RUTA_MODELO)
    print(f'\nModelo exportado a {RUTA_MODELO}')

    paso = modelo_avanzado.regressor_.named_steps
    preproc = paso['preprocesamiento']
    importancias = paso['modelo'].feature_importances_
    nombres = preproc.get_feature_names_out()
    df_importancia = pd.DataFrame({
        'caracteristica': nombres,
        'importancia': importancias,
    }).sort_values('importancia', ascending=False)
    df_importancia.to_csv(RUTA_IMPORTANCIA, index=False)

    resultados = {
        'generado_en': datetime.now(timezone.utc).isoformat(),
        'datos': RUTA_DATOS,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'validacion_cruzada': tabla_cv.round(6).reset_index().to_dict(orient='records'),
        'test_base': met_base,
        'test_avanzado': met_avanzado,
        'mejora_porcentual': reduccion,
        'top_caracteristicas': df_importancia.head(15).round(6).to_dict(orient='records'),
    }
    with open(RUTA_METRICAS, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f'Métricas guardadas en {RUTA_METRICAS}')


if __name__ == '__main__':
    main()
