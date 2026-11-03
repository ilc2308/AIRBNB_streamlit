"""Reentrena el modelo final de Airbnb NYC sin abrir los notebooks.

La selección completa (7 modelos, validación cruzada de 10 pliegues, ANOVA, Tukey,
GridSearch y optimización bayesiana) se hace en notebooks/Modelo_Airbnb_NYC.ipynb,
que guarda el ganador y sus hiperparámetros en models/metricas_modelo.json.

Este script lee ese ganador, lo vuelve a entrenar con el mismo 70 % de los datos,
lo evalúa con el mismo 30 % de prueba y exporta models/modelo_final_airbnb.pkl.
Sirve para regenerar el .pkl con otra versión de las librerías sin repetir la selección.

Uso:  python train_model.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from features import construir_pipeline

warnings.filterwarnings('ignore')

RUTA_DATOS = 'data/AB_NYC_2019.csv'
RUTA_MODELO = 'models/modelo_final_airbnb.pkl'
RUTA_METRICAS = 'models/metricas_modelo.json'


def cargar_datos():
    # Mismas reglas de calidad del notebook de preparación
    df = pd.read_csv(RUTA_DATOS)
    df = df.drop_duplicates(subset='id', keep='last')
    df = df[df['price'] > 0]
    df = df[(df['minimum_nights'] >= 1) & (df['minimum_nights'] <= 365)]
    return df


def crear_estimador(nombre, parametros):
    # nombre tiene la forma "<modelo> - <técnica>", por ejemplo "XGBoost - Bayesiana (Optuna)"
    modelo = nombre.split(' - ')[0]
    p = dict(parametros)
    if modelo == 'Ridge':
        return Ridge(**p)
    if modelo == 'KNN':
        return KNeighborsRegressor(**p)
    if modelo == 'Árbol de decisión':
        return DecisionTreeRegressor(random_state=42, **p)
    if modelo == 'Red neuronal (MLP)':
        capas = p.pop('capas', None) or p.pop('hidden_layer_sizes')
        capas = tuple(int(c) for c in capas.split('-')) if isinstance(capas, str) else tuple(capas)
        return MLPRegressor(hidden_layer_sizes=capas, early_stopping=True, max_iter=300, random_state=42, **p)
    if modelo == 'Random Forest':
        return RandomForestRegressor(n_jobs=-1, random_state=42, **p)
    if modelo == 'Gradient Boosting':
        return GradientBoostingRegressor(random_state=42, **p)
    if modelo == 'XGBoost':
        return XGBRegressor(n_jobs=-1, random_state=42, **p)
    raise ValueError(f'Modelo desconocido: {nombre}')


def main():
    with open(RUTA_METRICAS, encoding='utf-8') as f:
        ganador = json.load(f)['mejor_modelo']
    print('Modelo ganador según el notebook de modelado:', ganador['nombre'])
    print('Hiperparámetros:', ganador['parametros'])

    df = cargar_datos()
    X, y = df.drop(columns='price'), df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    modelo = construir_pipeline(crear_estimador(ganador['nombre'], ganador['parametros']))
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    print(f'MAE   = {mean_absolute_error(y_test, pred):.2f} USD')
    print(f'MedAE = {median_absolute_error(y_test, pred):.2f} USD')
    print(f'RMSE  = {np.sqrt(mean_squared_error(y_test, pred)):.2f} USD')
    print(f'R² log = {r2_score(np.log1p(y_test), np.log1p(pred)):.3f}')

    joblib.dump(modelo, RUTA_MODELO)
    print('Modelo exportado a', RUTA_MODELO)


if __name__ == '__main__':
    main()
