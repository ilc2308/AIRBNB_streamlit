"""Reentrena el modelo final de Airbnb NYC sin abrir los notebooks.

La selección completa (7 modelos, validación cruzada de 10 pliegues, ANOVA, Tukey,
GridSearch y optimización bayesiana) se hace en notebooks/Modelo_Airbnb_NYC.ipynb,
que guarda el ganador y sus hiperparámetros en models/metricas_modelo.json.

Este script lee ese ganador, lo vuelve a entrenar con el mismo 70 % de los datos limpios
(data/Airbnb_NYC_2019_limpio.csv, la salida de notebooks/PreparaciondeDatosAIRBNB.ipynb),
lo evalúa con el mismo 30 % de prueba y exporta models/modelo_final_airbnb.pkl.
Sirve para regenerar el .pkl con otra versión de las librerías sin repetir la selección.

También permite probar variantes robustas ante la cola pesada de precios:

    python train_model.py --objetivo huber
    python train_model.py --objetivo quantile --alpha 0.5 --recorte-cola 0.99

Las variantes robustas se guardan por defecto en un archivo aparte para no
sobrescribir el modelo desplegado sin revisar antes sus métricas en test.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from features import construir_pipeline

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent
RUTA_DATOS = RAIZ / 'data' / 'Airbnb_NYC_2019_limpio.csv'  # salida de la preparación de datos
RUTA_MODELO = RAIZ / 'models' / 'modelo_final_airbnb.pkl'
RUTA_METRICAS = RAIZ / 'models' / 'metricas_modelo.json'
RUTA_METRICAS_REENTRENAMIENTO = RAIZ / 'models' / 'metricas_reentrenamiento.json'


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Carga los datos limpios del notebook de preparación (las reglas de calidad ya están aplicadas)."""
    if not ruta.exists():
        raise FileNotFoundError(
            f'No existe {ruta}. Ejecuta notebooks/PreparaciondeDatosAIRBNB.ipynb para generar los datos limpios.'
        )
    df = pd.read_csv(ruta)
    df['name'] = df['name'].fillna('')  # el CSV lee los títulos vacíos como nulos
    if not ((df['price'] > 0).all() and df['minimum_nights'].between(1, 365).all()):
        raise ValueError(f'{ruta} no cumple las reglas de calidad de la preparación de datos.')
    return df


def crear_estimador(
    nombre: str, parametros: dict, objetivo: str = 'default', alpha: float = 0.5, base_score: float | None = None
):
    """Construye el estimador ganador o una variante robusta de XGBoost."""
    modelo = nombre.split(' - ')[0]
    p = dict(parametros)

    if objetivo != 'default':
        if modelo != 'XGBoost':
            raise ValueError('Las variantes robustas solo están disponibles para XGBoost.')
        p.pop('objective', None)
        p.pop('quantile_alpha', None)
        if objetivo == 'huber':
            p['objective'] = 'reg:pseudohubererror'
            # XGBoost estima mal el intercepto de pseudo-Huber cuando el target
            # viene transformado con log1p; fijarlo evita predicciones constantes.
            p['base_score'] = base_score if base_score is not None else 0.0
        elif objetivo == 'quantile':
            p['objective'] = 'reg:quantileerror'
            p['quantile_alpha'] = alpha
        else:
            raise ValueError(f'Objetivo robusto desconocido: {objetivo}')

    if modelo == 'Ridge':
        return Ridge(**p)
    if modelo == 'KNN':
        return KNeighborsRegressor(**p)
    if modelo == 'Árbol de decisión':
        return DecisionTreeRegressor(random_state=42, **p)
    if modelo == 'Red neuronal (MLP)':
        capas = p.pop('capas', None)
        if capas is None:
            capas = p.pop('hidden_layer_sizes', None)
        if capas is None:
            raise ValueError('El modelo MLP no define capas ocultas.')
        capas = tuple(int(c) for c in capas.split('-')) if isinstance(capas, str) else tuple(capas)
        return MLPRegressor(hidden_layer_sizes=capas, early_stopping=True, max_iter=300, random_state=42, **p)
    if modelo == 'Random Forest':
        return RandomForestRegressor(n_jobs=-1, random_state=42, **p)
    if modelo == 'Gradient Boosting':
        return GradientBoostingRegressor(random_state=42, **p)
    if modelo == 'XGBoost':
        return XGBRegressor(n_jobs=-1, random_state=42, **p)
    raise ValueError(f'Modelo desconocido: {nombre}')


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula el mismo conjunto de medidas usado en la comparativa del notebook."""
    y_pred = np.clip(y_pred, 1e-9, None)
    mse = mean_squared_error(y_real, y_pred)
    return {
        'MAE': float(mean_absolute_error(y_real, y_pred)),
        'MedAE': float(median_absolute_error(y_real, y_pred)),
        'RMSE': float(np.sqrt(mse)),
        'MSE': float(mse),
        'MAPE': float(mean_absolute_percentage_error(y_real, y_pred)),
        'Error_max': float(max_error(y_real, y_pred)),
        'R2_log': float(r2_score(np.log1p(y_real), np.log1p(y_pred))),
        'R2_USD': float(r2_score(y_real, y_pred)),
    }


def recortar_cola(y: pd.Series, cuantil: float) -> tuple[pd.Series, float]:
    """Recorta el target de entrenamiento en un cuantil para reducir el peso de outliers."""
    limite = float(y.quantile(cuantil))
    return y.clip(upper=limite), limite


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Reentrena el modelo final de Airbnb NYC.')
    parser.add_argument(
        '--objetivo',
        choices=['default', 'huber', 'quantile'],
        default='default',
        help='Objetivo de XGBoost: el ganador original, Huber o cuantílico.',
    )
    parser.add_argument('--alpha', type=float, default=0.5, help='Cuantil para --objetivo quantile.')
    parser.add_argument(
        '--recorte-cola',
        type=float,
        default=None,
        metavar='CUANTIL',
        help='Recorta el precio de entrenamiento en ese cuantil (por ejemplo 0.99).',
    )
    parser.add_argument('--salida', type=Path, default=None, help='Ruta del .pkl de salida.')
    parser.add_argument(
        '--metricas-salida', type=Path, default=None, help='Ruta del JSON con las métricas del reentrenamiento.'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.objetivo == 'quantile' and not 0 < args.alpha < 1:
        raise SystemExit('--alpha debe estar entre 0 y 1.')
    if args.recorte_cola is not None and not 0 < args.recorte_cola < 1:
        raise SystemExit('--recorte-cola debe estar entre 0 y 1.')

    with RUTA_METRICAS.open(encoding='utf-8') as f:
        ganador = json.load(f)['mejor_modelo']

    print('Modelo ganador según el notebook de modelado:', ganador['nombre'])
    print('Hiperparámetros:', ganador['parametros'])
    if args.objetivo != 'default':
        print(f'Variante robusta: {args.objetivo}')
    if args.recorte_cola is not None:
        print(f'Recorte de cola del target: cuantil {args.recorte_cola}')

    df = cargar_datos()
    X, y = df.drop(columns='price'), df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    limite = None
    y_train_fit = y_train
    if args.recorte_cola is not None:
        y_train_fit, limite = recortar_cola(y_train, args.recorte_cola)
        print(f'Precio máximo de entrenamiento recortado a {limite:.2f} USD')

    base_score = float(np.log1p(y_train_fit).mean()) if args.objetivo == 'huber' else None
    modelo = construir_pipeline(
        crear_estimador(ganador['nombre'], ganador['parametros'], args.objetivo, args.alpha, base_score)
    )
    modelo.fit(X_train, y_train_fit)
    pred = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, pred)

    print(f'MAE   = {metricas["MAE"]:.2f} USD')
    print(f'MedAE = {metricas["MedAE"]:.2f} USD')
    print(f'RMSE  = {metricas["RMSE"]:.2f} USD')
    print(f'MAPE  = {metricas["MAPE"]:.2%}')
    print(f'R2 log = {metricas["R2_log"]:.3f}')
    print(f'R2 USD = {metricas["R2_USD"]:.3f}')

    if args.salida is None:
        if args.objetivo == 'default':
            salida = RUTA_MODELO
        else:
            salida = RAIZ / 'models' / f'modelo_final_airbnb_{args.objetivo}.pkl'
    else:
        salida = args.salida

    if args.metricas_salida is None:
        if args.objetivo == 'default':
            metricas_salida = RUTA_METRICAS_REENTRENAMIENTO
        else:
            metricas_salida = RAIZ / 'models' / f'metricas_{args.objetivo}.json'
    else:
        metricas_salida = args.metricas_salida

    salida.parent.mkdir(parents=True, exist_ok=True)
    metricas_salida.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, salida)
    informe = {
        'modelo': ganador['nombre'],
        'objetivo': args.objetivo,
        'alpha': args.alpha if args.objetivo == 'quantile' else None,
        'base_score': base_score,
        'recorte_cola': args.recorte_cola,
        'limite_cola_usd': limite,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'metricas_test': metricas,
    }
    with metricas_salida.open('w', encoding='utf-8') as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)

    print('Modelo exportado a', salida)
    print('Métricas exportadas a', metricas_salida)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
