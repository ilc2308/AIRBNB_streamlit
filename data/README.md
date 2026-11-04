# Datos

## Archivos versionados

- `AB_NYC_2019.csv`: dataset original de anuncios de Airbnb en Nueva York (2019).
- `Airbnb_NYC_2019_limpio.csv`: salida de `notebooks/PreparaciondeDatosAIRBNB.ipynb`
  (48 870 anuncios × 12 columnas, sin errores ni nulos). Es la **entrada del entrenamiento**:
  la leen `notebooks/04_mineria_caracteristicas.ipynb`, `notebooks/Modelo_Airbnb_NYC.ipynb` y
  `train_model.py`, y la app la usa para comparar el precio con los anuncios del mismo barrio.
- `ubicaciones_distritos.csv`: centroides por micro-barrio (221 filas) usados por la
  app para estimar el distrito y el micro-barrio de un punto del mapa. Se generaron
  a partir de las coordenadas del dataset original.
- `titulos_ejemplo.csv`: 50 títulos de ejemplo agrupados por tipo de palabra clave, para la lista de la app.

## Archivos generados (no versionados)

- `*_minado.csv`: salidas exploratorias del notebook de minería de características.

## Fuente y licencia

El dataset original es **New York City Airbnb Open Data (2019)**, publicado en
Kaggle por `dgomonov` a partir de información pública de Airbnb/Inside Airbnb:

- https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- Licencia del dataset: **CC0: Public Domain**.

Los datos se usan con fines académicos. El código de este repositorio se distribuye
bajo la licencia MIT (ver `LICENSE`).
