# Datos

## Archivos versionados

- `AB_NYC_2019.csv`: dataset original de anuncios de Airbnb en Nueva York (2019).
- `ubicaciones_distritos.csv`: centroides por micro-barrio (221 filas) usados por la
  app para estimar el distrito y el micro-barrio de un punto del mapa. Se generaron
  a partir de las coordenadas del dataset original.

## Archivos generados (no versionados)

- `Airbnb_NYC_2019_limpio.csv`: lo genera el notebook
  `notebooks/PreparaciondeDatosAIRBNB.ipynb` al aplicar las reglas de calidad.
  Se ignora en git porque puede regenerarse y no lo consume ningún script posterior.
- `*_minado.csv`: salidas exploratorias del notebook de minería de características.

## Fuente y licencia

El dataset original es **New York City Airbnb Open Data (2019)**, publicado en
Kaggle por `dgomonov` a partir de información pública de Airbnb/Inside Airbnb:

- https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- Licencia del dataset: **CC0: Public Domain**.

Los datos se usan con fines académicos. El código de este repositorio se distribuye
bajo la licencia MIT (ver `LICENSE`).
