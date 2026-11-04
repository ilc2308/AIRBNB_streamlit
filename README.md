# ¿Cuánto cobrar por noche en Airbnb en Nueva York?

Si alguna vez has pensado en publicar un apartamento en Airbnb, seguramente te hiciste esta pregunta: **¿qué precio pongo?** Si cobras mucho, nadie reserva. Si cobras poco, pierdes dinero. En una ciudad como Nueva York, donde una noche puede costar desde 10 hasta 10 000 dólares, adivinar no es buena idea.

Este proyecto intenta responder esa pregunta con datos. Entrenamos un modelo de aprendizaje automático con cerca de 49 000 anuncios reales de Airbnb en Nueva York (2019). A partir de la ubicación, el tipo de alojamiento y algunos datos del anuncio, el modelo estima un **precio de referencia por noche**.

👉 **Pruébalo aquí:** https://airbnbapp-c8wca8smbq7ohcvhcgitgh.streamlit.app/

---

## ¿Qué hace la aplicación?

Ingresas los datos del alojamiento:

- la ubicación, marcándola con un clic en el mapa (el distrito, el **micro-barrio** y las distancias a 6 puntos de interés se calculan automáticamente); si el mapa no carga, también se pueden escribir las coordenadas o buscar una dirección, y el distrito y el micro-barrio se pueden corregir a mano,
- el **título del anuncio**, elegido de una lista de 50 ejemplos o escrito a mano; la app resalta en vivo las palabras de lujo, espacio, transporte o economía que lee el modelo,
- el tipo de alojamiento: casa o apartamento entero, habitación privada o habitación compartida,
- las noches mínimas de estadía y los días disponibles al año,
- el número de reseñas, las reseñas por mes y cuántos anuncios tiene el anfitrión.

Al presionar **Estimar precio**, la app muestra:

- el **precio sugerido** por noche y su **rango de referencia**,
- la comparación con la **mediana de los anuncios del mismo micro-barrio y tipo**, con un histograma que marca dónde queda tu precio,
- una sección **¿Qué pasaría si…?** que recalcula el precio con el mismo modelo cambiando una sola cosa: otro tipo de alojamiento, el título con o sin palabras de lujo, el mínimo de 30 noches que exige la Local Law 18 o el doble de reseñas.

> **Aviso regulatorio (NYC):** desde septiembre de 2023, la *Local Law 18* restringe los alquileres completos de corta estancia (menos de 30 días) cuando el anfitrión no está presente. El dataset utilizado es de 2019, por lo que los precios son una referencia histórica y no incorporan esta normativa.

---

## Cómo lo construimos

Seguimos la metodología **CRISP-DM**, que divide un proyecto de minería de datos en fases claras.

1. **Entender el problema.** Definir qué queremos predecir (el precio por noche) y con qué datos contamos.
2. **Entender y preparar los datos.**
   - Eliminamos columnas que no aportan: identificadores (`id`, `host_id`) y el nombre del anfitrión (`host_name`). Conservamos el título del anuncio (`name`) y el micro-barrio (`neighbourhood`) para transformarlos en la etapa siguiente.
   - Corregimos errores: precios de 0 dólares y estadías mínimas de más de un año.
   - Rellenamos con 0 las "reseñas por mes" vacías, que correspondían a anuncios sin reseñas.
   - El resultado, `data/Airbnb_NYC_2019_limpio.csv` (48 870 registros), es el **único conjunto de datos con el que se hace la minería de características y se entrenan los modelos**.
3. **Crear variables útiles (minería de características).**
   - La latitud y la longitud por separado casi no se relacionaban con el precio. Por eso construimos **distancias a 6 puntos de interés** (Times Square, Central Park, Financial District, Brooklyn Bridge, JFK y LaGuardia) y la distancia mínima a un aeropuerto.
   - Codificamos el **micro-barrio** (`neighbourhood`, 221 zonas) con **Target Encoding bayesiano out-of-fold**, evitando fuga de información.
   - Analizamos el **título del anuncio** con reglas semánticas: palabras de lujo, presupuesto, transporte y espacio, además de métricas estructurales del texto.
   - Añadimos **ratios de negocio** (disponibilidad anual, anfitrión multi-anuncio) y transformaciones logarítmicas en los conteos sesgados.
4. **Probar varios modelos.** Comparamos 7 modelos con validación cruzada de 10 pliegues sobre el 70 % de entrenamiento: 4 de aprendizaje supervisado (Ridge, KNN, árbol de decisión y red neuronal) y 3 de ensamble (Random Forest, Gradient Boosting y XGBoost). Todos usan el mismo pipeline de `features.py`.
5. **Elegir con estadística.** Medimos 8 medidas de calidad (MAE, MedAE, RMSE, MSE, MAPE, error máximo, R² log y R² en dólares). Como es un problema de regresión, el criterio de selección es **minimizar el error absoluto medio (MAE)**. Con ANOVA (F = 4.27, p = 0.0011) y la prueba de Tukey seleccionamos los 3 mejores: XGBoost, Random Forest y Gradient Boosting.
6. **Ajustar hiperparámetros.** Optimizamos los 3 modelos con GridSearch y con optimización bayesiana (Optuna), minimizando el MAE. Ganó **XGBoost optimizado con Optuna** (MAE de validación cruzada: 54.9 USD).
7. **Desplegar.** El ganador se guardó como *pipeline* completo (preparación + características + preprocesamiento + modelo) en `models/modelo_final_airbnb.pkl` y se publicó en Streamlit.

---

## ¿Qué tan bien funciona?

Lo evaluamos con un 30 % de los datos que el modelo nunca vio durante el entrenamiento. Todas las medidas son de error, así que **menor es mejor** (salvo el R²):

| Medida | Línea base ingenua (mediana) | Modelo base (9 variables) | Modelo final (34 características) |
|---|---|---|---|
| Error absoluto medio (MAE) | 85.16 USD | 56.92 USD | **55.83 USD** |
| Error mediano (MedAE) | 47.00 USD | 22.97 USD | **22.71 USD** |
| RMSE | 268.03 USD | 245.45 USD | **245.02 USD** |
| Error porcentual medio (MAPE) | 57.8 % | 31.0 % | **30.4 %** |
| Error máximo | 9 893 USD | 9 949 USD | **9 939 USD** |
| R² en escala logarítmica | — | 0.621 | **0.635** |
| R² en dólares | -0.031 | 0.135 | **0.138** |

En palabras simples: **para un alojamiento común, el modelo se equivoca en unos 23 dólares**, y el 75 % de sus estimaciones tiene un error menor de 50 dólares. Frente a cobrar siempre el precio mediano del mercado, reduce el error promedio un 34 %. Frente al modelo de 9 variables, la minería de características reduce el MAE un 1.9 % y el MAPE un 2.2 %: una mejora modesta pero en todas las medidas de error.

El **R² en dólares es bajo (0.138)** porque la varianza está dominada por unos pocos alojamientos de lujo con precios de miles de dólares; el R² logarítmico (0.635) describe mejor el comportamiento en el rango típico, que es el que la app intenta estimar. El error máximo supera los 9 900 USD y está concentrado en esos casos extremos.

La diferencia entre el ganador por Optuna y XGBoost con GridSearch es mínima: MAE de validación cruzada 54.89 vs 55.08 USD. En el 30 % de prueba, GridSearch incluso obtiene un MAE ligeramente menor (55.69 vs 55.83 USD), así que la ventaja del ganador está **dentro del ruido**. Se mantiene el criterio de selección por validación cruzada, pero conviene no interpretar esa diferencia como una mejora real.

Todas las cifras salen del notebook `Modelo_Airbnb_NYC.ipynb` y se guardan en `models/metricas_modelo.json`.

Lo que más influye en el precio es, por mucho, **el tipo de alojamiento**: no es lo mismo alquilar un apartamento completo que una habitación compartida. Le siguen **estar en Manhattan**, el **micro-barrio** (codificado con Target Encoding), las **palabras de lujo del título** y la **distancia a Times Square**.

### Lo que el modelo no sabe hacer

Preferimos ser honestos con sus límites:

- **No sirve para el segmento de lujo.** Hay alojamientos de miles de dólares por noche cuyo precio depende de cosas que el dataset no incluye, como el tamaño, la calidad, las fotos o la vista. Ahí el modelo se queda muy corto: su error máximo supera los 9 000 dólares, justamente en esos casos.
- **Los datos son de 2019.** Los precios han cambiado desde entonces por la inflación, la pandemia y la regulación de alquileres de corta estancia en Nueva York. Para usarlo en serio habría que reentrenarlo con datos actuales.

Tómalo como lo que es: **una referencia razonable para empezar**, no un precio exacto.

---

## Estructura del repositorio

El repositorio está organizado de forma modular para separar datos, experimentación, modelo y despliegue:

```text
AIRBNB_streamlit/
├── app.py                     # Aplicación web interactiva en Streamlit
├── features.py                # Transformaciones, minería y pipeline de producción
├── compat.py                  # Compatibilidad Linux -> Windows al cargar XGBoost
├── train_model.py             # Reentrena el modelo ganador (con los datos limpios), variantes robustas y métricas
├── requirements.txt           # Dependencias de producción, todas fijadas
├── requirements-dev.txt       # Tests, cobertura, lint y tipos
├── requirements-notebooks.txt # Dependencias adicionales de los notebooks
├── pyproject.toml             # Configuración de Ruff, mypy y coverage
├── LICENSE                    # Licencia MIT del código
├── .streamlit/config.toml     # Tema visual de la app
├── data/
│   ├── README.md              # Fuente, licencia y descripción de los datos
│   ├── AB_NYC_2019.csv        # Dataset original de Airbnb NYC 2019
│   ├── Airbnb_NYC_2019_limpio.csv # Salida de la preparación: entrada de la minería y del entrenamiento
│   ├── ubicaciones_distritos.csv  # Centroides por micro-barrio (221 filas)
│   └── titulos_ejemplo.csv    # 50 títulos de ejemplo para la app
├── models/
│   ├── modelo_final_airbnb.pkl    # Pipeline final (preparación + características + XGBoost optimizado)
│   ├── metricas_modelo.json       # Validación cruzada, ANOVA, ajuste, prueba y ganador
│   ├── resultados_validacion_cruzada.csv # Medidas de los 7 modelos en los 10 pliegues
│   └── importancia_caracteristicas.csv
├── notebooks/
│   ├── PreparaciondeDatosAIRBNB.ipynb # Fase 2 y 3 CRISP-DM: Limpieza y EDA
│   ├── 04_mineria_caracteristicas.ipynb # Exploración de las nuevas variables (genera data/*_minado.csv, no versionado)
│   ├── Modelo_Airbnb_NYC.ipynb        # Fase 4 y 5 CRISP-DM: 7 modelos, ANOVA/Tukey, GridSearch y Optuna
│   └── Despliegue_Airbnb_NYC.ipynb    # Fase 6 CRISP-DM: prueba del modelo y de la app Streamlit
├── docs/
│   ├── METODOLOGIA_CRISP_DM_Airbnb_NYC.pdf # Documento formal de la metodología CRISP-DM
│   └── output.html            # Reporte exploratorio automatizado (YData Profiling)
├── tests/
│   ├── test_features.py       # Pruebas unitarias e integración del pipeline
│   ├── test_model.py          # Carga e inferencia del modelo y datos limpios de entrenamiento
│   ├── test_app.py            # Pruebas de la app con Streamlit AppTest
│   └── test_metricas.py       # Consistencia de las métricas y de la documentación
└── .github/
    └── workflows/
        └── ci.yml             # CI: tests en Linux/Windows, Python 3.12/3.13, lint, tipos y cobertura
```

---

## Ejecutarlo en tu computador

Necesitas **Python 3.12 o 3.13** (el modelo se entrenó con 3.13.18, pero las versiones fijadas también funcionan en 3.12). Las versiones de las librerías están fijadas en `requirements.txt`; `requirements-dev.txt` añade tests, cobertura, lint y tipos, y `requirements-notebooks.txt` añade lo necesario para ejecutar los notebooks.

1. **Clonar el repositorio:**
   ```bash
   git clone <url-de-este-repositorio>
   cd <nombre-del-repositorio>
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux / Mac:
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   # Solo la app
   pip install -r requirements.txt

   # App + tests + herramientas de calidad
   pip install -r requirements-dev.txt

   # App + notebooks
   pip install -r requirements-notebooks.txt
   ```

4. **Ejecutar las pruebas y las comprobaciones de calidad:**
   ```bash
   pytest --cov=. --cov-report=term-missing tests/
   ruff check .
   ruff format --check .
   mypy app.py features.py train_model.py compat.py
   ```

5. **Iniciar la aplicación:**
   ```bash
   streamlit run app.py
   ```

La aplicación se abrirá en tu navegador en `http://localhost:8501`.

### Orden de ejecución de los notebooks

1. `PreparaciondeDatosAIRBNB.ipynb` genera `data/Airbnb_NYC_2019_limpio.csv`.
2. `04_mineria_caracteristicas.ipynb` y `Modelo_Airbnb_NYC.ipynb` leen **solo** ese archivo limpio; no vuelven a aplicar la limpieza, solo comprueban que se cumple.
3. `Despliegue_Airbnb_NYC.ipynb` prueba el modelo guardado y la app.

### Reentrenar y reproducir la comparativa

```bash
python train_model.py
```

El script lee el modelo ganador y sus hiperparámetros de `models/metricas_modelo.json`, lo reentrena con el mismo 70 % de los datos limpios (`data/Airbnb_NYC_2019_limpio.csv`), imprime sus métricas en el 30 % de prueba, reexporta `models/modelo_final_airbnb.pkl` y guarda las métricas del reentrenamiento en `models/metricas_reentrenamiento.json`. Con las versiones fijadas reproduce exactamente el modelo del notebook (MAE de prueba 55.83 USD). La selección completa de modelos se reproduce ejecutando `notebooks/Modelo_Airbnb_NYC.ipynb` (unos 70 minutos con 2 CPU).

### Probar alternativas robustas ante outliers

La cola pesada de precios (alojamientos de lujo) domina el error máximo y el R² en dólares. Para probar alternativas sin tocar el modelo desplegado:

```bash
# Pérdida Huber
python train_model.py --objetivo huber

# Pérdida cuantílica (mediana) y recorte del 1 % superior del target de entrenamiento
python train_model.py --objetivo quantile --alpha 0.5 --recorte-cola 0.99
```

Las variantes robustas se guardan por defecto en `models/modelo_final_airbnb_<objetivo>.pkl` y sus métricas en `models/metricas_<objetivo>.json`. Compara MAE, MedAE, R² en dólares y error máximo en el 30 % de prueba antes de reemplazar el modelo de producción.

Resultados con el mismo split (30 % de prueba, datos limpios):

| Variante | MAE | MedAE | RMSE | Error máximo | R² USD |
|---|---|---|---|---|---|
| Ganador original | 55.83 USD | 22.71 USD | **245.02 USD** | **9 939 USD** | **0.138** |
| Huber | **55.51 USD** | 22.30 USD | 246.53 USD | 9 949 USD | 0.128 |
| Cuantílica 0.5 + recorte 1 % | 57.39 USD | **21.84 USD** | 252.01 USD | 9 947 USD | 0.088 |

Huber reduce levemente el MAE y el MAPE, y la cuantílica reduce el error mediano, pero ambas empeoran el RMSE y el R² en dólares. **Ninguna reduce el error máximo**, porque los outliers extremos siguen fuera del alcance del modelo. Por eso el modelo desplegado sigue siendo el ganador original.

### Despliegue en Streamlit Community Cloud

El repositorio está pensado para desplegarse en Streamlit Community Cloud con `app.py` como archivo principal. **No es necesario cambiar la versión de Python del despliegue actual**: el código y las dependencias fijadas funcionan tanto en 3.12 como en 3.13.

Las dependencias de `requirements.txt` están fijadas para que el `.pkl` se cargue con las mismas versiones con las que se entrenó. Si se cambia de versión de `scikit-learn`/`xgboost`, hay que reentrenar y volver a exportar el modelo.

---

## Datos

Usamos el conjunto **New York City Airbnb Open Data (2019)**, publicado en Kaggle a partir de la información pública recopilada por [Inside Airbnb](http://insideairbnb.com/). Contiene 48 895 anuncios con su ubicación, tipo de alojamiento, precio, reseñas y disponibilidad.

- Fuente: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- Licencia del dataset: **CC0: Public Domain**.
- `data/Airbnb_NYC_2019_limpio.csv` es la salida de la preparación de datos y la entrada del entrenamiento.
- `data/ubicaciones_distritos.csv` son centroides por micro-barrio derivados de las coordenadas del dataset, no polígonos oficiales.
- El código de este repositorio se distribuye bajo licencia MIT (ver `LICENSE`).

---

## Autores

Proyecto desarrollado para el curso de Minería de Datos.

- *Ismael López Cardozo*
- *Santiago Concha Morales*
- *Tatiana Fernanda Suarez*
