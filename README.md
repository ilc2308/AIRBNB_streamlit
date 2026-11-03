# ¿Cuánto cobrar por noche en Airbnb en Nueva York?

Si alguna vez has pensado en publicar un apartamento en Airbnb, seguramente te hiciste esta pregunta: **¿qué precio pongo?** Si cobras mucho, nadie reserva. Si cobras poco, pierdes dinero. En una ciudad como Nueva York, donde una noche puede costar desde 10 hasta 10 000 dólares, adivinar no es buena idea.

Este proyecto intenta responder esa pregunta con datos. Entrenamos un modelo de aprendizaje automático con cerca de 49 000 anuncios reales de Airbnb en Nueva York (2019). A partir de la ubicación, el tipo de alojamiento y algunos datos del anuncio, el modelo estima un **precio de referencia por noche**.

👉 **Pruébalo aquí:** https://airbnbapp-c8wca8smbq7ohcvhcgitgh.streamlit.app/

---

## ¿Qué hace la aplicación?

Ingresas los datos del alojamiento:

- la ubicación, marcándola con un clic en el mapa (el distrito y el **micro-barrio** se detectan automáticamente),
- el **título del anuncio**, que el modelo analiza para detectar señales de lujo, economía, espacio o cercanía al transporte,
- el tipo de alojamiento: casa o apartamento entero, habitación privada o habitación compartida,
- las noches mínimas de estadía y los días disponibles al año,
- el número de reseñas, las reseñas por mes y cuántos anuncios tiene el anfitrión.

Al presionar **Estimar precio**, la app muestra un **precio sugerido** junto con un **rango razonable** y una explicación breve de los factores que más influyen en la estimación (tipo de alojamiento, distrito, atributos del título y cercanía a puntos de referencia).

> **Aviso regulatorio (NYC):** desde septiembre de 2023, la *Local Law 18* restringe los alquileres completos de corta estancia (menos de 30 días) cuando el anfitrión no está presente. El dataset utilizado es de 2019, por lo que los precios son una referencia histórica y no incorporan esta normativa.

---

## Cómo lo construimos

Seguimos la metodología **CRISP-DM**, que divide un proyecto de minería de datos en fases claras.

1. **Entender el problema.** Definir qué queremos predecir (el precio por noche) y con qué datos contamos.
2. **Entender y preparar los datos.**
   - Eliminamos columnas que no aportan, como identificadores y nombres.
   - Corregimos errores: precios de 0 dólares y estadías mínimas de más de un año.
   - Rellenamos con 0 las "reseñas por mes" vacías, que correspondían a anuncios sin reseñas.
3. **Crear variables útiles (minería de características).**
   - La latitud y la longitud por separado casi no se relacionaban con el precio. Por eso construimos **distancias a 6 puntos de interés** (Times Square, Central Park, Financial District, Brooklyn Bridge, JFK y LaGuardia) y la distancia mínima a un aeropuerto.
   - Codificamos el **micro-barrio** (`neighbourhood`, 221 zonas) con **Target Encoding bayesiano out-of-fold**, evitando fuga de información.
   - Analizamos el **título del anuncio** con reglas semánticas: palabras de lujo, presupuesto, transporte y espacio, además de métricas estructurales del texto.
   - Añadimos **ratios de negocio** (disponibilidad anual, anfitrión multi-anuncio) y transformaciones logarítmicas en los conteos sesgados.
4. **Probar varios modelos.** Comparamos 7 modelos con validación cruzada de 10 pliegues sobre el 70 % de entrenamiento: 4 de aprendizaje supervisado (Ridge, KNN, árbol de decisión y red neuronal) y 3 de ensamble (Random Forest, Gradient Boosting y XGBoost). Todos usan el mismo pipeline de `features.py`.
5. **Elegir con estadística.** Medimos 8 medidas de calidad (MAE, MedAE, RMSE, MSE, MAPE, error máximo, R² log y R² en dólares). Como es un problema de regresión, el criterio de selección es **minimizar el error absoluto medio (MAE)**. Con ANOVA (F = 4.39, p = 0.0009) y la prueba de Tukey seleccionamos los 3 mejores: XGBoost, Random Forest y Gradient Boosting.
6. **Ajustar hiperparámetros.** Optimizamos los 3 modelos con GridSearch y con optimización bayesiana (Optuna), minimizando el MAE. Ganó **XGBoost optimizado con Optuna** (MAE de validación cruzada: 54.9 USD).
7. **Desplegar.** El ganador se guardó como *pipeline* completo (preparación + características + preprocesamiento + modelo) en `models/modelo_final_airbnb.pkl` y se publicó en Streamlit.

---

## ¿Qué tan bien funciona?

Lo evaluamos con un 30 % de los datos que el modelo nunca vio durante el entrenamiento. Todas las medidas son de error, así que **menor es mejor** (salvo el R²):

| Medida | Línea base ingenua (mediana) | Modelo base (9 variables) | Modelo final (34 características) |
|---|---|---|---|
| Error absoluto medio (MAE) | 85.16 USD | 56.92 USD | **55.91 USD** |
| Error mediano (MedAE) | 47.00 USD | 22.97 USD | **22.91 USD** |
| RMSE | 268.03 USD | 245.45 USD | **245.10 USD** |
| Error porcentual medio (MAPE) | 57.8 % | 31.0 % | **30.4 %** |
| R² en escala logarítmica | — | 0.621 | **0.634** |

En palabras simples: **para un alojamiento común, el modelo se equivoca en unos 23 dólares**, y el 75 % de sus estimaciones tiene un error menor de 50 dólares. Frente a cobrar siempre el precio mediano del mercado, reduce el error promedio un 34 %. Frente al modelo de 9 variables, la minería de características reduce el MAE un 1.8 % y el MAPE un 2.0 %: una mejora modesta pero en todas las medidas de error.

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
├── train_model.py             # Reentrena el modelo ganador y regenera el .pkl
├── requirements.txt           # Dependencias para producción y despliegue
├── data/
│   ├── AB_NYC_2019.csv        # Dataset original de Airbnb NYC 2019
│   ├── Airbnb_NYC_2019_limpio.csv # Dataset con reglas de calidad aplicadas
│   └── ubicaciones_distritos.csv  # Coordenadas de referencia para inferir distrito y barrio
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
│   └── test_model.py          # Pruebas de carga e inferencia del modelo
└── .github/
    └── workflows/
        └── ci.yml             # Integración continua (CI) con GitHub Actions
```

---

## Ejecutarlo en tu computador

Necesitas Python 3.12 instalado.

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
   pip install -r requirements.txt
   ```

4. **Ejecutar las pruebas unitarias:**
   ```bash
   pytest tests/
   ```

5. **Iniciar la aplicación:**
   ```bash
   streamlit run app.py
   ```

La aplicación se abrirá en tu navegador en `http://localhost:8501`.

### Reentrenar y reproducir la comparativa

```bash
python train_model.py
```

El script lee el modelo ganador y sus hiperparámetros de `models/metricas_modelo.json`, lo reentrena con el mismo 70 % de los datos, imprime sus métricas en el 30 % de prueba y reexporta `models/modelo_final_airbnb.pkl`. Sirve para regenerar el modelo con otras versiones de las librerías. La selección completa de modelos se reproduce ejecutando `notebooks/Modelo_Airbnb_NYC.ipynb` (unos 70 minutos con 2 CPU).

---

## Datos

Usamos el conjunto **New York City Airbnb Open Data (2019)**, publicado en Kaggle a partir de la información pública recopilada por [Inside Airbnb](http://insideairbnb.com/). Contiene 48 895 anuncios con su ubicación, tipo de alojamiento, precio, reseñas y disponibilidad.

---

## Autores

Proyecto desarrollado para el curso de Minería de Datos.

- *Ismael López Cardozo*
- *Santiago Concha Morales*
- *Tatiana Fernanda Suarez*
