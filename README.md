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
4. **Probar varios modelos.** Comparamos Ridge, Random Forest, Gradient Boosting y XGBoost con validación cruzada de 5 pliegues, usando el pipeline de `features.py` como única fuente de verdad.
5. **Elegir y evaluar.** Comparamos el modelo avanzado contra una línea base de 9 variables sobre el mismo 30 % de prueba, y medimos MAE, MedAE, RMSE, MAPE y R².
6. **Desplegar.** El modelo ganador, **XGBoost**, se guardó como *pipeline* completo (preprocesamiento + Target Encoding + modelo) en `models/modelo_final_airbnb.pkl` y se publicó en Streamlit.

---

## ¿Qué tan bien funciona?

Lo evaluamos con un 30 % de los datos que el modelo nunca vio durante el entrenamiento (comparativa base 9 variables vs modelo avanzado):

| Medida | Modelo base (9 variables) | Modelo avanzado (minería) | Mejora |
|---|---|---|---|
| Error mediano (MedAE) | ~23.20 USD | ~22.89 USD | -1.31 % |
| Error absoluto medio (MAE) | ~57.08 USD | ~55.93 USD | -2.01 % |
| RMSE | ~245.12 USD | ~243.91 USD | -0.50 % |
| Error porcentual medio (MAPE) | ~31.12 % | ~30.52 % | -1.93 % |
| R² en escala logarítmica | ~0.619 | ~0.634 | +2.29 % |

En palabras simples: **para un alojamiento común, el modelo suele equivocarse en unos 23 dólares**. Frente a la estrategia ingenua de cobrar siempre el precio mediano del mercado, el modelo reduce el error promedio de forma sustancial (MAE mediano base: ~85 USD).

Las métricas detalladas y la importancia de características se generan con `python train_model.py` y se guardan en `models/metricas_modelo.json`.

Lo que más influye en el precio es, por mucho, **el tipo de alojamiento**: no es lo mismo alquilar un apartamento completo que una habitación compartida. Le siguen el **micro-barrio codificado con Target Encoding**, la **distancia a los POIs** (especialmente Financial District y Times Square) y los **atributos de lujo detectados en el título**.

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
├── train_model.py             # Entrenamiento reproducible, comparativa y exportación
├── requirements.txt           # Dependencias para producción y despliegue
├── data/
│   ├── AB_NYC_2019.csv        # Dataset original de Airbnb NYC 2019
│   ├── Airbnb_NYC_2019_limpio.csv # Dataset con reglas de calidad aplicadas
│   ├── Airbnb_NYC_2019_minado.csv # Dataset enriquecido con variables minadas
│   └── ubicaciones_distritos.csv  # Coordenadas de referencia para inferir distrito y barrio
├── models/
│   ├── modelo_final_airbnb.pkl    # Pipeline serializado (preprocesamiento + XGBoost)
│   ├── metricas_modelo.json       # Métricas base vs avanzado, CV e importancias
│   └── importancia_caracteristicas.csv
├── notebooks/
│   ├── PreparaciondeDatosAIRBNB.ipynb # Fase 2 y 3 CRISP-DM: Limpieza y EDA
│   ├── 04_mineria_caracteristicas.ipynb # Fase de minería: POIs, Target Encoding y NLP
│   ├── 05_modelado_avanzado_y_evaluacion.ipynb # Modelado y evaluación del modelo enriquecido
│   ├── Modelo_Airbnb_NYC.ipynb        # Fase 4 y 5 CRISP-DM: Modelado, ANOVA, Optuna
│   └── Despliegue_Airbnb_NYC.ipynb    # Fase 6 CRISP-DM: Preparación para producción
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

El script entrena la línea base de 9 variables y el modelo avanzado con el mismo split, ejecuta validación cruzada de 5 pliegues, imprime la comparativa, reexporta `models/modelo_final_airbnb.pkl` y guarda `models/metricas_modelo.json`.

---

## Datos

Usamos el conjunto **New York City Airbnb Open Data (2019)**, publicado en Kaggle a partir de la información pública recopilada por [Inside Airbnb](http://insideairbnb.com/). Contiene 48 895 anuncios con su ubicación, tipo de alojamiento, precio, reseñas y disponibilidad.

---

## Autores

Proyecto desarrollado para el curso de Minería de Datos.

- *Ismael López Cardozo*
- *Santiago Concha Morales*
- *Tatiana Fernanda Suarez*
