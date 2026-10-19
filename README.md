# ¿Cuánto cobrar por noche en Airbnb en Nueva York?

Si alguna vez has pensado en publicar un apartamento en Airbnb, seguramente te hiciste esta pregunta: **¿qué precio pongo?** Si cobras mucho, nadie reserva. Si cobras poco, pierdes dinero. En una ciudad como Nueva York, donde una noche puede costar desde 10 hasta 10 000 dólares, adivinar no es buena idea.

Este proyecto intenta responder esa pregunta con datos. Entrenamos un modelo de aprendizaje automático con cerca de 49 000 anuncios reales de Airbnb en Nueva York (2019). A partir de la ubicación, el tipo de alojamiento y algunos datos del anuncio, el modelo estima un **precio de referencia por noche**.

👉 **Pruébalo aquí:** https://airbnbapp-c8wca8smbq7ohcvhcgitgh.streamlit.app/

---

## ¿Qué hace la aplicación?

Ingresas los datos del alojamiento:

- el distrito y la ubicación (latitud y longitud),
- el tipo de alojamiento: casa o apartamento entero, habitación privada o habitación compartida,
- las noches mínimas de estadía y los días disponibles al año,
- el número de reseñas, las reseñas por mes y cuántos anuncios tiene el anfitrión.

Al presionar **Estimar precio**, la app muestra cuánto debería costar la noche según lo que el modelo aprendió del mercado.

---

## Cómo lo construimos

Seguimos la metodología **CRISP-DM**, que divide un proyecto de minería de datos en fases claras.

1. **Entender el problema.** Definir qué queremos predecir (el precio por noche) y con qué datos contamos.
2. **Entender y preparar los datos.**
   - Eliminamos columnas que no aportan, como identificadores y nombres.
   - Corregimos errores: precios de 0 dólares y estadías mínimas de más de un año.
   - Rellenamos con 0 las "reseñas por mes" vacías, que correspondían a anuncios sin reseñas.
3. **Crear variables útiles.**
   - La latitud y la longitud por separado casi no se relacionaban con el precio. Por eso las combinamos en una nueva variable: la **distancia a Times Square**. Resultó muy útil: cuanto más lejos del centro, más barato el alojamiento.
   - También aplicamos logaritmos al precio y a los conteos, que tenían valores muy extremos.
4. **Probar varios modelos.** Comparamos 7 modelos con validación cruzada de 10 pliegues:
   - 4 métodos de aprendizaje supervisado: Ridge, KNN, árbol de decisión y red neuronal;
   - 3 métodos de ensamble: Random Forest, Gradient Boosting y XGBoost.
5. **Elegir los mejores con estadística.** Con ANOVA y la prueba de Tukey seleccionamos los 3 mejores. Después afinamos sus hiperparámetros con GridSearch y con optimización bayesiana (Optuna).
6. **Desplegar.** El modelo ganador, **XGBoost**, se guardó en un *pipeline* que incluye toda la preparación de los datos, y se publicó en esta aplicación con Streamlit.

---

## ¿Qué tan bien funciona?

Lo evaluamos con un 30 % de los datos que el modelo nunca vio durante el entrenamiento:

| Medida | Valor |
|---|---|
| Error típico, la mitad de los alojamientos (MedAE) | ~23 USD |
| Error absoluto medio (MAE) | ~57 USD |
| Error porcentual medio (MAPE) | ~31 % |
| R² en escala logarítmica | 0.62 |

En palabras simples: **para un alojamiento común, el modelo suele equivocarse en unos 23 dólares**. Frente a la estrategia ingenua de cobrar siempre el precio mediano del mercado, el modelo reduce el error promedio en un 33 %.

Lo que más influye en el precio es, por mucho, **el tipo de alojamiento**: no es lo mismo alquilar un apartamento completo que una habitación compartida. Le siguen **estar en Manhattan** y **la distancia al centro**.

### Lo que el modelo no sabe hacer

Preferimos ser honestos con sus límites:

- **No sirve para el segmento de lujo.** Hay alojamientos de miles de dólares por noche cuyo precio depende de cosas que el dataset no incluye, como el tamaño, la calidad, las fotos o la vista. Ahí el modelo se queda muy corto: su error máximo supera los 9 000 dólares, justamente en esos casos.
- **Los datos son de 2019.** Los precios han cambiado desde entonces por la inflación, la pandemia y la regulación de alquileres de corta estancia en Nueva York. Para usarlo en serio habría que reentrenarlo con datos actuales.

Tómalo como lo que es: **una referencia razonable para empezar**, no un precio exacto.

---

## Archivos del repositorio

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La aplicación de Streamlit |
| `features.py` | Funciones de preparación de datos y de creación de variables (el modelo las necesita para funcionar) |
| `modelo_final_airbnb.pkl` | El modelo entrenado: pipeline con la preparación de datos y XGBoost |
| `requirements.txt` | Librerías necesarias, con las mismas versiones con las que se entrenó el modelo |

El análisis completo está en tres notebooks de Google Colab y en el documento de la metodología CRISP-DM:

- `PreparaciondeDatosAIRBNB.ipynb`: limpieza y preparación de los datos
- `Modelo_Airbnb_NYC.ipynb`: ingeniería de características, modelos, evaluación y pipeline final
- `Despliegue_Airbnb_NYC.ipynb`: preparación de los archivos para la aplicación
- `METODOLOGIA_CRISP_DM_Airbnb_NYC.docx`: documentación de todo el proceso

---

## Ejecutarlo en tu computador

Necesitas Python 3 instalado.

```bash
git clone <url-de-este-repositorio>
cd <nombre-del-repositorio>
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá en tu navegador, en `http://localhost:8501`.

> **Importante:** `app.py`, `features.py` y `modelo_final_airbnb.pkl` deben estar en la misma carpeta. Además, usa las versiones de `requirements.txt`: con otra versión de scikit-learn o XGBoost, el modelo podría no cargarse.

---

## Datos

Usamos el conjunto **New York City Airbnb Open Data (2019)**, publicado en Kaggle a partir de la información pública recopilada por [Inside Airbnb](http://insideairbnb.com/). Contiene 48 895 anuncios con su ubicación, tipo de alojamiento, precio, reseñas y disponibilidad.

---

## Autores

Proyecto desarrollado para el curso de Minería de Datos.

- *Ismael López Cardozo*
- *[Nombre del integrante]*
