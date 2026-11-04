from pathlib import Path

from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]


def _boton_por_texto(app: AppTest, texto: str):
    return next(boton for boton in app.button if texto in boton.label)


def _input_por_texto(app: AppTest, texto: str):
    return next(campo for campo in app.text_input if texto in campo.label)


def test_app_arranca_y_genera_estimacion():
    app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]

    _input_por_texto(app, 'Título').set_value('Luxury loft with terrace near subway')
    next(sel for sel in app.selectbox if 'Tipo de alojamiento' in sel.label).set_value('Entire home/apt')

    _boton_por_texto(app, 'Estimar precio').click().run()

    assert not app.exception, [e.value for e in app.exception]
    etiquetas = [metrica.label for metrica in app.metric]
    assert 'Precio sugerido (referencia)' in etiquetas
    assert 'Rango razonable' in etiquetas
    assert any('Error mediano' in etiqueta for etiqueta in etiquetas)


def test_app_permite_coordenadas_manuales():
    app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]

    latitud = next(campo for campo in app.number_input if campo.label == 'Latitud')
    longitud = next(campo for campo in app.number_input if campo.label == 'Longitud')
    latitud.set_value(40.7061)
    longitud.set_value(-73.9969)
    _boton_por_texto(app, 'Usar estas coordenadas').click().run()

    assert not app.exception, [e.value for e in app.exception]
    valores = {metrica.label: metrica.value for metrica in app.metric}
    assert valores['Latitud'] == '40.70610'
    assert valores['Longitud'] == '-73.99690'
