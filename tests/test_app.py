from pathlib import Path

from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]


def _iniciar() -> AppTest:
    app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_app_arranca_y_genera_estimacion():
    app = _iniciar()

    app.radio(key='modo_titulo').set_value('Escribir el mío').run()
    app.text_input(key='titulo').set_value('Luxury loft with terrace near subway').run()
    app.segmented_control(key='tipo').set_value('Entire home/apt').run()
    app.button(key='estimar').click().run()

    assert not app.exception, [e.value for e in app.exception]
    resultado = app.session_state['ultima_estimacion']
    bajo, alto = resultado['rango']
    assert bajo < resultado['precio'] < alto
    assert resultado['distrito'] == 'Manhattan'
    etiquetas = [metrica.label for metrica in app.main.metric]
    assert any('Mediana en' in etiqueta for etiqueta in etiquetas)
    assert 'Error típico del modelo' in etiquetas


def test_titulo_de_ejemplo_y_escenarios():
    app = _iniciar()

    app.selectbox(key='titulo_ejemplo').set_value('Luxury penthouse with skyline view and terrace').run()
    app.segmented_control(key='tipo').set_value('Private room').run()
    app.button(key='estimar').click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state['ultima_estimacion']['precio'] > 10


def test_app_permite_coordenadas_manuales():
    app = _iniciar()

    app.number_input(key='lat_manual').set_value(40.7061)
    app.number_input(key='lon_manual').set_value(-73.9969)
    next(b for b in app.button if 'Usar estas coordenadas' in b.label).click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state['ubicacion'] == [40.7061, -73.9969]


def test_app_permite_corregir_el_barrio():
    app = _iniciar()

    app.checkbox(key='ajuste_manual').check().run()
    app.selectbox(key='distrito_manual').set_value('Brooklyn').run()
    barrio = app.selectbox(key='barrio_manual')
    barrio.set_value('Williamsburg').run()
    app.button(key='estimar').click().run()

    assert not app.exception, [e.value for e in app.exception]
    resultado = app.session_state['ultima_estimacion']
    assert (resultado['distrito'], resultado['barrio']) == ('Brooklyn', 'Williamsburg')
