import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_metricas_incluyen_cola_y_r2_en_dolares():
    with (RAIZ / 'models' / 'metricas_modelo.json').open(encoding='utf-8') as f:
        metricas = json.load(f)

    nombre = metricas['mejor_modelo']['nombre']
    fila = next(r for r in metricas['test_modelos_ajustados'] if r['modelo'] == nombre)

    for clave in ('MAE', 'MedAE', 'RMSE', 'MAPE', 'Error_max', 'R2_log', 'R2_USD'):
        assert clave in fila

    q25 = metricas['rango_referencia_razon']['q25']
    q75 = metricas['rango_referencia_razon']['q75']
    assert q25 < q75


def test_readme_reporta_metricas_de_cola():
    texto = (RAIZ / 'README.md').read_text(encoding='utf-8')
    assert 'R² en dólares' in texto
    assert 'Error máximo' in texto
    assert 'dentro del ruido' in texto
