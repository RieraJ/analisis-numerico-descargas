"""Tests de analisis/volumen.py."""

import numpy as np
import pytest

from analisis.preparacion import preparar_corrida
from analisis.volumen import indices_muestreo, volumen_desde_velocidad
from metodos.derivacion import derivada_3_puntos
from tests.test_preparacion import H, V_INF, corrida_sintetica


# --- Índices de las muestras ----------------------------------------------

def test_indices_sin_desfase():
    # De 95 hacia atrás de a 10 hasta pasar 30: 95, 85, ..., 25.
    assert np.array_equal(indices_muestreo(30, 95, 10, 0, 200), np.arange(25, 96, 10))


def test_indices_con_desfase():
    # La última muestra es 95 + 3 = 98; hacia atrás hasta pasar 30: 28.
    assert np.array_equal(indices_muestreo(30, 95, 10, 3, 200), np.arange(28, 99, 10))


def test_indices_sin_submuestreo_cubren_justo_la_descarga():
    assert np.array_equal(indices_muestreo(30, 95, 1, 0, 200), np.arange(30, 96))


def test_indices_fuera_de_la_captura():
    with pytest.raises(ValueError):
        indices_muestreo(3, 95, 10, 0, 200)    # empezaría en el índice −5
    with pytest.raises(ValueError):
        indices_muestreo(30, 195, 10, 9, 200)  # terminaría después del último
    with pytest.raises(ValueError):
        indices_muestreo(30, 95, 10, 10, 200)  # desfase fuera de rango


# --- Integración ------------------------------------------------------------

def test_exacto_si_la_velocidad_es_lineal():
    # B_d = 3t² + 2t: la derivada de 3 puntos es exacta (v = 6t + 2) y
    # trapecios y Simpson integran una recta sin error.
    t = np.arange(0, 100 + H / 2, H)
    B_d = 3 * t**2 + 2 * t
    v = derivada_3_puntos(B_d, H)
    r = volumen_desde_velocidad(t, v, B_d, i_ini=100, i_fin=900, k=10, desfase=4)
    assert r["H"] == pytest.approx(1.0)
    assert r["trapecios"] == pytest.approx(r["referencia"], rel=1e-10)
    assert r["simpson"] == pytest.approx(r["referencia"], rel=1e-10)


def test_control_con_todas_las_muestras_telescopa():
    # Con H = h, trapecios sobre la derivada de 3 puntos da exactamente
    #   h/2·(v_a + v_b) + (B_b + B_{b−1} − B_a − B_{a+1}) / 2
    t, B, S = corrida_sintetica()
    p = preparar_corrida(t, B, S)
    r = volumen_desde_velocidad(t, p["v"], p["B_d"], p["i_ini"], p["i_fin"], k=1)
    a, b = r["indices"][0], r["indices"][-1]
    v, B_d, h = p["v"], p["B_d"], r["H"]
    esperado = h / 2 * (v[a] + v[b]) + (B_d[b] + B_d[b - 1] - B_d[a] - B_d[a + 1]) / 2
    assert r["trapecios"] == pytest.approx(esperado, rel=1e-12)


def test_corrida_sintetica_con_muestras_cada_1_s():
    # En la corrida sintética v es constante salvo en el arranque y en el
    # corte final. Con cualquier desfase, el error queda por debajo de lo que
    # se descarga en un paso H (v∞·H).
    t, B, S = corrida_sintetica()
    p = preparar_corrida(t, B, S)
    for desfase in range(10):
        r = volumen_desde_velocidad(t, p["v"], p["B_d"], p["i_ini"], p["i_fin"], 10, desfase)
        for metodo in ("trapecios", "simpson"):
            assert abs(r[metodo] - r["referencia"]) < V_INF * r["H"]
