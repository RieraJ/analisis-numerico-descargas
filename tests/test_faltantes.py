"""Tests de analisis/faltantes.py."""

import numpy as np
import pytest

from analisis.faltantes import (
    error_maximo,
    hueco_con_caida,
    indices_vecinos,
    posiciones_equiespaciadas,
    reconstruir_bloque,
)

H = 0.1
t = H * np.arange(100)


def test_indices_vecinos():
    # Hueco de 3 lecturas (10, 11, 12) con 2 vecinos por lado
    assert np.array_equal(indices_vecinos(10, 3, 2), [8, 9, 13, 14])


def test_recta_la_reconstruyen_todos_los_metodos():
    B = 1e7 * t + 500
    r = reconstruir_bloque(t, B, inicio=40, k=10)
    for metodo in ("lineal", "newton", "lagrange", "spline"):
        assert error_maximo(r, metodo) == pytest.approx(0, abs=1e-6)


def test_cubica_la_reconstruyen_newton_y_lagrange_de_grado_3():
    B = 2 * t**3 - 5 * t**2 + t
    r = reconstruir_bloque(t, B, inicio=40, k=10)
    assert error_maximo(r, "newton") == pytest.approx(0, abs=1e-8)
    assert error_maximo(r, "lagrange") == pytest.approx(0, abs=1e-8)
    assert error_maximo(r, "lineal") > 1  # la recta no alcanza


def test_newton_y_lagrange_dan_lo_mismo():
    B = np.cumsum(np.abs(np.sin(7 * t))) * 1e6  # acumulado de una "velocidad" no polinómica
    r = reconstruir_bloque(t, B, inicio=30, k=5)
    assert np.allclose(r["newton"], r["lagrange"], rtol=1e-12)


def test_valores_reales_y_bytes_del_hueco():
    B = 3.0 * np.arange(100)
    r = reconstruir_bloque(t, B, inicio=20, k=4)
    assert np.array_equal(r["real"], B[20:24])
    assert np.allclose(r["t"], t[20:24])
    assert r["bytes_hueco"] == B[24] - B[19]


def test_hueco_fuera_de_la_captura():
    B = 3.0 * np.arange(100)
    with pytest.raises(ValueError):
        reconstruir_bloque(t, B, inicio=1, k=4)      # faltan vecinos a la izquierda
    with pytest.raises(ValueError):
        reconstruir_bloque(t, B, inicio=95, k=4)     # faltan vecinos a la derecha


def test_posiciones_equiespaciadas():
    pos = posiciones_equiespaciadas(np.arange(100, 201), k=10, cantidad=5)
    # De 100 a 190 (el último hueco termina en 199): 100, 122, 145, 167, 190
    assert np.array_equal(pos, [100, 122, 145, 167, 190])


def test_hueco_con_caida():
    v = np.full(20, 100.0)
    v[10] = 30.0
    assert hueco_con_caida(v, 100, inicio=8, k=2)       # la lectura 10 toca el hueco 8–9
    assert not hueco_con_caida(v, 100, inicio=3, k=4)   # lecturas 2 a 7, sin caída
