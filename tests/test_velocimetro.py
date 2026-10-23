"""Tests de analisis/velocimetro.py."""

import numpy as np
import pytest

from analisis.velocimetro import error_en_regimen, velocidades
from tests.test_preparacion import H, T_TOTAL, V_INF, descargado


def descarga_sintetica():
    t = np.arange(0, T_TOTAL + H / 2, H)
    return t, descargado(t)


def test_velocidades_con_y_sin_submuestreo():
    t, B_d = descarga_sintetica()
    v_fino = velocidades(t, B_d, k=1)
    v_grueso = velocidades(t, B_d, k=10)
    assert v_fino["h"] == pytest.approx(0.1)
    assert v_grueso["h"] == pytest.approx(1.0)
    assert len(v_grueso["t"]) == len(t[::10])


def test_en_regimen_las_dos_formulas_dan_v_inf():
    # En régimen B_d es una recta: las dos fórmulas son exactas.
    t, B_d = descarga_sintetica()
    for k in (1, 10):
        r = velocidades(t, B_d, k)
        for v in (r["v3"], r["v5"]):
            media, desvio = error_en_regimen(v, V_INF, r["B_d"])
            assert media == pytest.approx(0, abs=1e-6 * V_INF)
            assert desvio == pytest.approx(0, abs=1e-6 * V_INF)


def test_error_en_regimen_con_oscilacion_conocida():
    # B_d de 0 a 99: el régimen son los índices 20 a 79 (60 lecturas).
    B_d = np.arange(100, dtype=float)
    a = 3.0
    v = 50 + a * (-1.0) ** np.arange(100)   # v∞ ± a, alternando
    media, desvio = error_en_regimen(v, 50, B_d)
    assert media == pytest.approx(0, abs=1e-12)
    assert desvio == pytest.approx(a)


def test_amplificacion_del_ruido():
    # B = recta + ruido blanco de desvío σ. El desvío de v es
    #   3 puntos: σ·√2/(2h) ≈ 0,71·σ/h
    #   5 puntos: σ·√130/(12h) ≈ 0,95·σ/h
    rng = np.random.default_rng(0)
    h, sigma, n = 0.1, 1000.0, 100_000
    t = h * np.arange(n)
    B = V_INF * t + rng.normal(0, sigma, n)
    r = velocidades(t, B)
    interior = slice(2, -2)  # las fórmulas de los extremos son otras
    assert np.std(r["v3"][interior]) == pytest.approx(np.sqrt(2) / 2 * sigma / h, rel=0.02)
    assert np.std(r["v5"][interior]) == pytest.approx(np.sqrt(130) / 12 * sigma / h, rel=0.02)
