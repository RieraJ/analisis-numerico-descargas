"""Tests de metodos/integracion.py con integrales de valor exacto conocido."""

import numpy as np
import pytest
from scipy import integrate

from metodos.integracion import simpson, trapecios


def muestrear(f, a, b, n):
    """Valores de f en n intervalos iguales de [a, b], y el paso h."""
    x = np.linspace(a, b, n + 1)
    return f(x), (b - a) / n


def orden_observado(integrar, f, a, b, exacto, intervalos):
    """Orden p estimado con e(h) / e(h/2) ≈ 2^p para cada par de pasos."""
    errores = []
    for n in intervalos:
        y, h = muestrear(f, a, b, n)
        errores.append(abs(integrar(y, h) - exacto))
    return [np.log2(errores[k] / errores[k + 1]) for k in range(len(errores) - 1)]


# --- Exactitud con polinomios --------------------------------------------

def test_trapecios_exacta_para_recta():
    y, h = muestrear(lambda x: 3 * x + 1, 0, 2, 7)
    assert trapecios(y, h) == pytest.approx(8.0)


def test_simpson_exacta_para_cubica():
    # ∫_0^2 (x³ − x + 2) dx = 4 − 2 + 4 = 6
    y, h = muestrear(lambda x: x**3 - x + 2, 0, 2, 8)
    assert simpson(y, h) == pytest.approx(6.0)


# --- Orden de convergencia (∫_0^π sen x dx = 2) --------------------------

INTERVALOS = [8, 16, 32, 64]


def test_orden_trapecios_es_2():
    ordenes = orden_observado(trapecios, np.sin, 0, np.pi, 2.0, INTERVALOS)
    assert all(1.9 < p < 2.1 for p in ordenes), ordenes


def test_orden_simpson_es_4():
    ordenes = orden_observado(simpson, np.sin, 0, np.pi, 2.0, INTERVALOS)
    assert all(3.9 < p < 4.1 for p in ordenes), ordenes


# --- Comparación con SciPy -----------------------------------------------

def test_trapecios_igual_a_scipy():
    y, h = muestrear(lambda x: np.exp(-x) * np.cos(3 * x), 0, 4, 37)
    assert trapecios(y, h) == pytest.approx(integrate.trapezoid(y, dx=h))


def test_simpson_igual_a_scipy_con_intervalos_pares():
    y, h = muestrear(lambda x: np.exp(-x) * np.cos(3 * x), 0, 4, 36)
    assert simpson(y, h) == pytest.approx(integrate.simpson(y, dx=h))


# --- Intervalos impares --------------------------------------------------

def test_simpson_impar_es_simpson_mas_ultimo_trapecio():
    y, h = muestrear(np.exp, 0, 1, 9)
    esperado = simpson(y[:-1], h) + h / 2 * (y[-2] + y[-1])
    assert simpson(y, h) == pytest.approx(esperado)


def test_simpson_impar_sigue_siendo_mejor_que_trapecios():
    y, h = muestrear(np.exp, 0, 1, 31)
    exacto = np.e - 1
    assert abs(simpson(y, h) - exacto) < abs(trapecios(y, h) - exacto)


def test_un_solo_intervalo():
    assert simpson([1.0, 3.0], 2.0) == pytest.approx(4.0)


# --- Datos de entrada ----------------------------------------------------

def test_pocos_puntos():
    with pytest.raises(ValueError):
        trapecios([1.0], 1.0)
    with pytest.raises(ValueError):
        simpson([1.0], 1.0)


def test_paso_no_positivo():
    with pytest.raises(ValueError):
        trapecios([1.0, 2.0], -1.0)
