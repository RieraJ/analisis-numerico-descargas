"""Tests de metodos/derivacion.py con funciones de derivada exacta conocida."""

import numpy as np
import pytest

from metodos.derivacion import derivada_3_puntos, derivada_5_puntos


def error_maximo(derivar, f, df, h, a=0.0, b=2.0):
    """Error máximo de derivar(f) contra df en [a, b] con paso h."""
    x = np.arange(a, b + h / 2, h)
    return np.max(np.abs(derivar(f(x), h) - df(x)))


def orden_observado(derivar, f, df, pasos):
    """Orden p estimado con e(h) / e(h/2) ≈ 2^p para cada par de pasos."""
    errores = [error_maximo(derivar, f, df, h) for h in pasos]
    return [np.log2(errores[k] / errores[k + 1]) for k in range(len(errores) - 1)]


# --- Exactitud con polinomios --------------------------------------------

def test_3_puntos_exacta_para_polinomio_grado_2():
    h = 0.1
    x = np.arange(0, 1 + h / 2, h)
    y = 3 * x**2 - 2 * x + 5
    assert np.allclose(derivada_3_puntos(y, h), 6 * x - 2)


def test_5_puntos_exacta_para_polinomio_grado_4():
    h = 0.1
    x = np.arange(0, 1 + h / 2, h)
    y = x**4 - 2 * x**3 + x - 7
    assert np.allclose(derivada_5_puntos(y, h), 4 * x**3 - 6 * x**2 + 1)


# --- Orden de convergencia -----------------------------------------------

PASOS = [0.1, 0.05, 0.025, 0.0125]


def test_orden_3_puntos_es_2():
    ordenes = orden_observado(derivada_3_puntos, np.sin, np.cos, PASOS)
    assert all(1.9 < p < 2.1 for p in ordenes), ordenes


def test_orden_5_puntos_es_4():
    ordenes = orden_observado(derivada_5_puntos, np.exp, np.exp, PASOS)
    assert all(3.8 < p < 4.2 for p in ordenes), ordenes


def test_5_puntos_mas_precisa_que_3_puntos():
    h = 0.05
    e3 = error_maximo(derivada_3_puntos, np.sin, np.cos, h)
    e5 = error_maximo(derivada_5_puntos, np.sin, np.cos, h)
    assert e5 < e3 / 100


# --- Comparación con NumPy -----------------------------------------------

def test_3_puntos_igual_a_np_gradient():
    # np.gradient con edge_order=2 usa las mismas fórmulas de 3 puntos
    h = 0.2
    x = np.arange(0, 3 + h / 2, h)
    y = np.exp(-x) * np.sin(4 * x)
    assert np.allclose(derivada_3_puntos(y, h), np.gradient(y, h, edge_order=2))


# --- Datos de entrada ----------------------------------------------------

def test_largo_del_resultado():
    y = np.arange(10.0)
    assert len(derivada_3_puntos(y, 1.0)) == 10
    assert len(derivada_5_puntos(y, 1.0)) == 10


def test_pocos_puntos():
    with pytest.raises(ValueError):
        derivada_3_puntos([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        derivada_5_puntos([1.0, 2.0, 3.0, 4.0], 1.0)


def test_paso_no_positivo():
    with pytest.raises(ValueError):
        derivada_3_puntos([1.0, 2.0, 3.0], 0.0)
