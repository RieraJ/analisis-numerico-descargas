"""Tests de metodos/regresion.py con rectas conocidas y comparación con SciPy."""

import numpy as np
import pytest
from scipy import stats

from metodos.regresion import coeficiente_determinacion, regresion_lineal, regresion_por_origen


# --- Recta exacta --------------------------------------------------------

def test_recupera_recta_exacta():
    x = np.linspace(0, 10, 11)
    a, b = regresion_lineal(x, 4.0 - 2.5 * x)
    assert a == pytest.approx(4.0)
    assert b == pytest.approx(-2.5)


def test_por_origen_recupera_pendiente_exacta():
    x = np.linspace(1, 5, 9)
    assert regresion_por_origen(x, 0.3 * x) == pytest.approx(0.3)


def test_r2_vale_1_con_ajuste_perfecto():
    x = np.linspace(0, 1, 5)
    y = 2 * x + 1
    assert coeficiente_determinacion(y, y) == pytest.approx(1.0)


# --- Datos con ruido, comparados con SciPy y NumPy -----------------------

@pytest.fixture
def datos_con_ruido():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 50)
    y = 7.0 + 0.8 * x + rng.normal(0, 5, size=x.size)
    return x, y


def test_igual_a_scipy_linregress(datos_con_ruido):
    x, y = datos_con_ruido
    a, b = regresion_lineal(x, y)
    referencia = stats.linregress(x, y)
    assert a == pytest.approx(referencia.intercept)
    assert b == pytest.approx(referencia.slope)


def test_r2_igual_a_scipy(datos_con_ruido):
    x, y = datos_con_ruido
    a, b = regresion_lineal(x, y)
    r2 = coeficiente_determinacion(y, a + b * x)
    assert r2 == pytest.approx(stats.linregress(x, y).rvalue ** 2)


def test_por_origen_igual_a_lstsq(datos_con_ruido):
    x, y = datos_con_ruido
    referencia = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0][0]
    assert regresion_por_origen(x, y) == pytest.approx(referencia)


def test_residuos_ortogonales(datos_con_ruido):
    # Las ecuaciones normales dicen que Σr = 0 y Σx·r = 0
    x, y = datos_con_ruido
    a, b = regresion_lineal(x, y)
    r = y - (a + b * x)
    assert np.sum(r) == pytest.approx(0, abs=1e-9)
    assert np.sum(x * r) == pytest.approx(0, abs=1e-6)


# --- Escala de los datos reales ------------------------------------------

def test_escala_bytes_y_segundos():
    # Una descarga de ~1 GB a 10 MB/s, muestreada cada 1 s
    t = np.arange(0.0, 100.0)
    B = 3e8 + 1e7 * t
    a, b = regresion_lineal(t, B)
    assert b == pytest.approx(1e7)
    assert a == pytest.approx(3e8)


# --- Datos de entrada ----------------------------------------------------

def test_x_todos_iguales():
    with pytest.raises(ValueError):
        regresion_lineal([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])


def test_pocos_puntos():
    with pytest.raises(ValueError):
        regresion_lineal([1.0], [1.0])


def test_largos_distintos():
    with pytest.raises(ValueError):
        regresion_lineal([1.0, 2.0, 3.0], [1.0, 2.0])
