"""Tests de metodos/interpolacion.py con polinomios conocidos y comparación con SciPy."""

import numpy as np
import pytest
from scipy import interpolate

from metodos.interpolacion import (diferencias_divididas, lagrange, newton,
                                   resolver_tridiagonal, spline_natural)

# Nodos no equiespaciados, como cuando faltan muestras
X = np.array([0.0, 0.5, 1.5, 2.0, 3.5, 4.0])


def runge(x):
    return 1 / (1 + 25 * x**2)


# --- Diferencias divididas -----------------------------------------------

def test_diferencias_divididas_de_x_al_cuadrado():
    # f = x²:  f[x0] = x0²,  f[x0,x1] = x0 + x1,  f[x0,x1,x2] = 1,  el resto 0
    x = np.array([1.0, 2.0, 4.0, 7.0])
    c = diferencias_divididas(x, x**2)
    assert np.allclose(c, [1.0, 3.0, 1.0, 0.0])


# --- Newton y Lagrange ---------------------------------------------------

@pytest.mark.parametrize("interpolar", [newton, lagrange])
def test_pasa_por_los_nodos(interpolar):
    y = np.sin(X)
    assert np.allclose(interpolar(X, y, X), y)


@pytest.mark.parametrize("interpolar", [newton, lagrange])
def test_exacto_para_polinomio_del_mismo_grado(interpolar):
    # 6 nodos → polinomio de grado 5, reproduce exactamente uno de grado 5
    p = lambda x: x**5 - 3 * x**3 + x - 2
    x_eval = np.linspace(0, 4, 41)
    assert np.allclose(interpolar(X, p(X), x_eval), p(x_eval))


def test_newton_y_lagrange_dan_el_mismo_polinomio():
    y = np.exp(-X) * np.cos(X)
    x_eval = np.linspace(-0.5, 4.5, 50)
    assert np.allclose(newton(X, y, x_eval), lagrange(X, y, x_eval))


def test_igual_a_scipy_barycentric():
    y = np.exp(-X) * np.cos(X)
    x_eval = np.linspace(0, 4, 50)
    referencia = interpolate.BarycentricInterpolator(X, y)(x_eval)
    assert np.allclose(newton(X, y, x_eval), referencia)


def test_acepta_un_numero():
    assert newton(X, X**2, 1.0) == pytest.approx(1.0)
    assert lagrange(X, X**2, 1.0) == pytest.approx(1.0)


# --- Spline cúbico natural -----------------------------------------------

def test_tridiagonal_igual_a_numpy():
    a = np.array([0.0, 1.0, 2.0, 1.0])
    b = np.array([4.0, 5.0, 6.0, 4.0])
    c = np.array([1.0, 2.0, 1.0, 0.0])
    d = np.array([1.0, 2.0, 3.0, 4.0])
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    assert np.allclose(resolver_tridiagonal(a, b, c, d), np.linalg.solve(A, d))


def test_spline_pasa_por_los_nodos():
    y = np.sin(X)
    assert np.allclose(spline_natural(X, y, X), y)


def test_spline_igual_a_scipy_natural():
    y = np.sin(X)
    x_eval = np.linspace(0, 4, 81)
    referencia = interpolate.CubicSpline(X, y, bc_type="natural")(x_eval)
    assert np.allclose(spline_natural(X, y, x_eval), referencia)


def test_spline_exacto_para_recta():
    x_eval = np.linspace(0, 4, 17)
    assert np.allclose(spline_natural(X, 2 * X - 1, x_eval), 2 * x_eval - 1)


def test_spline_con_dos_nodos_es_la_recta():
    assert spline_natural([0.0, 2.0], [1.0, 5.0], 0.5) == pytest.approx(2.0)


def test_spline_evita_runge():
    # Con 11 nodos equiespaciados, el polinomio de grado 10 oscila en los
    # extremos (fenómeno de Runge) y el spline no.
    x = np.linspace(-1, 1, 11)
    x_eval = np.linspace(-1, 1, 201)
    error_polinomio = np.max(np.abs(newton(x, runge(x), x_eval) - runge(x_eval)))
    error_spline = np.max(np.abs(spline_natural(x, runge(x), x_eval) - runge(x_eval)))
    assert error_polinomio > 1.0
    assert error_spline < 0.1


# --- Datos de entrada ----------------------------------------------------

def test_nodos_repetidos():
    with pytest.raises(ValueError):
        newton([1.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.5)


def test_spline_nodos_desordenados():
    with pytest.raises(ValueError):
        spline_natural([0.0, 2.0, 1.0], [0.0, 1.0, 2.0], 0.5)


def test_largos_distintos():
    with pytest.raises(ValueError):
        lagrange([1.0, 2.0], [1.0], 1.5)
