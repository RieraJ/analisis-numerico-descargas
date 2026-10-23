"""Tests de metodos/edo.py con ecuaciones de solución exacta conocida."""

import numpy as np
import pytest
from scipy import integrate

from metodos.edo import euler, heun, rk4
from metodos.integracion import simpson, trapecios

METODOS = [euler, heun, rk4]


def f_decaimiento(t, y):
    """y' = −y + sen(t): depende de t y de y.  Con y(0) = 1:
    y(t) = 1,5·e^(−t) + (sen t − cos t)/2."""
    return -y + np.sin(t)


def exacta_decaimiento(t):
    return 1.5 * np.exp(-t) + (np.sin(t) - np.cos(t)) / 2


def orden_observado(metodo, pasos, T=2.0):
    """Orden p estimado con e(h) / e(h/2) ≈ 2^p, con e el error máximo en [0, T].

    Se usa el máximo y no el error en t = T porque, en un punto suelto, los
    términos del error pueden cancelarse en parte y el orden tarda en verse.
    """
    errores = []
    for n in pasos:
        t, y = metodo(f_decaimiento, 0.0, 1.0, T / n, n)
        errores.append(np.max(np.abs(y - exacta_decaimiento(t))))
    return [np.log2(errores[k] / errores[k + 1]) for k in range(len(errores) - 1)]


# --- Orden de convergencia ----------------------------------------------------

def test_orden_euler_es_1():
    assert orden_observado(euler, [100, 200, 400, 800])[-1] == pytest.approx(1, abs=0.05)


def test_orden_heun_es_2():
    assert orden_observado(heun, [100, 200, 400, 800])[-1] == pytest.approx(2, abs=0.05)


def test_orden_rk4_es_4():
    assert orden_observado(rk4, [10, 20, 40, 80])[-1] == pytest.approx(4, abs=0.1)


# --- Exactitud y comparación con SciPy -------------------------------------

@pytest.mark.parametrize("metodo", METODOS)
def test_exacto_si_la_solucion_es_una_recta(metodo):
    # y' = 3  →  y = 1 + 3t: todos los métodos la siguen sin error
    t, y = metodo(lambda t, y: 3.0, 0.0, 1.0, 0.5, 10)
    assert np.allclose(y, 1 + 3 * t)


def test_rk4_igual_a_scipy():
    t, y = rk4(f_decaimiento, 0.0, 1.0, 0.01, 300)
    referencia = integrate.solve_ivp(f_decaimiento, (0, 3), [1.0], t_eval=t, rtol=1e-12, atol=1e-12)
    assert np.allclose(y, referencia.y[0], atol=1e-9)


# --- Si f no depende de y: Heun = trapecios, RK4 = Simpson ------------------

def test_heun_es_trapecios_y_rk4_es_simpson():
    g = lambda t: np.exp(-t) * np.cos(3 * t)
    h, n = 0.2, 10
    _, y_heun = heun(lambda t, y: g(t), 0.0, 0.0, h, n)
    _, y_rk4 = rk4(lambda t, y: g(t), 0.0, 0.0, h, n)
    assert y_heun[-1] == pytest.approx(trapecios(g(h * np.arange(n + 1)), h))
    # RK4 con paso h usa g en los puntos medios: Simpson con paso h/2
    assert y_rk4[-1] == pytest.approx(simpson(g(h / 2 * np.arange(2 * n + 1)), h / 2))


# --- Estabilidad -----------------------------------------------------------------

def test_euler_estable_solo_si_h_menor_que_2_tau():
    # y' = −y/τ: Euler multiplica y por (1 − h/τ) en cada paso.
    # Si h > 2τ, |1 − h/τ| > 1 y la solución crece en lugar de decaer.
    tau = 0.3
    _, y_estable = euler(lambda t, y: -y / tau, 0.0, 1.0, 0.5, 40)    # h < 2τ = 0,6
    _, y_inestable = euler(lambda t, y: -y / tau, 0.0, 1.0, 0.7, 40)  # h > 2τ
    assert abs(y_estable[-1]) < 1e-3
    assert abs(y_inestable[-1]) > 1


# --- Datos de entrada --------------------------------------------------------------

@pytest.mark.parametrize("metodo", METODOS)
def test_paso_no_positivo(metodo):
    with pytest.raises(ValueError):
        metodo(f_decaimiento, 0.0, 1.0, 0.0, 10)


@pytest.mark.parametrize("metodo", METODOS)
def test_sin_pasos(metodo):
    with pytest.raises(ValueError):
        metodo(f_decaimiento, 0.0, 1.0, 0.1, 0)
