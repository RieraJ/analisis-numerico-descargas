"""Tests de metodos/raices.py con raíces conocidas y comparación con SciPy."""

import numpy as np
import pytest
from scipy import optimize

from metodos.raices import biseccion, newton_raphson, punto_fijo, secante

RAIZ_2 = np.sqrt(2)


def f(x):
    return x**2 - 2


def df(x):
    return 2 * x


def ordenes(historial, raiz):
    """Orden p estimado con e_{k+1} ≈ C·e_k^p  →  p ≈ ln(e_{k+1}/e_k) / ln(e_k/e_{k−1}).

    Se descartan los errores ya cercanos al redondeo de la máquina.
    """
    e = [abs(x - raiz) for x in historial]
    e = [ek for ek in e if ek > 1e-13]
    return [np.log(e[k + 1] / e[k]) / np.log(e[k] / e[k - 1]) for k in range(1, len(e) - 1)]


# --- Raíz de x² − 2 ------------------------------------------------------

def test_biseccion():
    raiz, historial = biseccion(f, 0, 2)
    assert raiz == pytest.approx(RAIZ_2, abs=1e-10)


def test_punto_fijo():
    # x = φ(x) = (x + 2/x)/2  tiene punto fijo √2
    raiz, _ = punto_fijo(lambda x: (x + 2 / x) / 2, 1.0)
    assert raiz == pytest.approx(RAIZ_2)


def test_newton_raphson():
    raiz, _ = newton_raphson(f, df, 1.0)
    assert raiz == pytest.approx(RAIZ_2)


def test_secante():
    raiz, _ = secante(f, 1.0, 2.0)
    assert raiz == pytest.approx(RAIZ_2)


# --- Comparación con SciPy (cos x = x) -----------------------------------

def test_los_cuatro_igual_a_brentq():
    g = lambda x: np.cos(x) - x
    referencia = optimize.brentq(g, 0, 1, xtol=1e-14)

    assert biseccion(g, 0, 1)[0] == pytest.approx(referencia, abs=1e-9)
    assert punto_fijo(np.cos, 1.0)[0] == pytest.approx(referencia, abs=1e-9)
    assert newton_raphson(g, lambda x: -np.sin(x) - 1, 1.0)[0] == pytest.approx(referencia)
    assert secante(g, 0.0, 1.0)[0] == pytest.approx(referencia)


# --- Velocidad de convergencia -------------------------------------------

def test_biseccion_divide_el_intervalo_por_2():
    # Con [0, 2] y tol: hacen falta unas log2(2 / tol) iteraciones
    tol = 1e-6
    _, historial = biseccion(f, 0, 2, tol=tol)
    assert len(historial) == pytest.approx(np.log2(2 / tol), abs=1)


def test_punto_fijo_lineal_con_razon_phi_prima():
    # φ = cos, el error se multiplica por |φ'(r)| = sen(r) en cada paso
    r = optimize.brentq(lambda x: np.cos(x) - x, 0, 1, xtol=1e-14)
    _, historial = punto_fijo(np.cos, 1.0, tol=1e-12)
    e = [abs(x - r) for x in historial]
    razones = [e[k + 1] / e[k] for k in range(10, 30)]
    assert np.allclose(razones, np.sin(r), rtol=0.01)


def test_newton_orden_2():
    _, historial = newton_raphson(f, df, 3.0, tol=1e-15)
    p = ordenes(historial, RAIZ_2)
    assert p[-1] == pytest.approx(2, abs=0.1), p


def test_secante_orden_aureo():
    _, historial = secante(f, 3.0, 2.5, tol=1e-15)
    p = ordenes(historial, RAIZ_2)
    assert p[-1] == pytest.approx((1 + np.sqrt(5)) / 2, abs=0.15), p


def test_newton_menos_iteraciones_que_biseccion():
    _, h_biseccion = biseccion(f, 0, 2)
    _, h_newton = newton_raphson(f, df, 1.0)
    assert len(h_newton) < len(h_biseccion) / 5


# --- Problema del proyecto con valores sintéticos -------------------------
# f(t) = D̂(t) − S,  D̂(t) = v∞ [t − τ(1 − e^(−t/τ))]

V_INF = 1e7   # 10 MB/s
TAU = 3.0     # s
S = 7e8       # 700 MB


def D_modelo(t):
    return V_INF * (t - TAU * (1 - np.exp(-t / TAU)))


def v_modelo(t):
    return V_INF * (1 - np.exp(-t / TAU))


def test_instante_de_fin_los_cuatro_coinciden():
    f_fin = lambda t: D_modelo(t) - S
    phi = lambda t: S / V_INF + TAU * (1 - np.exp(-t / TAU))
    referencia = optimize.brentq(f_fin, 1, 1000, xtol=1e-12)

    t_bis, _ = biseccion(f_fin, 1, 1000)
    t_pf, _ = punto_fijo(phi, S / V_INF)
    t_nr, _ = newton_raphson(f_fin, v_modelo, S / V_INF)
    t_sec, _ = secante(f_fin, S / V_INF, S / V_INF + 1)

    for t in (t_bis, t_pf, t_nr, t_sec):
        assert t == pytest.approx(referencia, abs=1e-6)
    # Con arranque lento el fin llega más tarde que S/v∞, unos τ segundos después
    assert referencia == pytest.approx(S / V_INF + TAU, abs=1e-6)


def test_instante_de_fin_punto_fijo_converge_rapido():
    # φ'(t) = e^(−t/τ) es casi 0 para t ≫ τ: converge en muy pocas iteraciones
    phi = lambda t: S / V_INF + TAU * (1 - np.exp(-t / TAU))
    _, historial = punto_fijo(phi, S / V_INF)
    assert len(historial) <= 5


# --- Errores -------------------------------------------------------------

def test_biseccion_sin_cambio_de_signo():
    with pytest.raises(ValueError):
        biseccion(f, 2, 3)


def test_biseccion_raiz_en_un_extremo():
    assert biseccion(lambda x: x - 1, 1, 3) == (1, [])


def test_punto_fijo_que_diverge():
    # φ(x) = 3x tiene |φ'| = 3 > 1
    with pytest.raises(RuntimeError):
        punto_fijo(lambda x: 3 * x, 1.0)


def test_newton_derivada_nula():
    with pytest.raises(ZeroDivisionError):
        newton_raphson(f, df, 0.0)


def test_secante_valores_iguales():
    with pytest.raises(ZeroDivisionError):
        secante(f, -1.0, 1.0)
