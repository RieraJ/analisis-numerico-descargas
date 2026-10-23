"""Tests de analisis/edo_modelo.py."""

import numpy as np
import pytest

from analisis.edo_modelo import ecuacion_velocidad, error_velocidad, volumen_con_edo
from metodos.edo import euler, heun, rk4
from metodos.integracion import simpson, trapecios
from tests.test_preparacion import TAU, V_INF


def test_ecuacion_velocidad():
    f = ecuacion_velocidad(V_INF, TAU)
    assert f(0.0, V_INF) == 0             # en régimen no cambia
    assert f(0.0, 0.0) == V_INF / TAU     # pendiente inicial de v̂


def test_rk4_casi_exacto_con_paso_chico():
    assert error_velocidad(rk4, V_INF, TAU, h=0.01) < 1e-7


def test_el_error_baja_con_el_orden_del_metodo():
    errores = [error_velocidad(m, V_INF, TAU, h=0.05) for m in (euler, heun, rk4)]
    assert errores[0] > errores[1] > errores[2]


def test_inestable_si_h_supera_el_limite():
    # Euler y Heun: estables si h < 2τ = 0,6 s. RK4: si h < 2,785τ ≈ 0,84 s.
    assert error_velocidad(euler, V_INF, TAU, h=0.5) < 1
    assert error_velocidad(euler, V_INF, TAU, h=1.0) > 1
    assert error_velocidad(heun, V_INF, TAU, h=1.0) > 1
    assert error_velocidad(rk4, V_INF, TAU, h=0.75) < 1
    assert error_velocidad(rk4, V_INF, TAU, h=1.0) > 1


# --- dD/dt = v medida -----------------------------------------------------------

t = 0.1 * np.arange(200)
v = 1e7 * (1 + 0.3 * np.sin(t))   # una "velocidad medida" cualquiera
indices = np.arange(10, 171, 10)  # 17 muestras cada 1 s: 16 intervalos


def test_heun_sobre_muestras_es_trapecios():
    assert volumen_con_edo(t, v, indices, heun) == pytest.approx(trapecios(v[indices], 1.0))


def test_rk4_sobre_muestras_es_simpson():
    assert volumen_con_edo(t, v, indices, rk4, pasos_por_muestra=2) == pytest.approx(simpson(v[indices], 1.0))


def test_rk4_necesita_intervalos_pares():
    with pytest.raises(ValueError):
        volumen_con_edo(t, v, indices[:-1], rk4, pasos_por_muestra=2)  # 15 intervalos
