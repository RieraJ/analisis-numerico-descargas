"""Tests de analisis/modelo.py."""

import numpy as np
import pytest

from analisis.modelo import (
    ajustar_tau,
    descargado_modelo,
    origen_libre,
    r2_velocidad_regimen,
    rampa,
    velocidad_modelo,
)
from analisis.preparacion import preparar_corrida
from metodos.derivacion import derivada_3_puntos
from tests.test_preparacion import H, TAU, V_INF, corrida_sintetica

I_INI = 20  # lecturas antes del arranque en las velocidades de prueba


def velocidad_exacta(t0=0.0):
    """v = 0 hasta t0 y v∞(1 − e^(−(t − t0)/τ)) después; t = 0 en la lectura I_INI."""
    t = H * (np.arange(200) - I_INI)
    v = np.where(t >= t0, V_INF * (1 - np.exp(-(t - t0) / TAU)), 0.0)
    return t, v


# --- Rampa -----------------------------------------------------------------

def test_rampa_hasta_el_primero_que_llega_al_90_por_ciento():
    v = np.array([0, 0, 20, 50, 80, 95, 70, 100], dtype=float)
    # Desde el índice 2 hasta el 5 (el primero con v ≥ 90), sin incluirlo.
    # El 70 del índice 6 ya no cuenta: la rampa termina en el primer corte.
    assert np.array_equal(rampa(v, 100, 2), [2, 3, 4])


# --- τ -----------------------------------------------------------------------

def test_ajustar_tau_con_velocidad_exacta():
    t, v = velocidad_exacta()
    r = ajustar_tau(t, v, V_INF, I_INI)
    assert r["tau"] == pytest.approx(TAU, rel=1e-9)
    assert r["r2"] == pytest.approx(1.0)
    # v < 0,9·v∞ mientras t < τ·ln 10 ≈ 0,69 s: 7 lecturas (0 a 0,6 s)
    assert r["n"] == 7


def test_rampa_con_pocos_puntos_deja_tau_sin_determinar():
    v = np.array([0, 50, 95, 100, 100], dtype=float)
    t = H * np.arange(5)
    r = ajustar_tau(t, v, 100, 1)
    assert r["n"] == 1
    assert np.isnan(r["tau"])


def test_origen_libre_recupera_el_desplazamiento():
    # El arranque real está 0,05 s antes de la lectura I_INI.
    t, v = velocidad_exacta(t0=-0.05)
    t0, tau, r2 = origen_libre(t, v, V_INF, I_INI)
    assert t0 == pytest.approx(-0.05, abs=1e-9)
    assert tau == pytest.approx(TAU, rel=1e-9)
    assert r2 == pytest.approx(1.0)


def test_corrida_sintetica_origen_corrido_un_paso():
    # En la corrida sintética t_ini queda una lectura después del arranque
    # verdadero (30,1 contra 30,0 s). La regresión por el origen lo toma como
    # t = 0 y subestima τ; con el origen libre se recupera τ y el corrimiento.
    t, B, S = corrida_sintetica()
    p = preparar_corrida(t, B, S)
    r = ajustar_tau(t, p["v"], p["v_inf"], p["i_ini"])
    assert r["tau"] == pytest.approx(TAU, rel=0.25)
    assert r["tau"] < TAU

    # La derivada de 3 puntos de D̂ cumple 1 − v/v∞ = c·e^(−t/τ): con el origen
    # libre, la recta es exacta.
    t0, tau, r2 = origen_libre(t, p["v"], p["v_inf"], p["i_ini"])
    assert tau == pytest.approx(TAU, rel=1e-6)
    assert t0 == pytest.approx(-H, abs=0.01)
    assert r2_velocidad_regimen(t, p["B_d"]) == pytest.approx(1.0)


# --- v̂ y D̂ -------------------------------------------------------------------

def test_descargado_modelo_es_la_integral_de_la_velocidad():
    h = 0.001  # paso fino: la derivada de 3 puntos tiene error O(h²)
    t = np.arange(0, 10 + h / 2, h)
    D = descargado_modelo(t, V_INF, TAU)
    assert D[0] == 0
    # D̂' = v̂
    assert np.allclose(derivada_3_puntos(D, h), velocidad_modelo(t, V_INF, TAU), rtol=0, atol=1e-4 * V_INF)
    # Para t ≫ τ, D̂ ≈ v∞(t − τ)
    assert D[-1] == pytest.approx(V_INF * (t[-1] - TAU))
