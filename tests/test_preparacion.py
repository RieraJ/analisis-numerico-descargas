"""Tests de analisis/preparacion.py con una corrida sintética de parámetros conocidos."""

import numpy as np
import pytest

from analisis.preparacion import (
    cargar_corrida,
    limites_descarga,
    paso_medio,
    preparar_corrida,
    quitar_fondo,
    sobrecarga,
    submuestrear,
    tasa_de_fondo,
    velocidad_regimen,
)

# --- Corrida sintética ---------------------------------------------------
# 30 s sin descarga, una descarga con arranque v∞(1 − e^(−t/τ)) entre T_A y T_B,
# y 30 s más sin descarga. Todo el tiempo entra tráfico de fondo a R_F B/s.

H = 0.1
R_F = 500.0     # B/s
V_INF = 1e7     # 10 MB/s
TAU = 0.3       # s
T_A = 30.0      # inicio de la descarga
T_B = 130.0     # fin de la descarga
T_TOTAL = 160.0
P = 0.04        # sobrecarga


def descargado(t):
    """D(t) = v∞ [s − τ(1 − e^(−s/τ))], con s = t − T_A; constante después de T_B."""
    s = np.clip(t, T_A, T_B) - T_A
    return V_INF * (s - TAU * (1 - np.exp(-s / TAU)))


def corrida_sintetica():
    t = np.arange(0, T_TOTAL + H / 2, H)
    B = R_F * t + descargado(t)
    S = descargado(T_B) / (1 + P)  # tamaño del archivo, para que la sobrecarga sea P
    return t, B, S


# --- Lectura y paso ------------------------------------------------------

def test_cargar_corrida_resta_el_contador_inicial(tmp_path):
    ruta = tmp_path / "corrida.csv"
    ruta.write_text(
        "t_s,bytes_recv,fecha_hora\n"
        "0.0,5000000000,2026-10-23T10:00:00.000\n"
        "0.1,5000001500,2026-10-23T10:00:00.100\n"
        "0.2,5000004000,2026-10-23T10:00:00.200\n"
    )
    t, B = cargar_corrida(ruta)
    assert np.allclose(t, [0.0, 0.1, 0.2])
    assert np.array_equal(B, [0, 1500, 4000])


def test_paso_medio():
    h, desvio = paso_medio(np.array([0.0, 0.1, 0.2, 0.3]))
    assert h == pytest.approx(0.1)
    assert desvio == pytest.approx(0, abs=1e-15)


def test_submuestrear_cada_10_da_paso_de_1_s():
    t, B, _ = corrida_sintetica()
    t1, B1 = submuestrear(t, B, 10)
    assert paso_medio(t1)[0] == pytest.approx(1.0)
    assert B1[5] == B[50]


# --- Fondo y velocidad en régimen ----------------------------------------

def test_tasa_de_fondo_recupera_la_pendiente():
    t, B, _ = corrida_sintetica()
    r_f, r_inicio, r_final = tasa_de_fondo(t, B)
    assert r_inicio == pytest.approx(R_F)
    assert r_final == pytest.approx(R_F)
    assert r_f == pytest.approx(R_F)


def test_quitar_fondo_deja_solo_la_descarga():
    t, B, _ = corrida_sintetica()
    assert np.allclose(quitar_fondo(t, B, R_F), descargado(t))


def test_velocidad_regimen_recupera_v_inf():
    t, B, _ = corrida_sintetica()
    v_inf = velocidad_regimen(t, quitar_fondo(t, B, R_F))
    assert v_inf == pytest.approx(V_INF)


# --- Inicio, fin y sobrecarga --------------------------------------------

def test_limites_descarga_con_umbral():
    v = np.array([0, 0, 5, 50, 100, 100, 3, 100, 40, 1, 0], dtype=float)
    # 0,1·v∞ = 10: el primero que lo supera es el índice 3 y el último, el 8.
    # El 6 queda por debajo, pero no corta la descarga.
    assert limites_descarga(v, v_inf=100) == (3, 8)


def test_limites_descarga_sin_descarga():
    with pytest.raises(ValueError):
        limites_descarga(np.zeros(10), v_inf=100)


def test_sobrecarga():
    assert sobrecarga(104, 100) == pytest.approx(0.04)


# --- Toda la etapa ---------------------------------------------------------

def test_preparar_corrida_recupera_los_parametros():
    t, B, S = corrida_sintetica()
    r = preparar_corrida(t, B, S)

    assert r["h"] == pytest.approx(H)
    assert r["r_f"] == pytest.approx(R_F)
    assert r["v_inf"] == pytest.approx(V_INF)
    # Inicio y fin, con error de a lo sumo un paso (1,01 por el redondeo de t)
    assert r["t_ini"] == pytest.approx(T_A, abs=1.01 * H)
    assert r["t_fin"] == pytest.approx(T_B, abs=1.01 * H)
    # En el último paso entran a lo sumo v∞·h bytes: error en p ≤ v∞·h / S
    assert r["p"] == pytest.approx(P, abs=V_INF * H / S)
