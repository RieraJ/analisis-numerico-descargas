"""Etapa 4: parámetros del modelo de velocidad.

Modelo del arranque de TCP, con t medido desde t_ini:

    v̂(t) = v∞ (1 − e^(−t/τ))
    D̂(t) = v∞ [t − τ(1 − e^(−t/τ))]

v∞ sale de la etapa 0. τ se ajusta linealizando:  ln(1 − v/v∞) = −t/τ,
una recta por el origen con pendiente −1/τ, sobre la rampa (v < 0,9·v∞).

Unidades: bytes y segundos.
"""

import numpy as np

from analisis.preparacion import tramo_regimen
from metodos.regresion import coeficiente_determinacion, regresion_lineal, regresion_por_origen

MINIMO_PUNTOS_RAMPA = 3


def rampa(v, v_inf, i_ini, tope=0.9):
    """Índices de la rampa: desde i_ini hasta el primero con v ≥ 0,9·v∞ (sin incluirlo).

    Si v ≥ v∞ el logaritmo de ln(1 − v/v∞) no está definido; por eso se corta antes.
    """
    i = i_ini
    while i < len(v) and v[i] < tope * v_inf:
        i += 1
    return np.arange(i_ini, i)


def _puntos_rampa(t, v, v_inf, i_ini):
    """x = t − t_ini  e  y = ln(1 − v/v∞)  sobre la rampa."""
    indices = rampa(v, v_inf, i_ini)
    x = t[indices] - t[i_ini]
    y = np.log(1 - v[indices] / v_inf)
    return indices, x, y


def ajustar_tau(t, v, v_inf, i_ini):
    """τ por regresión por el origen sobre  ln(1 − v/v∞) = −t/τ.  Devuelve un diccionario.

    Pendiente b = Σxy / Σx²  →  τ = −1/b.  Con menos de 3 puntos en la rampa,
    τ queda mal determinado y se devuelve nan (no se fuerza el ajuste).
    """
    indices, x, y = _puntos_rampa(t, v, v_inf, i_ini)
    if len(indices) < MINIMO_PUNTOS_RAMPA:
        return {"tau": np.nan, "r2": np.nan, "n": len(indices), "indices": indices}
    b = regresion_por_origen(x, y)
    return {
        "tau": -1 / b,
        "r2": coeficiente_determinacion(y, b * x),
        "n": len(indices),
        "indices": indices,
    }


def origen_libre(t, v, v_inf, i_ini):
    """Control del origen: recta con ordenada  y = a + b·x  sobre los mismos puntos.

    Si el modelo es  ln(1 − v/v∞) = −(t − t0)/τ,  entonces b = −1/τ y a = t0/τ,
    así que t0 = −a/b.  Devuelve (t0 respecto de t_ini, τ, R²).
    """
    _, x, y = _puntos_rampa(t, v, v_inf, i_ini)
    a, b = regresion_lineal(x, y)
    return -a / b, -1 / b, coeficiente_determinacion(y, a + b * x)


def r2_velocidad_regimen(t, B_d):
    """R² de la recta de mínimos cuadrados de B_d en el tramo de régimen (la que da v∞)."""
    tramo = tramo_regimen(B_d)
    a, b = regresion_lineal(t[tramo], B_d[tramo])
    return coeficiente_determinacion(B_d[tramo], a + b * t[tramo])


def velocidad_modelo(t, v_inf, tau):
    """v̂(t) = v∞ (1 − e^(−t/τ)),  con t desde t_ini."""
    return v_inf * (1 - np.exp(-t / tau))


def descargado_modelo(t, v_inf, tau):
    """D̂(t) = v∞ [t − τ(1 − e^(−t/τ))],  la integral de v̂ desde 0 hasta t."""
    return v_inf * (t - tau * (1 - np.exp(-t / tau)))


def f_fin(t, v_inf, tau, S_prima):
    """f(t) = D̂(t) − S'.  Su raíz es la duración prevista de la descarga.  f'(t) = v̂(t)."""
    return descargado_modelo(t, v_inf, tau) - S_prima


def phi_fin(t, v_inf, tau, S_prima):
    """φ(t) = S'/v∞ + τ(1 − e^(−t/τ)),  despejando t de f(t) = 0.

    φ'(t) = e^(−t/τ), entre 0 y 1 para t > 0: la iteración de punto fijo converge.
    """
    return S_prima / v_inf + tau * (1 - np.exp(-t / tau))
