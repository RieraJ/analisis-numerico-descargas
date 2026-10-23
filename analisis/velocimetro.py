"""Etapa 1: velocímetro.

La velocidad v = dB_d/dt se calcula con las fórmulas de 3 y de 5 puntos, con
el paso de la captura (h = 0,1 s) y submuestreando B_d (h = k·0,1 s). Se mide
cuánto se aparta v de v∞ en el tramo de régimen.

Unidades: bytes y segundos.
"""

import numpy as np

from analisis.preparacion import paso_medio, submuestrear, tramo_regimen
from metodos.derivacion import derivada_3_puntos, derivada_5_puntos


def velocidades(t, B_d, k=1):
    """Velocidad con 3 y con 5 puntos, usando una lectura de cada k.

    El paso es el paso medio de las lecturas que quedan (≈ k·h).
    Devuelve un diccionario con t, B_d, h, v3 y v5 submuestreados.
    """
    t_k, B_k = submuestrear(t, B_d, k)
    h, _ = paso_medio(t_k)
    return {
        "t": t_k,
        "B_d": B_k,
        "h": h,
        "v3": derivada_3_puntos(B_k, h),
        "v5": derivada_5_puntos(B_k, h),
    }


def error_en_regimen(v, v_inf, B_d):
    """Media y desvío estándar de v − v∞ en el tramo de régimen. Devuelve (media, desvío).

    La media mide si el velocímetro está corrido respecto de v∞; el desvío,
    cuánto oscila la lectura de una muestra a otra.
    """
    e = v[tramo_regimen(B_d)] - v_inf
    return np.mean(e), np.std(e)
