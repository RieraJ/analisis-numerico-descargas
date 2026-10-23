"""Parte opcional: el modelo de velocidad como ecuación diferencial.

v̂(t) = v∞ (1 − e^(−t/τ)) es la solución de

    dv/dt = (v∞ − v) / τ,   v(0) = 0,

así que se puede resolver con Euler, Heun y RK4 y comparar contra v̂.

También dD/dt = v(t), con v medida: el lado derecho no depende de D, y Heun y
RK4 se reducen a trapecios y a Simpson sobre las muestras de v.

Unidades: bytes y segundos.
"""

import numpy as np

from analisis.modelo import velocidad_modelo
from analisis.preparacion import paso_medio


def ecuacion_velocidad(v_inf, tau):
    """Lado derecho  f(t, v) = (v∞ − v) / τ."""
    return lambda t, v: (v_inf - v) / tau


def error_velocidad(metodo, v_inf, tau, h, T=5.0):
    """Error máximo de v respecto de v̂ en [0, T], relativo a v∞, con paso h."""
    n = int(round(T / h))
    t, v = metodo(ecuacion_velocidad(v_inf, tau), 0.0, 0.0, h, n)
    return np.max(np.abs(v - velocidad_modelo(t, v_inf, tau))) / v_inf


def volumen_con_edo(t, v, indices, metodo, pasos_por_muestra=1):
    """Resuelve dD/dt = v(t), D(t_a) = 0, usando solo las muestras v[indices].

    La muestra j está en t_a + j·H. El paso del método es pasos_por_muestra·H:
    con Heun se usa 1 (evalúa v en t_k y t_k + h); con RK4, 2 (también evalúa
    en el punto medio t_k + h/2, que tiene que ser una muestra).
    Devuelve D en la última muestra.
    """
    H, _ = paso_medio(t[indices])
    intervalos = len(indices) - 1
    if intervalos % pasos_por_muestra != 0:
        raise ValueError("la cantidad de intervalos tiene que ser múltiplo de pasos_por_muestra")

    t_a = t[indices[0]]

    def v_medida(s, D):
        j = int(round((s - t_a) / H))  # número de muestra
        return v[indices[j]]

    h = pasos_por_muestra * H
    _, D = metodo(v_medida, t_a, 0.0, h, intervalos // pasos_por_muestra)
    return D[-1]
