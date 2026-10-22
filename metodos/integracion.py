"""Integración numérica con paso uniforme h.

Las funciones reciben los valores y_i = f(x_i), con x_i = x_0 + i·h, y
devuelven la aproximación de la integral de f entre x_0 y x_n.
"""

import numpy as np


def _validar(y, h):
    """Convierte y a array de floats y controla los datos de entrada."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y tiene que ser un array de una dimensión")
    if len(y) < 2:
        raise ValueError("se necesitan al menos 2 puntos")
    if h <= 0:
        raise ValueError("el paso h tiene que ser positivo")
    return y


def trapecios(y, h):
    """Regla de los trapecios compuesta. Error O(h²).

    I ≈ h/2 · [y_0 + 2(y_1 + ... + y_{n−1}) + y_n]
    """
    y = _validar(y, h)
    n = len(y) - 1  # cantidad de intervalos

    suma_interior = 0.0
    for i in range(1, n):
        suma_interior += y[i]

    return h / 2 * (y[0] + 2 * suma_interior + y[n])


def simpson(y, h):
    """Regla de Simpson 1/3 compuesta. Error O(h⁴).

    Con n intervalos par:
        I ≈ h/3 · [y_0 + 4(y_1 + y_3 + ... + y_{n−1}) + 2(y_2 + y_4 + ... + y_{n−2}) + y_n]

    Si n es impar, se aplica Simpson a los primeros n − 1 intervalos y
    trapecios al último. Ese último intervalo agrega un error O(h³).
    """
    y = _validar(y, h)
    n = len(y) - 1  # cantidad de intervalos

    if n == 1:
        return trapecios(y, h)

    if n % 2 == 1:
        return simpson(y[:-1], h) + trapecios(y[-2:], h)

    suma_impares = 0.0
    for i in range(1, n, 2):
        suma_impares += y[i]

    suma_pares = 0.0
    for i in range(2, n - 1, 2):
        suma_pares += y[i]

    return h / 3 * (y[0] + 4 * suma_impares + 2 * suma_pares + y[n])
