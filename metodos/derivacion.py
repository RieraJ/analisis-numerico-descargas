"""Derivación numérica con paso uniforme h.

Las dos funciones reciben los valores y_i = f(x_i), con x_i = x_0 + i·h, y
devuelven un array del mismo largo con la aproximación de f'(x_i). En el
interior se usan fórmulas centradas y en los extremos, fórmulas descentradas
del mismo orden, para que todo el resultado tenga el mismo orden de error.
"""

import numpy as np


def _validar(y, h, minimo):
    """Convierte y a array de floats y controla los datos de entrada."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y tiene que ser un array de una dimensión")
    if len(y) < minimo:
        raise ValueError(f"se necesitan al menos {minimo} puntos")
    if h <= 0:
        raise ValueError("el paso h tiene que ser positivo")
    return y


def derivada_3_puntos(y, h):
    """Derivada con fórmulas de 3 puntos. Error O(h²).

    Interior (centrada):     y'_i ≈ (y_{i+1} − y_{i−1}) / 2h
    Primer punto (progr.):   y'_0 ≈ (−3y_0 + 4y_1 − y_2) / 2h
    Último punto (regr.):    y'_n ≈ (3y_n − 4y_{n−1} + y_{n−2}) / 2h
    """
    y = _validar(y, h, 3)
    n = len(y) - 1  # índice del último punto
    d = np.empty_like(y)

    d[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)
    for i in range(1, n):
        d[i] = (y[i + 1] - y[i - 1]) / (2 * h)
    d[n] = (3 * y[n] - 4 * y[n - 1] + y[n - 2]) / (2 * h)

    return d


def derivada_5_puntos(y, h):
    """Derivada con fórmulas de 5 puntos. Error O(h⁴).

    Interior (centrada):
        y'_i ≈ (−y_{i+2} + 8y_{i+1} − 8y_{i−1} + y_{i−2}) / 12h
    Dos primeros puntos (progresivas):
        y'_0 ≈ (−25y_0 + 48y_1 − 36y_2 + 16y_3 − 3y_4) / 12h
        y'_1 ≈ (−3y_0 − 10y_1 + 18y_2 − 6y_3 + y_4) / 12h
    Dos últimos puntos (regresivas, las mismas reflejadas):
        y'_{n−1} ≈ (3y_n + 10y_{n−1} − 18y_{n−2} + 6y_{n−3} − y_{n−4}) / 12h
        y'_n     ≈ (25y_n − 48y_{n−1} + 36y_{n−2} − 16y_{n−3} + 3y_{n−4}) / 12h
    """
    y = _validar(y, h, 5)
    n = len(y) - 1  # índice del último punto
    d = np.empty_like(y)

    d[0] = (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * h)
    d[1] = (-3 * y[0] - 10 * y[1] + 18 * y[2] - 6 * y[3] + y[4]) / (12 * h)

    for i in range(2, n - 1):
        d[i] = (-y[i + 2] + 8 * y[i + 1] - 8 * y[i - 1] + y[i - 2]) / (12 * h)

    d[n - 1] = (3 * y[n] + 10 * y[n - 1] - 18 * y[n - 2] + 6 * y[n - 3] - y[n - 4]) / (12 * h)
    d[n] = (25 * y[n] - 48 * y[n - 1] + 36 * y[n - 2] - 16 * y[n - 3] + 3 * y[n - 4]) / (12 * h)

    return d
