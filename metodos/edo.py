"""Ecuaciones diferenciales ordinarias  y' = f(t, y),  y(t_0) = y_0,  con paso fijo h.

Las tres funciones devuelven (t, y): arrays de n + 1 valores, con t_k = t_0 + k·h
e y_k la aproximación de y(t_k).
"""

import numpy as np


def _validar(h, n):
    """Controla los datos de entrada y arma la grilla de pasos."""
    if h <= 0:
        raise ValueError("el paso h tiene que ser positivo")
    if n < 1:
        raise ValueError("se necesita al menos un paso")


def euler(f, t0, y0, h, n):
    """Método de Euler. Error global O(h).

    y_{k+1} = y_k + h·f(t_k, y_k)
    """
    _validar(h, n)
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for k in range(n):
        y[k + 1] = y[k] + h * f(t[k], y[k])
    return t, y


def heun(f, t0, y0, h, n):
    """Método de Heun (Euler mejorado). Error global O(h²).

    k1 = f(t_k, y_k)
    k2 = f(t_k + h, y_k + h·k1)
    y_{k+1} = y_k + h/2 · (k1 + k2)

    Si f no depende de y, es la regla de los trapecios.
    """
    _validar(h, n)
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for k in range(n):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h, y[k] + h * k1)
        y[k + 1] = y[k] + h / 2 * (k1 + k2)
    return t, y


def rk4(f, t0, y0, h, n):
    """Runge-Kutta clásico de orden 4. Error global O(h⁴).

    k1 = f(t_k, y_k)
    k2 = f(t_k + h/2, y_k + h/2·k1)
    k3 = f(t_k + h/2, y_k + h/2·k2)
    k4 = f(t_k + h, y_k + h·k3)
    y_{k+1} = y_k + h/6 · (k1 + 2k2 + 2k3 + k4)

    Si f no depende de y, k2 = k3 y es la regla de Simpson con paso h/2.
    """
    _validar(h, n)
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for k in range(n):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h / 2, y[k] + h / 2 * k1)
        k3 = f(t[k] + h / 2, y[k] + h / 2 * k2)
        k4 = f(t[k] + h, y[k] + h * k3)
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
