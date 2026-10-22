"""Regresión por mínimos cuadrados.

Se minimiza la suma de los cuadrados de los residuos  Σ (y_i − ŷ_i)².
Igualando a cero las derivadas parciales se obtienen las ecuaciones normales,
que acá se resuelven a mano.
"""

import numpy as np


def _validar(x, y, minimo):
    """Convierte x e y a arrays de floats y controla los datos de entrada."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x e y tienen que ser arrays de una dimensión del mismo largo")
    if len(x) < minimo:
        raise ValueError(f"se necesitan al menos {minimo} puntos")
    return x, y


def regresion_lineal(x, y):
    """Recta de mínimos cuadrados  ŷ = a + b·x.  Devuelve (a, b).

    Ecuaciones normales:
        n·a  + Σx·b  = Σy
        Σx·a + Σx²·b = Σxy
    Se resuelven por la regla de Cramer, con det = n·Σx² − (Σx)².
    """
    x, y = _validar(x, y, 2)
    n = len(x)

    suma_x = np.sum(x)
    suma_y = np.sum(y)
    suma_x2 = np.sum(x * x)
    suma_xy = np.sum(x * y)

    det = n * suma_x2 - suma_x**2
    if det == 0:
        raise ValueError("todos los x son iguales: la recta no queda determinada")

    a = (suma_y * suma_x2 - suma_x * suma_xy) / det
    b = (n * suma_xy - suma_x * suma_y) / det
    return a, b


def regresion_por_origen(x, y):
    """Recta de mínimos cuadrados que pasa por el origen  ŷ = b·x.  Devuelve b.

    Ecuación normal:  Σx²·b = Σxy   →   b = Σxy / Σx²
    Se usa para τ, porque  ln(1 − v/v∞) = −t/τ  no tiene término independiente.
    """
    x, y = _validar(x, y, 1)

    suma_x2 = np.sum(x * x)
    if suma_x2 == 0:
        raise ValueError("todos los x son cero: la pendiente no queda determinada")

    return np.sum(x * y) / suma_x2


def coeficiente_determinacion(y, y_ajustado):
    """R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)².  Vale 1 si el ajuste es perfecto."""
    y, y_ajustado = _validar(y, y_ajustado, 2)

    residuos = np.sum((y - y_ajustado) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    if total == 0:
        raise ValueError("todos los y son iguales: R² no está definido")

    return 1 - residuos / total
