"""Interpolación: polinomio de Newton, polinomio de Lagrange y spline cúbico natural.

Las tres funciones reciben los nodos (x_i, y_i), con los x_i distintos y no
necesariamente equiespaciados (por ejemplo, cuando faltan muestras), y
devuelven el valor interpolado en x_eval (un número o un array).
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
    if len(np.unique(x)) != len(x):
        raise ValueError("los nodos x tienen que ser distintos")
    return x, y


# --- Polinomio de Newton -------------------------------------------------

def diferencias_divididas(x, y):
    """Coeficientes c_k = f[x_0, ..., x_k] del polinomio de Newton.

    f[x_i] = y_i
    f[x_i, ..., x_{i+j}] = (f[x_{i+1}, ..., x_{i+j}] − f[x_i, ..., x_{i+j−1}]) / (x_{i+j} − x_i)

    La tabla se calcula columna por columna sobre un solo array: después de
    la columna j, c[i] (con i ≥ j) vale f[x_{i−j}, ..., x_i].
    """
    x, y = _validar(x, y, 1)
    n = len(x) - 1
    c = y.copy()

    for j in range(1, n + 1):
        # De abajo hacia arriba, para no pisar valores que todavía se usan
        for i in range(n, j - 1, -1):
            c[i] = (c[i] - c[i - 1]) / (x[i] - x[i - j])

    return c


def newton(x, y, x_eval):
    """Polinomio interpolante en la forma de Newton. Grado n con n + 1 nodos.

    p(x) = c_0 + c_1(x − x_0) + c_2(x − x_0)(x − x_1) + ... + c_n(x − x_0)···(x − x_{n−1})

    Se evalúa de adentro hacia afuera (esquema de Horner):
    p(x) = c_0 + (x − x_0)(c_1 + (x − x_1)(c_2 + ... + (x − x_{n−1})c_n))
    """
    x, y = _validar(x, y, 1)
    x_eval = np.asarray(x_eval, dtype=float)
    c = diferencias_divididas(x, y)
    n = len(x) - 1

    p = c[n] * np.ones_like(x_eval)
    for k in range(n - 1, -1, -1):
        p = c[k] + (x_eval - x[k]) * p

    return p


# --- Polinomio de Lagrange -----------------------------------------------

def lagrange(x, y, x_eval):
    """Polinomio interpolante en la forma de Lagrange. Grado n con n + 1 nodos.

    p(x) = Σ_i y_i · L_i(x),   con   L_i(x) = Π_{j≠i} (x − x_j) / (x_i − x_j)

    Es el mismo polinomio que el de Newton, escrito de otra forma.
    """
    x, y = _validar(x, y, 1)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x) - 1

    p = np.zeros_like(x_eval)
    for i in range(n + 1):
        L_i = np.ones_like(x_eval)
        for j in range(n + 1):
            if j != i:
                L_i = L_i * (x_eval - x[j]) / (x[i] - x[j])
        p = p + y[i] * L_i

    return p


# --- Spline cúbico natural -----------------------------------------------

def resolver_tridiagonal(a, b, c, d):
    """Resuelve un sistema tridiagonal por el algoritmo de Thomas.

    Ecuación i:  a_i·z_{i−1} + b_i·z_i + c_i·z_{i+1} = d_i
    (a_0 y c_{m−1} no se usan.)

    1. Eliminación hacia adelante: se anula la diagonal de abajo.
    2. Sustitución hacia atrás: se despeja desde la última incógnita.
    """
    m = len(d)
    c_prima = np.zeros(m)
    d_prima = np.zeros(m)

    c_prima[0] = c[0] / b[0]
    d_prima[0] = d[0] / b[0]
    for i in range(1, m):
        divisor = b[i] - a[i] * c_prima[i - 1]
        if i < m - 1:
            c_prima[i] = c[i] / divisor
        d_prima[i] = (d[i] - a[i] * d_prima[i - 1]) / divisor

    z = np.zeros(m)
    z[m - 1] = d_prima[m - 1]
    for i in range(m - 2, -1, -1):
        z[i] = d_prima[i] - c_prima[i] * z[i + 1]

    return z


def segundas_derivadas_spline(x, y):
    """Segundas derivadas M_i del spline cúbico natural en cada nodo.

    Con h_i = x_{i+1} − x_i, para i = 1, ..., n − 1:
        h_{i−1}·M_{i−1} + 2(h_{i−1} + h_i)·M_i + h_i·M_{i+1}
            = 6 [ (y_{i+1} − y_i)/h_i − (y_i − y_{i−1})/h_{i−1} ]
    Natural:  M_0 = M_n = 0.
    """
    x, y = _validar(x, y, 2)
    if np.any(np.diff(x) <= 0):
        raise ValueError("los nodos x tienen que estar ordenados de menor a mayor")

    n = len(x) - 1
    h = np.diff(x)
    M = np.zeros(n + 1)
    if n == 1:
        return M  # con dos nodos el spline natural es la recta que los une

    # Sistema para las incógnitas M_1, ..., M_{n−1}
    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(1, n):
        a[i - 1] = h[i - 1]
        b[i - 1] = 2 * (h[i - 1] + h[i])
        c[i - 1] = h[i]
        d[i - 1] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M[1:n] = resolver_tridiagonal(a, b, c, d)
    return M


def spline_natural(x, y, x_eval):
    """Spline cúbico natural: un polinomio cúbico por intervalo, con S, S' y S''
    continuas en los nodos y S'' = 0 en los extremos.

    En [x_i, x_{i+1}], con h_i = x_{i+1} − x_i:
        S(x) = M_i·(x_{i+1} − x)³ / 6h_i + M_{i+1}·(x − x_i)³ / 6h_i
             + (y_i/h_i − M_i·h_i/6)·(x_{i+1} − x)
             + (y_{i+1}/h_i − M_{i+1}·h_i/6)·(x − x_i)

    Fuera de [x_0, x_n] se extiende el cúbico del intervalo más cercano.
    """
    x, y = _validar(x, y, 2)
    M = segundas_derivadas_spline(x, y)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x) - 1

    # Intervalo i de cada punto: el i tal que x_i ≤ x_eval < x_{i+1}
    i = np.searchsorted(x, x_eval, side="right") - 1
    i = np.clip(i, 0, n - 1)

    h_i = x[i + 1] - x[i]
    izq = x[i + 1] - x_eval  # distancia al nodo de la derecha
    der = x_eval - x[i]      # distancia al nodo de la izquierda

    return (M[i] * izq**3 / (6 * h_i)
            + M[i + 1] * der**3 / (6 * h_i)
            + (y[i] / h_i - M[i] * h_i / 6) * izq
            + (y[i + 1] / h_i - M[i + 1] * h_i / 6) * der)
