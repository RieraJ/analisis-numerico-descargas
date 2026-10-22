"""Búsqueda de raíces de f(x) = 0.

Todas las funciones devuelven (raiz, historial), donde historial es la lista
de aproximaciones x_1, x_2, ..., una por iteración (sin los valores iniciales).
Así len(historial) es la cantidad de iteraciones y se puede armar la tabla
de convergencia.

Criterio de parada: |x_{k+1} − x_k| < tol (en bisección, semiancho del
intervalo < tol). Si no se cumple en max_iter iteraciones, se lanza un error.
"""


def biseccion(f, a, b, tol=1e-10, max_iter=200):
    """Bisección. Convergencia lineal: el error se divide por 2 en cada paso.

    Requiere f(a)·f(b) < 0. En cada paso se toma c = (a + b)/2 y se conserva
    la mitad del intervalo donde f cambia de signo.
    """
    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a, []
    if fb == 0:
        return b, []
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) tienen que tener distinto signo")

    historial = []
    for _ in range(max_iter):
        c = (a + b) / 2
        historial.append(c)
        fc = f(c)

        if fc == 0 or (b - a) / 2 < tol:
            return c, historial

        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc

    raise RuntimeError(f"bisección no convergió en {max_iter} iteraciones")


def punto_fijo(phi, x0, tol=1e-10, max_iter=200):
    """Iteración de punto fijo  x_{k+1} = φ(x_k).  Convergencia lineal.

    Converge si |φ'(x)| < 1 cerca de la raíz; el error se multiplica
    aproximadamente por |φ'(r)| en cada paso.
    """
    historial = []
    x = x0
    for _ in range(max_iter):
        x_nuevo = phi(x)
        historial.append(x_nuevo)
        if abs(x_nuevo - x) < tol:
            return x_nuevo, historial
        x = x_nuevo

    raise RuntimeError(f"punto fijo no convergió en {max_iter} iteraciones")


def newton_raphson(f, df, x0, tol=1e-10, max_iter=100):
    """Newton-Raphson  x_{k+1} = x_k − f(x_k) / f'(x_k).  Convergencia cuadrática.

    Cerca de una raíz simple, el error del paso siguiente es proporcional
    al cuadrado del error actual.
    """
    historial = []
    x = x0
    for _ in range(max_iter):
        derivada = df(x)
        if derivada == 0:
            raise ZeroDivisionError(f"f'(x) = 0 en x = {x}")

        x_nuevo = x - f(x) / derivada
        historial.append(x_nuevo)
        if abs(x_nuevo - x) < tol:
            return x_nuevo, historial
        x = x_nuevo

    raise RuntimeError(f"Newton-Raphson no convergió en {max_iter} iteraciones")


def secante(f, x0, x1, tol=1e-10, max_iter=100):
    """Secante  x_{k+1} = x_k − f(x_k)·(x_k − x_{k−1}) / (f(x_k) − f(x_{k−1})).

    Es Newton con la derivada reemplazada por la pendiente de la recta entre
    los dos últimos puntos. Convergencia superlineal, de orden (1 + √5)/2 ≈ 1,618.
    """
    historial = []
    f0 = f(x0)
    f1 = f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ZeroDivisionError(f"f(x_k) = f(x_(k-1)) en x = {x1}")

        x_nuevo = x1 - f1 * (x1 - x0) / (f1 - f0)
        historial.append(x_nuevo)
        if abs(x_nuevo - x1) < tol:
            return x_nuevo, historial

        x0, f0 = x1, f1
        x1, f1 = x_nuevo, f(x_nuevo)

    raise RuntimeError(f"secante no convergió en {max_iter} iteraciones")
