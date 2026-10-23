"""Etapa 5: muestras faltantes.

Se borra un bloque de k lecturas consecutivas de B_d y se reconstruyen con:
- lineal: Newton con un vecino a cada lado (grado 1, velocidad constante en el hueco);
- Newton y Lagrange con dos vecinos a cada lado (grado 3);
- spline cúbico natural con todas las lecturas que quedan.

La referencia son los valores borrados.

Unidades: bytes y segundos.
"""

import numpy as np

from metodos.interpolacion import lagrange, newton, spline_natural


def indices_vecinos(inicio, k, por_lado):
    """Índices de las `por_lado` lecturas a cada lado del hueco [inicio, inicio + k)."""
    return np.r_[inicio - por_lado:inicio, inicio + k:inicio + k + por_lado]


def reconstruir_bloque(t, B, inicio, k, vecinos=2):
    """Borra las lecturas inicio, ..., inicio + k − 1 y las reconstruye.

    Devuelve un diccionario con los instantes borrados, los valores reales,
    la reconstrucción de cada método y los bytes que entraron en el hueco,
    B(t_{inicio+k}) − B(t_{inicio−1}).
    """
    if inicio - vecinos < 0 or inicio + k + vecinos > len(t):
        raise ValueError("el hueco y sus vecinos se salen de la captura")

    borrados = np.arange(inicio, inicio + k)
    quedan = np.ones(len(t), dtype=bool)
    quedan[borrados] = False
    t_b = t[borrados]

    lin = indices_vecinos(inicio, k, 1)
    cerca = indices_vecinos(inicio, k, vecinos)
    return {
        "t": t_b,
        "real": B[borrados],
        "lineal": newton(t[lin], B[lin], t_b),
        "newton": newton(t[cerca], B[cerca], t_b),
        "lagrange": lagrange(t[cerca], B[cerca], t_b),
        "spline": spline_natural(t[quedan], B[quedan], t_b),
        "bytes_hueco": B[inicio + k] - B[inicio - 1],
    }


def error_maximo(reconstruccion, metodo):
    """Mayor |B̂ − B| sobre las lecturas borradas, en bytes."""
    return np.max(np.abs(reconstruccion[metodo] - reconstruccion["real"]))


def posiciones_equiespaciadas(indices, k, cantidad=100):
    """`cantidad` comienzos de hueco repartidos de forma pareja entre indices[0] e indices[-1] − k.

    Se usan posiciones fijas, no al azar, para que el resultado se pueda repetir.
    """
    return np.linspace(indices[0], indices[-1] - k, cantidad).astype(int)


def hueco_con_caida(v, v_inf, inicio, k, fraccion=0.5):
    """True si en el hueco (o en las lecturas que lo tocan) v baja de 0,5·v∞."""
    return bool(np.min(v[inicio - 1:inicio + k + 1]) < fraccion * v_inf)
