"""Etapa 2: volumen descargado a partir de la velocidad.

Un sistema que solo ve el velocímetro cada H segundos estima los bytes
descargados como D = ∫ v dt, por trapecios o por Simpson. La referencia
exacta es lo que marcó el contador en el mismo intervalo: B_d(b) − B_d(a).

Unidades: bytes y segundos.
"""

import numpy as np

from analisis.preparacion import paso_medio
from metodos.integracion import simpson, trapecios


def indices_muestreo(i_ini, i_fin, k, desfase, n_lecturas):
    """Índices de las muestras, una cada k lecturas, que cubren toda la descarga.

    La última muestra es i_fin + desfase (0 ≤ desfase < k) y las demás van
    hacia atrás de a k hasta llegar a i_ini o pasarlo. Así el paso es
    uniforme y la integral cubre [t_ini, t_fin] entero; lo que queda de más
    en los bordes es tiempo con v casi nula.
    """
    if not 0 <= desfase < k:
        raise ValueError("el desfase tiene que estar entre 0 y k − 1")
    b = i_fin + desfase
    pasos = int(np.ceil((b - i_ini) / k))
    a = b - pasos * k
    if a < 0 or b > n_lecturas - 1:
        raise ValueError("las muestras se salen de la captura")
    return np.arange(a, b + 1, k)


def volumen_desde_velocidad(t, v, B_d, i_ini, i_fin, k, desfase=0):
    """D = ∫ v dt con muestras de v cada H = k·h, por trapecios y por Simpson.

    Devuelve un diccionario con los índices usados, H, las dos estimaciones
    y la referencia B_d(b) − B_d(a) en el mismo intervalo [t_a, t_b].
    """
    indices = indices_muestreo(i_ini, i_fin, k, desfase, len(t))
    H, _ = paso_medio(t[indices])
    return {
        "indices": indices,
        "H": H,
        "trapecios": trapecios(v[indices], H),
        "simpson": simpson(v[indices], H),
        "referencia": B_d[indices[-1]] - B_d[indices[0]],
    }


def errores_por_desfase(t, v, B_d, i_ini, i_fin, k):
    """Error relativo (D − referencia) / referencia para los k desfases posibles.

    Con muestras cada k lecturas hay k formas de ubicarlas (desfase 0 a k − 1).
    Devuelve un diccionario con un array de k errores para trapecios y otro
    para Simpson.
    """
    errores = {"trapecios": np.empty(k), "simpson": np.empty(k)}
    for desfase in range(k):
        r = volumen_desde_velocidad(t, v, B_d, i_ini, i_fin, k, desfase)
        for metodo in errores:
            errores[metodo][desfase] = (r[metodo] - r["referencia"]) / r["referencia"]
    return errores
