"""Etapa 0: preparación de una corrida.

Pasa del contador de bytes del sistema a los bytes de la descarga, y ubica
cuándo empieza y cuándo termina. Orden de cálculo:

    B  →  tasa de fondo r_f  →  B_d = B − r_f·t  →  v∞  →  v = dB_d/dt
       →  t_ini y t_fin  →  sobrecarga p

Unidades: bytes y segundos.
"""

import numpy as np
import pandas as pd

from metodos.derivacion import derivada_3_puntos
from metodos.regresion import regresion_lineal


def cargar_corrida(ruta):
    """Lee el CSV de una captura. Devuelve (t, B), con B(t_i) = C(t_i) − C(t_0)."""
    datos = pd.read_csv(ruta)
    t = datos["t_s"].to_numpy(dtype=float)
    C = datos["bytes_recv"].to_numpy(dtype=float)
    return t, C - C[0]


def paso_medio(t):
    """Paso medio entre lecturas y su desvío estándar. Devuelve (h, desvío)."""
    pasos = np.diff(t)
    return np.mean(pasos), np.std(pasos)


def submuestrear(t, B, k):
    """Se queda con una lectura de cada k: el paso pasa de h a k·h."""
    return t[::k], B[::k]


def tasa_de_fondo(t, B, ventana=30.0):
    """Tasa del tráfico de fondo r_f, en B/s. Devuelve (r_f, r_inicio, r_final).

    Recta de mínimos cuadrados sobre los primeros y los últimos `ventana`
    segundos, cuando no hay descarga. Cada tramo se ajusta por separado,
    porque entre los dos está el salto de la descarga, y r_f es el promedio
    de las dos pendientes.
    """
    inicio = t <= t[0] + ventana
    final = t >= t[-1] - ventana
    _, r_inicio = regresion_lineal(t[inicio], B[inicio])
    _, r_final = regresion_lineal(t[final], B[final])
    return (r_inicio + r_final) / 2, r_inicio, r_final


def quitar_fondo(t, B, r_f):
    """Bytes de la descarga:  B_d(t) = B(t) − r_f·t."""
    return B - r_f * t


def velocidad_regimen(t, B_d, desde=0.2, hasta=0.8):
    """Velocidad en régimen v∞, en B/s.

    Pendiente de la recta de mínimos cuadrados de B_d(t) entre el 20 % y el
    80 % de los bytes de la descarga. Ahí el arranque ya pasó y el final
    todavía no llegó.
    """
    total = B_d[-1]
    tramo = (B_d >= desde * total) & (B_d <= hasta * total)
    _, v_inf = regresion_lineal(t[tramo], B_d[tramo])
    return v_inf


def limites_descarga(v, v_inf, fraccion=0.1):
    """Índices del primer y del último instante con v > 0,1·v∞. Devuelve (i_ini, i_fin)."""
    activos = np.nonzero(v > fraccion * v_inf)[0]
    if len(activos) == 0:
        raise ValueError("ninguna velocidad supera el umbral: no se encontró la descarga")
    return activos[0], activos[-1]


def sobrecarga(bytes_descarga, S):
    """Sobrecarga de protocolo:  p = (B_d(t_fin) − S) / S."""
    return (bytes_descarga - S) / S


def preparar_corrida(t, B, S, ventana=30.0):
    """Aplica toda la etapa 0 a una corrida. Devuelve un diccionario con los resultados.

    La velocidad se calcula con la fórmula de 3 puntos sobre B_d, con h igual
    al paso medio de la captura.
    """
    h, desvio_h = paso_medio(t)
    r_f, r_inicio, r_final = tasa_de_fondo(t, B, ventana)
    B_d = quitar_fondo(t, B, r_f)
    v_inf = velocidad_regimen(t, B_d)
    v = derivada_3_puntos(B_d, h)
    i_ini, i_fin = limites_descarga(v, v_inf)

    return {
        "h": h,
        "desvio_h": desvio_h,
        "r_f": r_f,
        "r_inicio": r_inicio,
        "r_final": r_final,
        "B_d": B_d,
        "v": v,
        "v_inf": v_inf,
        "i_ini": i_ini,
        "i_fin": i_fin,
        "t_ini": t[i_ini],
        "t_fin": t[i_fin],
        "p": sobrecarga(B_d[i_fin], S),
    }
