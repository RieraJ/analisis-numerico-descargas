"""Etapa 6: predicción del instante de fin durante la descarga.

Con los datos hasta un cierto progreso (25, 50, 75 % de S'), se estiman v∞ y τ
y se resuelve  f(t) = D̂(t) − S' = 0  con los cuatro métodos de raíces. El fin
previsto es t_ini + t*. Se compara con dos estimaciones ingenuas:
(lo que falta) / (velocidad actual), con dos formas de medir la velocidad actual.

Unidades: bytes y segundos.
"""

import numpy as np

from analisis.modelo import ajustar_tau, f_fin, phi_fin, velocidad_modelo
from analisis.preparacion import limites_descarga, velocidad_regimen
from metodos.derivacion import derivada_3_puntos
from metodos.raices import biseccion, newton_raphson, punto_fijo, secante


def indice_progreso(B_d, S_prima, fraccion):
    """Primera lectura con B_d ≥ fracción·S'."""
    return int(np.nonzero(B_d >= fraccion * S_prima)[0][0])


def estimar_parametros(t, B_d, h, i_corte):
    """v∞, t_ini y τ usando solo las lecturas hasta i_corte (lo que se conoce en ese momento).

    v∞: pendiente de B_d entre el 20 % y el 80 % de lo descargado hasta ahora.
    v: derivada de 3 puntos de los datos parciales (en el último punto, la regresiva).
    """
    t_p, B_p = t[:i_corte + 1], B_d[:i_corte + 1]
    v_inf = velocidad_regimen(t_p, B_p)
    v = derivada_3_puntos(B_p, h)
    i_ini, _ = limites_descarga(v, v_inf)
    tau = ajustar_tau(t_p, v, v_inf, i_ini)["tau"]
    return {"v_inf": v_inf, "tau": tau, "i_ini": i_ini, "v": v}


def resolver_fin(v_inf, tau, S_prima):
    """Raíz de f(t) = D̂(t) − S' con los cuatro métodos. Devuelve {método: (raíz, historial)}.

    Valores iniciales: x0 = S'/v∞ (velocidad constante desde el principio).
    Bisección en [0, 2·S'/v∞], donde f cambia de signo. Secante con x0 y 1,1·x0.
    """
    f = lambda s: f_fin(s, v_inf, tau, S_prima)
    x0 = S_prima / v_inf
    return {
        "bisección": biseccion(f, 0.0, 2 * x0),
        "punto fijo": punto_fijo(lambda s: phi_fin(s, v_inf, tau, S_prima), x0),
        "Newton": newton_raphson(f, lambda s: velocidad_modelo(s, v_inf, tau), x0),
        "secante": secante(f, x0, 1.1 * x0),
    }


def fines_ingenuos(t, B_d, v, i_corte, S_prima, ventana=10):
    """Fin previsto como t_actual + (S' − B_d actual) / v_actual, con dos v_actual:

    - instantánea: la derivada de 3 puntos en la última lectura;
    - último segundo: (B_d(t_i) − B_d(t_{i−10})) / (t_i − t_{i−10}).
    Devuelve (fin_instantanea, fin_ultimo_segundo).
    """
    falta = S_prima - B_d[i_corte]
    v_ultimo = (B_d[i_corte] - B_d[i_corte - ventana]) / (t[i_corte] - t[i_corte - ventana])
    return t[i_corte] + falta / v[i_corte], t[i_corte] + falta / v_ultimo


def predecir_fin(t, B_d, h, S_prima, fraccion):
    """Toda la predicción con los datos hasta fracción·S'. Devuelve un diccionario."""
    i_corte = indice_progreso(B_d, S_prima, fraccion)
    p = estimar_parametros(t, B_d, h, i_corte)
    if np.isnan(p["tau"]):
        raise ValueError("τ quedó sin determinar (rampa con muy pocos puntos)")
    raices = resolver_fin(p["v_inf"], p["tau"], S_prima)
    duracion = raices["Newton"][0]
    fin_inst, fin_seg = fines_ingenuos(t, B_d, p["v"], i_corte, S_prima)
    return {
        "t_corte": t[i_corte],
        "v_inf": p["v_inf"],
        "tau": p["tau"],
        "t_ini": t[p["i_ini"]],
        "raices": raices,
        "fin_modelo": t[p["i_ini"]] + duracion,
        "fin_instantanea": fin_inst,
        "fin_ultimo_segundo": fin_seg,
    }


def velocidad_por_tramos(t, B_d, i_ini, i_fin, partes=5):
    """Velocidad media en cada quinto de los bytes de la descarga, entre t_ini y t_fin.

    Sirve para ver si la velocidad fue estable (supuesto del modelo) o cambió.
    """
    t_d, B_dd = t[i_ini:i_fin + 1], B_d[i_ini:i_fin + 1]
    cortes = np.linspace(B_dd[0], B_dd[-1], partes + 1)
    indices = [min(int(np.searchsorted(B_dd, c)), len(B_dd) - 1) for c in cortes]
    return np.array([(B_dd[b] - B_dd[a]) / (t_d[b] - t_d[a]) for a, b in zip(indices[:-1], indices[1:])])
