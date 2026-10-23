"""Tests de analisis/prediccion.py con la descarga sintética de tests/test_preparacion.py."""

import numpy as np
import pytest

from analisis.prediccion import indice_progreso, predecir_fin, resolver_fin, velocidad_por_tramos
from tests.test_preparacion import H, T_A, T_B, T_TOTAL, TAU, V_INF, descargado

t = np.arange(0, T_TOTAL + H / 2, H)
B_d = descargado(t)
S_PRIMA = descargado(T_B)  # todo lo que se descarga


def test_indice_progreso():
    B = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    assert indice_progreso(B, 50, 0.5) == 3   # primera lectura con B ≥ 25


def test_los_cuatro_metodos_coinciden_y_dan_la_aproximacion_asintotica():
    raices = resolver_fin(V_INF, TAU, S_PRIMA)
    valores = [raiz for raiz, _ in raices.values()]
    assert max(valores) - min(valores) < 1e-8
    # Con t ≫ τ, e^(−t/τ) ≈ 0 y la raíz es S'/v∞ + τ
    assert valores[0] == pytest.approx(S_PRIMA / V_INF + TAU)
    # Newton converge en muy pocas iteraciones; bisección necesita muchas más
    assert len(raices["Newton"][1]) <= 3
    assert len(raices["bisección"][1]) > 30


@pytest.mark.parametrize("fraccion", [0.25, 0.5, 0.75])
def test_prediccion_del_fin_en_la_descarga_sintetica(fraccion):
    r = predecir_fin(t, B_d, H, S_PRIMA, fraccion)
    assert r["v_inf"] == pytest.approx(V_INF)
    # t_ini queda un paso después del arranque y τ sale subestimado (etapa 4):
    # el fin previsto se corre a lo sumo un par de pasos.
    assert r["t_ini"] == pytest.approx(T_A, abs=1.01 * H)
    assert r["fin_modelo"] == pytest.approx(T_B, abs=2 * H)
    # Con velocidad constante, las dos estimaciones ingenuas son exactas
    assert r["fin_instantanea"] == pytest.approx(T_B, abs=1e-6)
    assert r["fin_ultimo_segundo"] == pytest.approx(T_B, abs=1e-6)


def test_velocidad_por_tramos():
    # Primera mitad a 10 B/s y segunda a 30 B/s, en tiempos de a 1 s
    tt = np.arange(0, 21, dtype=float)
    B = np.where(tt <= 10, 10 * tt, 100 + 30 * (tt - 10))
    # Bytes: 0 a 400. El corte del medio (200 B) se toma en la primera lectura
    # con B ≥ 200: t = 14 s, B = 220.
    v = velocidad_por_tramos(tt, B, 0, 20, partes=2)
    assert v[0] == pytest.approx(220 / 14)
    assert v[1] == pytest.approx(30)
