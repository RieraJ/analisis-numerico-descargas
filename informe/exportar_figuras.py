"""Guarda como PNG los gráficos que ya tiene analisis.ipynb, para el informe.

No vuelve a dibujar nada: toma las salidas image/png del notebook en el orden
en que aparecen. Si el notebook cambia, primero se ejecuta y después se corre esto.

Uso:
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace analisis.ipynb
    python informe/exportar_figuras.py
"""

import base64
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "analisis.ipynb"
CARPETA = RAIZ / "informe" / "figuras"

# Un nombre por gráfico, en el orden del notebook.
NOMBRES = [
    "fig1_bytes_descarga.png",
    "fig2_velocimetro_c1.png",
    "fig3_volumen_h1.png",
    "fig4_loglog_muestreo.png",
    "fig5_rampa_modelo.png",
    "fig6_faltantes.png",
    "fig7_prediccion.png",
    "fig8_edo_orden.png",
]


def imagenes(notebook):
    """Devuelve (índice de celda, bytes del PNG) de cada gráfico, en orden."""
    encontradas = []
    for i, celda in enumerate(notebook.cells):
        for salida in celda.get("outputs", []):
            if "image/png" in salida.get("data", {}):
                encontradas.append((i, base64.b64decode(salida["data"]["image/png"])))
    return encontradas


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    encontradas = imagenes(notebook)
    if len(encontradas) != len(NOMBRES):
        raise SystemExit(f"El notebook tiene {len(encontradas)} gráficos y se esperaban "
                         f"{len(NOMBRES)}: revisar NOMBRES.")

    CARPETA.mkdir(exist_ok=True)
    for nombre, (i, png) in zip(NOMBRES, encontradas):
        (CARPETA / nombre).write_bytes(png)
        print(f"celda {i:2d} → {nombre} ({len(png) // 1024} KB)")


if __name__ == "__main__":
    main()
