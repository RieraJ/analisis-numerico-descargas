"""Revisa una corrida: resumen por consola, gráfico de B(t) y del paso real de muestreo.

B(t_i) = C(t_i) - C(t_0), con C el contador de bytes del sistema.

Uso:
    python captura/graficar.py datos/c1.csv
    python captura/graficar.py datos/c1.csv --guardar c1.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="archivo CSV generado por registrar.py")
    parser.add_argument("--guardar", help="guardar el gráfico en este archivo en lugar de mostrarlo")
    args = parser.parse_args()

    datos = pd.read_csv(args.csv)
    if len(datos) < 2:
        raise SystemExit("La corrida necesita al menos 2 lecturas.")

    t = datos["t_s"].to_numpy()
    contador = datos["bytes_recv"].to_numpy()
    B = contador - contador[0]  # datos acumulados desde t_0, en bytes
    paso = np.diff(t)           # paso real entre lecturas, en segundos

    print(f"Lecturas:         {len(t)}")
    print(f"Duración:         {t[-1] - t[0]:.2f} s")
    print(f"Bytes recibidos:  {B[-1]:,} B  ({B[-1] / 1e6:.2f} MB)")
    print("Paso real [s]:")
    print(f"  media           {paso.mean():.6f}")
    print(f"  desvío          {paso.std():.6f}")
    print(f"  mínimo          {paso.min():.6f}")
    print(f"  máximo          {paso.max():.6f}")

    fig, (ax_b, ax_paso) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    ax_b.plot(t, B / 1e6)
    ax_b.set_ylabel("B(t) [MB]")
    ax_b.set_title(Path(args.csv).name)
    ax_b.grid(True)

    ax_paso.plot(t[1:], paso, ".", markersize=3)
    ax_paso.set_ylabel("paso real [s]")
    ax_paso.set_xlabel("t [s]")
    ax_paso.grid(True)

    fig.tight_layout()
    if args.guardar:
        fig.savefig(args.guardar, dpi=120)
        print(f"Gráfico guardado en {args.guardar}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
