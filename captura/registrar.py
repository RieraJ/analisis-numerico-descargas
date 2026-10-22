"""Registra el contador de bytes recibidos de una interfaz de red cada h segundos.

Cada fila del CSV tiene:
    t_s         segundos desde el inicio (time.perf_counter)
    bytes_recv  contador del sistema C(t), en bytes
    fecha_hora  fecha y hora ISO, solo como referencia

Uso:
    python captura/registrar.py --listar
    python captura/registrar.py --interfaz wlp0s20f3 --intervalo 1 --salida datos/c1.csv
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import psutil


def listar_interfaces():
    """Muestra las interfaces con sus bytes recibidos, de mayor a menor."""
    contadores = psutil.net_io_counters(pernic=True)
    ordenadas = sorted(contadores.items(), key=lambda par: par[1].bytes_recv, reverse=True)
    print(f"{'interfaz':<20} {'bytes recibidos':>18} {'MB':>12}")
    for nombre, contador in ordenadas:
        print(f"{nombre:<20} {contador.bytes_recv:>18,} {contador.bytes_recv / 1e6:>12.1f}")


def leer_bytes(interfaz):
    """Devuelve el contador de bytes recibidos de la interfaz."""
    return psutil.net_io_counters(pernic=True)[interfaz].bytes_recv


def registrar(interfaz, h, salida):
    """Lee el contador en t0 + n·h y escribe una fila por lectura.

    Cada lectura se agenda respecto del instante inicial t0 (no de la lectura
    anterior), así el paso no acumula atraso. Si una lectura llega tarde por
    más de un paso, se saltean los instantes ya pasados.
    """
    if interfaz not in psutil.net_io_counters(pernic=True):
        raise SystemExit(f"No existe la interfaz '{interfaz}'. Ver --listar.")

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    lecturas = 0

    with open(salida, "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["t_s", "bytes_recv", "fecha_hora"])
        print(f"Registrando '{interfaz}' cada {h} s en {salida}. Ctrl+C para terminar.")

        t0 = time.perf_counter()
        bytes_inicio = None
        n = 0
        try:
            while True:
                # Esperar hasta el instante agendado t0 + n·h
                espera = t0 + n * h - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)

                t = time.perf_counter() - t0
                bytes_recv = leer_bytes(interfaz)
                fecha_hora = datetime.now().isoformat(timespec="milliseconds")
                escritor.writerow([f"{t:.6f}", bytes_recv, fecha_hora])
                archivo.flush()
                lecturas += 1

                if bytes_inicio is None:
                    bytes_inicio = bytes_recv
                mb = (bytes_recv - bytes_inicio) / 1e6
                print(f"\r{lecturas:6d} lecturas   t = {t:8.1f} s   B = {mb:10.2f} MB",
                      end="", flush=True)

                # Siguiente instante agendado que todavía no pasó
                n += 1
                while t0 + n * h <= time.perf_counter():
                    n += 1
        except KeyboardInterrupt:
            pass

    print(f"\nSe guardaron {lecturas} lecturas en {salida}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listar", action="store_true",
                        help="mostrar las interfaces de red y salir")
    parser.add_argument("--interfaz", help="interfaz a registrar (ver --listar)")
    parser.add_argument("--intervalo", type=float, default=1.0,
                        help="paso de muestreo h en segundos (default: 1)")
    parser.add_argument("--salida", default="datos/prueba.csv",
                        help="archivo CSV de salida (default: datos/prueba.csv)")
    args = parser.parse_args()

    if args.listar:
        listar_interfaces()
        return
    if args.interfaz is None:
        parser.error("falta --interfaz (o usar --listar)")
    if args.intervalo <= 0:
        parser.error("--intervalo tiene que ser positivo")

    registrar(args.interfaz, args.intervalo, args.salida)


if __name__ == "__main__":
    main()
