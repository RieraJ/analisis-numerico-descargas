# Proyecto: análisis numérico de descargas

Proyecto de la materia **Análisis Numérico** (Ingeniería en Sistemas de Información, UCC). La consigna es tomar una aplicación real de un método numérico en ingeniería, plantear el problema y resolverlo con métodos vistos en la materia. Se entrega con un informe y se **defiende en una exposición oral**, así que todo el código tiene que poder explicarse línea por línea.

**Problema:** estimar los datos descargados y el tiempo restante de una descarga a partir de mediciones reales de velocidad. Es lo que hacen el contador de datos móviles, la facturación por volumen y el "quedan 3 minutos" de un navegador.

## Reglas de trabajo

- **Los métodos numéricos se implementan desde cero** en `metodos/`, como funciones puras sobre arrays de NumPy. Prohibido dentro de `metodos/`: `scipy.integrate`, `scipy.interpolate`, `scipy.optimize`, `np.polyfit`, `np.gradient`, `np.trapz`.
- **SciPy solo se usa en `tests/`**, para verificar los métodos propios.
- Cada método tiene tests con funciones de **solución exacta conocida** y, cuando corresponde, se verifica el **orden de convergencia** (trapecios ~h², Simpson ~h⁴, 3 puntos ~h², 5 puntos ~h⁴).
- Código, nombres, comentarios y docstrings **en español**. Docstrings cortos con la fórmula usada y el orden de error.
- Código simple y legible antes que optimizado: es material de exposición.
- Unidades internas: **bytes y segundos**. Conversiones solo al mostrar. 1 MB/s = 8 Mbps; 1 MB = 10⁶ B; 1 MiB = 2²⁰ B.
- No inventar datos: los análisis se corren sobre CSV reales en `datos/`. Para probar métodos, usar funciones sintéticas en los tests.
- Trabajar de a un paso, con commits chicos y mensajes en español. Al terminar cada paso, actualizar la lista de estado de este archivo.

## Stack

Python 3 con psutil, NumPy, pandas, Matplotlib, SciPy (solo tests), pytest y Jupyter. Entorno virtual en `.venv/`.

## Estructura

```
captura/registrar.py   registra el contador de bytes de una interfaz cada h segundos → CSV
captura/graficar.py    revisa una corrida: B(t) y paso real de muestreo
datos/                 un CSV por corrida
metodos/
  derivacion.py        fórmulas centradas de 3 y 5 puntos (descentradas del mismo orden en los extremos)
  integracion.py       trapecios y Simpson 1/3 (paso uniforme; si los intervalos son impares, el último por trapecios)
  interpolacion.py     diferencias divididas de Newton, Lagrange, spline cúbico natural
  regresion.py         mínimos cuadrados lineal (ecuaciones normales resueltas a mano)
  raices.py            bisección, punto fijo, Newton-Raphson, secante (devuelven raíz + historial de iteraciones)
tests/                 pytest, un archivo por módulo de metodos/
analisis.ipynb         experimentos, tablas y gráficos del informe
requirements.txt, .gitignore (.venv, __pycache__, .ipynb_checkpoints, *.iso), README.md
```

## Especificación de la captura

`captura/registrar.py`, por línea de comandos (argparse):
- `--listar`: muestra las interfaces de `psutil.net_io_counters(pernic=True)` con sus bytes recibidos, ordenadas de mayor a menor.
- `--interfaz`, `--intervalo` (default 1 s), `--salida` (default `datos/prueba.csv`).
- Escribe un CSV con columnas `t_s` (segundos desde el inicio, con `time.perf_counter`), `bytes_recv` y `fecha_hora` (ISO, solo referencia). Hace flush en cada fila.
- Agenda cada lectura respecto del instante inicial (`t0 + n·h`), no de la lectura anterior, para que el paso no acumule atraso.
- Termina limpio con Ctrl+C informando cuántas lecturas guardó.

`captura/graficar.py <csv>`: imprime cantidad de lecturas, duración, bytes recibidos y estadísticas del paso real; grafica B(t) en MB y el paso real contra t.

## Modelo matemático

1. **Datos acumulados:** `B(t_i) = C(t_i) − C(t_0)`, con `C` el contador del sistema.
2. **Velocidad:** `v = dB/dt` por derivación numérica.
   - 3 puntos: `v_i ≈ (B_{i+1} − B_{i−1}) / 2h`
   - 5 puntos: `v_i ≈ (−B_{i+2} + 8B_{i+1} − 8B_{i−1} + B_{i−2}) / 12h`
3. **Datos descargados:** `D(t) = ∫ v(τ) dτ` desde t₀, por trapecios y Simpson. Se compara `D(t_f)` con el tamaño exacto del archivo y con `B(t_f)`.
4. **Modelo de velocidad (arranque de TCP):** `v̂(t) = v∞ (1 − e^(−t/τ))`, de donde `D̂(t) = v∞ [t − τ(1 − e^(−t/τ))]`.
   - `v∞`: pendiente de `B(t)` en régimen, por regresión lineal.
   - `τ`: regresión lineal sobre `ln(1 − v/v∞) = −t/τ`.
5. **Instante de fin:** raíz de `f(t) = D̂(t) − S = 0`, con `S` el tamaño del archivo.
   - Newton usa `f'(t) = v̂(t)`.
   - Punto fijo: `t = φ(t) = S/v∞ + τ(1 − e^(−t/τ))`, con `φ'(t) = e^(−t/τ)` en (0, 1) para t > 0, así que converge.
   - Comparar contra la estimación ingenua: (lo que falta) / (velocidad actual).

## Etapas del análisis

| Etapa | Métodos | Referencia para validar |
|---|---|---|
| 1. Velocidad desde el contador | 3 vs. 5 puntos | Velocidad media por regresión |
| 2. Datos descargados | Trapecios vs. Simpson | Tamaño exacto del archivo y `B(t_f)` |
| 3. Efecto del muestreo | Misma integración con h = 1, 5, 10, 30 s (log-log) | Resultado con h = 1 s |
| 4. Parámetros del modelo | Regresión lineal | — |
| 5. Muestras faltantes | Newton, Lagrange, spline: borrar muestras reales y reconstruirlas | Valores borrados |
| 6. Instante de fin | Bisección, punto fijo, Newton, secante | Duración real de la descarga |
| Opcional: EDO | Euler, Heun, RK4 sobre `dv/dt = (v∞ − v)/τ` | Solución exacta |

Dato para el informe: con `dD/dt = v(t)` (lado derecho que no depende de D), Heun equivale a trapecios y RK4 a Simpson 1/3.

## Corridas previstas

Todas por Wi-Fi (la notebook no tiene puerto de red), en el departamento de Julian (red sin otros usuarios).

| Corrida | Banda | Ubicación | Carga extra |
|---|---|---|---|
| C1 | 5 GHz | Cerca del router | Ninguna (referencia) |
| C2 | 5 GHz | Cerca del router | Netflix en el TV |
| C3 | 2.4 GHz | Cerca del router | Ninguna |
| C4 | 2.4 GHz | Lejos del router | Ninguna |

Comparaciones (cambia una sola variable): C1 vs C2 → otro dispositivo usando la red; C1 vs C3 → banda; C3 vs C4 → distancia.

Procedimiento: 10 s de registro sin descargar (tráfico de fondo), descarga completa de un archivo grande de tamaño conocido (por ejemplo, una ISO de Linux), 10 s más de registro. Anotar el tamaño exacto en bytes (`stat -c %s archivo`).
- Mismo archivo en las cuatro corridas y en horarios parecidos.
- C2: Netflix arrancado antes del registro, calidad fija (alta), TV conectado por Wi-Fi.
- No editar los CSV después de la captura.

## Estado

- [x] Estructura base: requirements.txt, .gitignore, README.md, entorno virtual
- [x] Captura: `registrar.py` y `graficar.py`, probados con una corrida corta
- [ ] Corridas C1 a C4 (las hace Julian)
- [x] `metodos/derivacion.py` e `integracion.py` con tests (núcleo del proyecto)
- [x] `metodos/regresion.py` e `interpolacion.py` con tests
- [x] `metodos/raices.py` con tests
- [ ] `analisis.ipynb` con las seis etapas
- [ ] Informe y exposición
