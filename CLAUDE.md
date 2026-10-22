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
datos/                 un CSV por corrida y corridas.csv (condiciones y tamaño exacto de cada una)
metodos/
  derivacion.py        fórmulas centradas de 3 y 5 puntos (descentradas del mismo orden en los extremos)
  integracion.py       trapecios y Simpson 1/3 (paso uniforme; si los intervalos son impares, el último por trapecios)
  interpolacion.py     diferencias divididas de Newton, Lagrange, spline cúbico natural
  regresion.py         mínimos cuadrados lineal (ecuaciones normales resueltas a mano)
  raices.py            bisección, punto fijo, Newton-Raphson, secante (devuelven raíz + historial de iteraciones)
tests/                 pytest, un archivo por módulo de metodos/
analisis.ipynb         experimentos, tablas y gráficos del informe
DEFENSA.md             preguntas probables en la exposición, con respuestas cortas
requirements.txt, .gitignore (.venv, __pycache__, .ipynb_checkpoints, *.iso, datos/prueba*.csv), README.md
```

## Especificación de la captura

`captura/registrar.py`, por línea de comandos (argparse):
- `--listar`: muestra las interfaces de `psutil.net_io_counters(pernic=True)` con sus bytes recibidos, ordenadas de mayor a menor.
- `--interfaz`, `--intervalo` (default 0,1 s), `--salida` (default `datos/prueba.csv`).
- Escribe un CSV con columnas `t_s` (segundos desde el inicio, con `time.perf_counter`), `bytes_recv` y `fecha_hora` (ISO, solo referencia). Hace flush en cada fila.
- Agenda cada lectura respecto del instante inicial (`t0 + n·h`), no de la lectura anterior, para que el paso no acumule atraso. Si una lectura se atrasa más de un paso, saltea los instantes ya pasados.
- Termina limpio con Ctrl+C informando cuántas lecturas guardó.

Por qué 0,1 s: el arranque de TCP dura unas décimas de segundo (en la prueba del 22/10, unos 0,3 s). Con 1 s no se ve. Capturar fino no cuesta nada: después se puede submuestrear, nunca al revés.

`captura/graficar.py <csv>`: imprime cantidad de lecturas, duración, bytes recibidos, estadísticas del paso real y lecturas salteadas; grafica B(t) en MB y el paso real contra t.

## Modelo matemático

1. **Datos acumulados:** `B(t_i) = C(t_i) − C(t_0)`, con `C` el contador del sistema.
2. **Tráfico de fondo y sobrecarga:** el contador cuenta todo lo que entra por la interfaz, encabezados incluidos.
   - Tasa de fondo `r_f`: pendiente por regresión lineal de B en los 30 s sin descarga del principio y del final. Bytes de la descarga: `B_d(t) = B(t) − r_f·t`.
   - Inicio y fin de la descarga (`t_ini`, `t_fin`): primer y último instante con `v > 0,1·v∞`. Los tiempos del modelo se miden desde `t_ini`.
   - Sobrecarga de protocolo: `p = (B_d(t_fin) − S) / S`, con `S` el tamaño del archivo. Se espera del orden de 3–5 % (encabezados Ethernet/IP/TCP/TLS); se mide, no se supone.
3. **Velocidad (velocímetro):** `v = dB/dt` por derivación numérica sobre la captura.
   - 3 puntos: `v_i ≈ (B_{i+1} − B_{i−1}) / 2h`
   - 5 puntos: `v_i ≈ (−B_{i+2} + 8B_{i+1} − 8B_{i−1} + B_{i−2}) / 12h`
   - Con datos reales hay un balance entre truncamiento y ruido: el error de truncamiento baja con h (h² o h⁴), pero el ruido de medición se amplifica como 1/h, y más en 5 puntos (√130/12 ≈ 0,95/h) que en 3 puntos (√2/2 ≈ 0,71/h). Por eso se compara con h = 0,1 y h = 1 s (submuestreando B). El orden teórico se muestra con los tests, no con los datos reales.
4. **Datos descargados a partir de la velocidad:** un sistema que solo ve la velocidad cada H segundos estima el volumen como `D = ∫ v dt` por trapecios y Simpson, con las muestras del velocímetro tomadas cada H. La referencia exacta es `B_d`.
   - Control: con todas las muestras (H = h), trapecios sobre la derivada de 3 puntos telescopa, `Σ (B_{i+1} − B_{i−1})/2 ≈ B_n − B_0`, y recupera B salvo los extremos. Es el teorema fundamental del cálculo, no una medida de precisión. El problema de integración de verdad aparece con H > h.
5. **Modelo de velocidad (arranque de TCP):** `v̂(t) = v∞ (1 − e^(−t/τ))`, con t desde `t_ini`, de donde `D̂(t) = v∞ [t − τ(1 − e^(−t/τ))]`.
   - `v∞`: pendiente de `B_d(t)` en régimen (entre el 20 % y el 80 % de los bytes de la descarga), por regresión lineal.
   - `τ`: regresión por el origen sobre `ln(1 − v/v∞) = −t/τ`, usando solo la rampa con `v < 0,9·v∞` (si `v ≥ v∞` el logaritmo no está definido).
   - Si la rampa tiene muy pocos puntos (τ del orden de h), se informa así: τ queda mal determinado y el modelo se reduce a velocidad constante. No se fuerza el ajuste.
6. **Instante de fin, como predicción durante la descarga:** con los datos hasta el 25 %, 50 % y 75 % del progreso, estimar v∞ y τ y resolver `f(t) = D̂(t) − S·(1 + p) = 0`. Se usa `p` medido en C1, la corrida de referencia.
   - La ecuación `t + τe^(−t/τ) = S'/v∞ + τ` (con `S' = S·(1 + p)`) es trascendente: su forma cerrada necesita la función W de Lambert. `S'/v∞ + τ` es la aproximación asintótica, con error `τ·e^(−t/τ)`.
   - Newton usa `f'(t) = v̂(t)`.
   - Punto fijo: `t = φ(t) = S'/v∞ + τ(1 − e^(−t/τ))`, con `φ'(t) = e^(−t/τ)` en (0, 1) para t > 0, así que converge.
   - Comparar contra la duración real (`t_fin − t_ini`) y contra la estimación ingenua: (lo que falta) / (velocidad actual).

## Etapas del análisis

| Etapa | Métodos | Referencia para validar |
|---|---|---|
| 0. Preparación | Regresión (tasa de fondo), umbral sobre v | Tramos de 30 s sin descarga |
| 1. Velocímetro | 3 vs. 5 puntos, con h = 0,1 y 1 s | Velocidad media por regresión |
| 2. Volumen desde la velocidad | Trapecios vs. Simpson con muestras cada H = 1 s | `B_d(t_fin)` exacto; después, contra `S` → sobrecarga p |
| 3. Efecto del muestreo | Misma integración con H = 1, 5, 10, 30 s (log-log) | `B_d(t_fin)` exacto |
| 4. Parámetros del modelo | Regresión lineal y por el origen | R² del ajuste |
| 5. Muestras faltantes | Borrar bloques de k lecturas reales de B y reconstruirlas: Newton y Lagrange con pocos vecinos (grado 3), spline con todas las restantes | Valores borrados |
| 6. Predicción del fin | Bisección, punto fijo, Newton, secante, al 25/50/75 % | Duración real; estimación ingenua |
| Opcional: EDO | Euler, Heun, RK4 sobre `dv/dt = (v∞ − v)/τ` | Solución exacta |

Notas:
- **Paso real:** los métodos asumen h uniforme. Se usa h = paso medio y se informa su desvío (≈ 10⁻⁴ s en las pruebas). Si `graficar.py` muestra lecturas salteadas, se rellenan con interpolación antes de derivar (como en la etapa 5).
- **Etapa 3:** con datos reales no se espera ver la pendiente teórica (h² o h⁴), porque la velocidad real no es una función suave. Lo que se mide es cómo crece el error con H.
- **Etapa 5:** Newton y Lagrange dan el mismo polinomio; la diferencia es de forma y costo, no de precisión. Con muchos nodos, el polinomio oscila (Runge); el spline no.
- Dato para el informe: con `dD/dt = v(t)` (lado derecho que no depende de D), Heun equivale a trapecios y RK4 a Simpson 1/3.

## Corridas previstas

Todas por Wi-Fi (la notebook no tiene puerto de red), en el departamento de Julian (red sin otros usuarios).

| Corrida | Banda | Ubicación | Carga extra |
|---|---|---|---|
| C1 | 5 GHz | Cerca del router | Ninguna (referencia) |
| C2 | 5 GHz | Cerca del router | Netflix en el TV |
| C3 | 2.4 GHz | Cerca del router | Ninguna |
| C4 | 2.4 GHz | Lejos del router | Ninguna |

Comparaciones (cambia una sola variable): C1 vs C2 → otro dispositivo usando la red; C1 vs C3 → banda; C3 vs C4 → distancia.

Procedimiento (dos terminales):
1. Terminal 1: `python captura/registrar.py --interfaz wlp0s20f3 --salida datos/cN.csv` (paso de 0,1 s por defecto).
2. Esperar **30 s** sin usar la red (tráfico de fondo).
3. Terminal 2: descargar el archivo con `wget` (siempre el mismo, una ISO de Linux de al menos unos cientos de MB).
4. Cuando termina `wget`, esperar **30 s** más y cortar la Terminal 1 con Ctrl+C.
5. Revisar con `python captura/graficar.py datos/cN.csv`: tiene que haber 0 lecturas salteadas.
6. Anotar una fila en `datos/corridas.csv`, con el tamaño exacto en bytes (`stat -c %s archivo`).

- Mismo archivo en las cuatro corridas y en horarios parecidos.
- C2: Netflix arrancado antes del registro, calidad fija (alta), TV conectado por Wi-Fi.
- No editar los CSV después de la captura.

## Estado

- [x] Estructura base: requirements.txt, .gitignore, README.md, entorno virtual
- [x] Captura: `registrar.py` y `graficar.py`, probados con una corrida corta
- [x] Revisión de la especificación para la defensa (`DEFENSA.md`)
- [ ] Corridas C1 a C4 (las hace Julian)
- [x] `metodos/derivacion.py` e `integracion.py` con tests (núcleo del proyecto)
- [x] `metodos/regresion.py` e `interpolacion.py` con tests
- [x] `metodos/raices.py` con tests
- [ ] Revisar con Julian los valores sintéticos de los tests, después de las corridas
- [ ] `analisis.ipynb` con las etapas 0 a 6
- [ ] Informe y exposición

Para retomar (22/10): se espera que Julian haga las corridas. Mientras tanto se puede adelantar (1) las partes del informe que no dependen de datos (planteo, modelo, métodos, resultados de los tests) o (2) el código de la etapa 0 con tests sintéticos (inicio y fin de la descarga, tasa de fondo, submuestreo). `datos/c3.csv` (sin seguimiento en git) es una prueba del 22/10 sin descarga y con columnas agregadas a mano: no sirve, se reemplaza con la C3 real.
