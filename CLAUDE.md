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
datos/                 c1.csv … c4.csv (una captura por corrida) y corridas.csv (condiciones, tamaño exacto, señal, velocidad de wget)
metodos/
  derivacion.py        fórmulas centradas de 3 y 5 puntos (descentradas del mismo orden en los extremos)
  integracion.py       trapecios y Simpson 1/3 (paso uniforme; si los intervalos son impares, el último por trapecios)
  interpolacion.py     diferencias divididas de Newton, Lagrange, spline cúbico natural
  regresion.py         mínimos cuadrados lineal (ecuaciones normales resueltas a mano)
  raices.py            bisección, punto fijo, Newton-Raphson, secante (devuelven raíz + historial de iteraciones)
analisis/
  preparacion.py       etapa 0: carga, paso medio, submuestreo, tasa de fondo, v∞, inicio y fin, sobrecarga;
                       preparar_corrida(t, B, S) hace todo y devuelve un diccionario
tests/                 pytest, un archivo por módulo de metodos/ y de analisis/
analisis.ipynb         una sección por etapa: solo llama a analisis/ y muestra tablas y gráficos
DEFENSA.md             preguntas probables en la exposición (solo local, en .gitignore)
requirements.txt, .gitignore (.venv, __pycache__, .ipynb_checkpoints, *.iso, datos/prueba*.csv, DEFENSA.md, .~lock.*), README.md
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

## Corridas

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

Hechas el 23/10 con `ubuntu-26.04.1-live-server-amd64.iso` (S = 2 927 861 760 B); detalle en `datos/corridas.csv`. Cerca = 4 m del router; lejos = balcón a 6 m con la puerta cerrada. Las cuatro con 0 lecturas salteadas, paso medio 0,100000 s (desvío ≈ 10⁻⁴ s) y más de 30 s quietos al principio y al final. A C4 le falta la señal (no se midió) y a C1, la línea final de `wget`.

Para el informe:
- El contador solo ve el tráfico de la notebook. En C2, Netflix no aparece como bytes: influye solo porque compite por el Wi-Fi.
- `wget` muestra las velocidades en unidades de 1024 aunque diga "MB/s", y es la velocidad media S/(t_fin − t_ini), no v∞. Confirmado en la etapa 0: en C2 a C4, S/duración coincide con lo de `wget` tomando 1 MB = 2²⁰ B.

## Resultados (datos reales)

Etapa 0 (`preparar_corrida`, sección 0 de `analisis.ipynb`):

| | v∞ | t_ini → t_fin | Duración | p | r_f |
|---|---|---|---|---|---|
| C1 | 11,58 MB/s | 32,6 → 299,0 s | 266,4 s | 5,07 % | 928 B/s |
| C2 | 11,27 MB/s | 35,6 → 309,4 s | 273,8 s | 5,05 % | 620 B/s |
| C3 | 9,43 MB/s | 34,9 → 361,5 s | 326,6 s | 5,01 % | 886 B/s |
| C4 | 4,67 MB/s | 33,7 → 682,2 s | 648,5 s | 5,02 % | 428 B/s |

- p ≈ 5 % en las cuatro: depende del protocolo, no de la red. Es el borde superior de lo esperado (3–5 %).
- Netflix (C2) bajó v∞ menos de 3 % respecto de C1. La banda (C1 → C3) y la distancia (C3 → C4) pesan mucho más.
- Decisiones de implementación, para explicar en la defensa:
  - r_f: cada tramo de 30 s se ajusta por separado y se promedian las pendientes (una sola regresión con los dos tramos no sirve, por el salto de la descarga).
  - t_ini y t_fin salen de la derivada de 3 puntos, así que tienen error de hasta un paso h. En el test sintético el inicio da 30,1 s contra 30,0 (tolerancia 1,01·h por redondeo).
  - B_d(t_fin) se toma en el último instante con v > 0,1·v∞; los bytes que faltan del último paso cambian p en menos de v∞·h/S ≈ 0,04 puntos.
- Para la etapa 4: en las cuatro corridas la velocidad llega al régimen en menos de 0,5 s, así que la rampa tiene pocos puntos con h = 0,1 s. Es probable el caso "τ mal determinado" (ver modelo, punto 5); se confirma ahí.

## Estado

- [x] Estructura base: requirements.txt, .gitignore, README.md, entorno virtual
- [x] Captura: `registrar.py` y `graficar.py`, probados con una corrida corta
- [x] Revisión de la especificación para la defensa (`DEFENSA.md`)
- [x] Corridas C1 a C4 (23/10)
- [x] `metodos/derivacion.py` e `integracion.py` con tests (núcleo del proyecto)
- [x] `metodos/regresion.py` e `interpolacion.py` con tests
- [x] `metodos/raices.py` con tests
- [x] Revisar con Julian los valores sintéticos de los tests (23/10: S y τ de raíces llevados a las magnitudes reales)
- [x] Etapa 0: `analisis/preparacion.py` con tests
- [ ] `analisis.ipynb` con las etapas 0 a 6
- [ ] Informe y exposición

## Para retomar (23/10)

Hecho en esta sesión: corridas limpias y versionadas, tests de raíces con magnitudes reales, etapa 0 con tests y aplicada a C1 a C4 en el notebook. 75 tests pasan. Todo subido a GitHub.

Sigue la **etapa 1 (velocímetro)**, en una sección nueva de `analisis.ipynb`:
- Para cada corrida, `v` con `derivada_3_puntos` y `derivada_5_puntos` sobre `B_d`, con h = 0,1 s y con h = 1 s (`submuestrear(t, B_d, 10)`).
- Referencia: v∞ de la etapa 0. Medir en el tramo de régimen (20–80 % de los bytes) la media y el desvío de v respecto de v∞, para las cuatro combinaciones (fórmula × h).
- Lo esperado según el modelo (punto 3): con h = 0,1 s el desvío es grande por el ruido (más en 5 puntos que en 3) y con h = 1 s baja. Se informa lo que dé, sin forzarlo.
- Gráfico: v(t) de una corrida (C1) con las cuatro combinaciones, mismo eje, y v∞ como línea horizontal.
- Si hace falta una función reutilizable (por ejemplo, el tramo de régimen como máscara), va a `analisis/` con test; el notebook solo llama.

Cómo trabajar con el notebook:
- `analisis.ipynb` ya tiene `resultados[c]` (c = "C1"…"C4") con `t`, `B`, `S` y todo lo que devuelve `preparar_corrida` (`h`, `r_f`, `B_d`, `v`, `v_inf`, `i_ini`, `i_fin`, `t_ini`, `t_fin`, `p`).
- Para regenerar las salidas: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace analisis.ipynb`.

Después: etapas 2 a 6 en orden, una por paso y commit. Lo que se repita entre etapas (D̂, f y φ del modelo; borrar bloques de lecturas) va a `analisis/modelo.py`, con tests.
