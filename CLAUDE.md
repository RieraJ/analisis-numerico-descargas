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
  edo.py               Euler, Heun, RK4 con paso fijo (devuelven t e y)
analisis/
  preparacion.py       etapa 0: carga, paso medio, submuestreo, tasa de fondo, v∞, inicio y fin, sobrecarga;
                       preparar_corrida(t, B, S) hace todo y devuelve un diccionario;
                       tramo_regimen(B_d): máscara del 20–80 % de los bytes
  velocimetro.py       etapa 1: velocidades(t, B_d, k) con 3 y 5 puntos (una lectura de cada k),
                       error_en_regimen(v, v∞, B_d) → media y desvío de v − v∞
  volumen.py           etapa 2: indices_muestreo (una de cada k, alineadas con el fin, con desfase),
                       volumen_desde_velocidad → D por trapecios y Simpson + referencia B_d(t_b) − B_d(t_a)
                       errores_por_desfase → error relativo de los k desfases (etapa 3)
  modelo.py            etapa 4: rampa, ajustar_tau (regresión por el origen), origen_libre (control),
                       r2_velocidad_regimen, velocidad_modelo (v̂), descargado_modelo (D̂), f_fin, phi_fin
  faltantes.py         etapa 5: reconstruir_bloque (lineal, Newton y Lagrange grado 3, spline con todas),
                       error_maximo, posiciones_equiespaciadas, hueco_con_caida
  prediccion.py        etapa 6: indice_progreso, estimar_parametros (con datos parciales), resolver_fin
                       (4 métodos), fines_ingenuos, predecir_fin, velocidad_por_tramos
  edo_modelo.py        opcional: ecuacion_velocidad, error_velocidad (contra v̂), volumen_con_edo (dD/dt = v medida)
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
   - Origen de t: `t_ini`, aunque ahí v ≈ 0,15·v∞ y no 0. Se controla con una recta con ordenada (`origen_libre`), que deja que los datos ubiquen el origen: en C1 a C3 cae a menos de 0,04 s de `t_ini`. Se descartó tomar como origen el último instante con v ≈ 0 (unos 0,8 s antes): necesita un umbral arbitrario y empeora el ajuste (τ de 1,5 a 2,5 s, R² < 0,6), y para la predicción del fin la diferencia es de 1,2 s como máximo.
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

Etapa 1 (`analisis/velocimetro.py`, sección 1 del notebook). Desvío de v − v∞ en régimen, en MB/s:

| | 3 p., h = 0,1 | 5 p., h = 0,1 | 3 p., h = 1 | 5 p., h = 1 | mediana de v | v < 0,5·v∞ |
|---|---|---|---|---|---|---|
| C1 | 1,94 | 2,08 | 0,93 | 1,09 | 12,00 | 2,7 % |
| C2 | 2,02 | 2,21 | 0,93 | 1,08 | 11,81 | 3,2 % |
| C3 | 2,18 | 2,45 | 0,95 | 1,12 | 10,06 | 4,5 % |
| C4 | 1,48 | 1,59 | 0,92 | 0,99 | 4,90 | 6,5 % |

- Media de v − v∞ menor al 1 % de v∞: sin sesgo.
- El modelo de ruido blanco en B (punto 3) **no** describe los datos: de h = 0,1 a 1 s el desvío baja 1,6–2,3 veces (no 10) y 5 p./3 p. da 1,07–1,18 (no 1,34). El test `test_amplificacion_del_ruido` muestra que con ruido blanco sí da 0,71/h y 0,95/h.
- Lo que hay son caídas reales de velocidad de unas décimas de segundo, a veces hasta 0 (gráfico de C1). Por eso la mediana de v supera a v∞ (que es la media, con caídas incluidas). Más caídas con peor enlace: v < 0,9·v∞ en 8 % (C1), 12 % (C2), 20 % (C3), 31 % (C4).
- 3 puntos con h = 1 s es la combinación de menor desvío en las cuatro.

Etapa 2 (`analisis/volumen.py`, sección 2 del notebook). v de 3 puntos (h = 0,1 s) muestreada cada H = 1 s; las muestras se alinean con t_fin y van hacia atrás hasta pasar t_ini (paso uniforme, cubre toda la descarga). Se prueban los 10 desfases posibles (0 a 0,9 s). Error relativo de D contra B_d(t_b) − B_d(t_a), en %:

| | trapecios: mín / máx | promedio | Simpson: mín / máx | promedio |
|---|---|---|---|---|
| C1 | −1,22 / +0,78 | −0,02 | −0,88 / +1,24 | +0,26 |
| C2 | −1,47 / +1,64 | −0,01 | −1,39 / +1,75 | +0,08 |
| C3 | −0,73 / +0,64 | −0,01 | −0,26 / +1,08 | +0,51 |
| C4 | −1,77 / +1,76 | −0,00 | −1,20 / +1,67 | +0,19 |

- Control (H = h): error < 0,012 % (telescopado).
- Con H = 1 s, el signo y el tamaño del error dependen del desfase: las muestras caen o no en las caídas cortas de la etapa 1.
- Simpson no mejora a trapecios (v no es suave). El promedio de trapecios sobre los 10 desfases es ≈ 0 porque juntos usan todas las lecturas; el de Simpson queda positivo (0,1–0,5 %); con H = 1 s es mayormente variación de las caídas (ver etapa 3 para el sesgo de los bordes).
- p con D en lugar del contador: de 3,2 a 6,9 % según el desfase, contra 5,0 % del contador. El error de integración es del orden de la sobrecarga: para medir p hace falta el contador.

Etapa 3 (`errores_por_desfase`, sección 3 del notebook). Misma integración con H = 1, 5, 10, 30 s y los k desfases de cada H. ECM sobre los desfases, en %:

| | trap. H=1 | 5 | 10 | 30 | Simpson H=1 | 5 | 10 | 30 | pendiente trap. / Simpson |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0,51 | 1,50 | 2,88 | 7,03 | 0,58 | 1,92 | 3,42 | 8,32 | 0,77 / 0,78 |
| C2 | 0,98 | 2,21 | 3,64 | 7,33 | 1,06 | 2,32 | 3,63 | 7,62 | 0,59 / 0,58 |
| C3 | 0,43 | 2,75 | 4,33 | 7,88 | 0,67 | 3,03 | 4,53 | 8,68 | 0,87 / 0,76 |
| C4 | 1,19 | 2,42 | 3,74 | 6,06 | 0,97 | 2,72 | 3,87 | 6,78 | 0,48 / 0,57 |

- Pendientes log-log 0,5–0,9, lejos de 2 y 4. Peor desfase con H = 30 s: hasta 35 %.
- Control solo en régimen (sin saltos de arranque y final): pendientes 0,5–0,7 y ECM parecido. El error lo dominan las caídas de v (muestreo), no los bordes. Con errores independientes sería √H.
- Sesgo de Simpson con la descarga completa: si v es una caja (0 afuera, v∞ adentro), Simpson da H·v∞/3 de más con n par y H·v∞/6 con n impar (`test_simpson_sobreestima_una_caja`); en relativo, ≈ H/(3·duración). Con H = 10 y 30 s coincide en C1 a C3 (C1, H = 30: 3,58 % contra 3,75 %). En régimen desaparece (±0,5 % sin signo fijo). Trapecios no tiene sesgo.
- Conclusión práctica: trapecios y muestrear lo más seguido posible.

Etapa 4 (`analisis/modelo.py`, sección 4 del notebook):

| | v∞ | R² v∞ | rampa | τ | R² τ | origen libre: t₀ − t_ini / τ |
|---|---|---|---|---|---|---|
| C1 | 11,58 MB/s | 1,0000 | 5 lect. (0,5 s) | 0,31 s | 0,84 | +0,011 s / 0,30 s |
| C2 | 11,27 MB/s | 0,9999 | 6 lect. (0,6 s) | 0,33 s | 0,78 | +0,034 s / 0,30 s |
| C3 | 9,43 MB/s | 1,0000 | 5 lect. (0,5 s) | 0,39 s | 0,85 | −0,012 s / 0,41 s |
| C4 | 4,67 MB/s | 0,9971 | 22 lect. (2,2 s) | 1,13 s | 0,30 | −0,536 s / 1,56 s |

- τ mal determinado (pocos puntos); en C4 la rampa tiene caídas y el ajuste no sirve.
- La forma del arranque no es la del modelo: v se multiplica por 1,4–1,5 por lectura (arranque lento de TCP, crecimiento acelerado) y se corta en v∞. Antes de t_ini hay ~0,8 s con v < 0,1·v∞.
- En la práctica el modelo es velocidad constante con retraso τ: D̂ ≈ v∞(t − τ); τ mueve el fin previsto menos de 0,2 % de la duración.
- En la corrida sintética (test), t_ini queda un paso después del arranque y la regresión por el origen da τ = 0,24 en vez de 0,30; el origen libre lo recupera exacto. Por eso se informa el control.

Etapa 5 (`analisis/faltantes.py`, sección 5 del notebook). Huecos de k = 1, 5, 10, 30 lecturas en B_d: 100 posiciones parejas en régimen por corrida (separadas en con y sin caída, v < 0,5·v∞) y uno centrado en t_ini. Además de los métodos de la consigna, la recta (Newton grado 1) como referencia. Error = máx |B̂ − B| / bytes del hueco, mediana en %:

| | lineal k=1 / 30 | grado 3 k=1 / 30 | spline k=1 / 30 |
|---|---|---|---|
| régimen sin caída | 4,4 / 1,5 | 5,2 / 3,4 | 6,0 / 4,4 |
| régimen con caída | 49,9 / 7,3 | 22,4 / 9,0 | 18,4 / 10,4 |
| arranque (4 casos) | 12,3 / 52,1 | 5,8 / 23,2 | 8,1 / 23,0 |

- Newton = Lagrange: diferencia máxima 7·10⁻⁶ B (redondeo).
- Régimen sin caída: gana la recta (70–84 % de los huecos); grado 3 y spline siguen las irregularidades de los vecinos. En kB el error crece con k (recta: 77 → 459 kB); en relativo baja porque el hueco tiene más bytes.
- Arranque (B_d curvada): grado 3 y spline mucho mejores que la recta.
- Caída dentro del hueco: ningún método la recupera.
- Sin Runge con grado 3 ni con el spline.

Etapa 6 (`analisis/prediccion.py`, sección 6 del notebook). Con datos hasta el progreso q: v∞ de 20–80 % de lo descargado hasta ahora, t_ini y τ con ese v∞, S' = S·(1 + p_C1). Fin previsto = t_ini + t*. Ingenuas: (S' − B_d)/v con v instantánea (3 puntos, regresiva) o media del último segundo. |Error| en s sobre 18 puntos (q = 10 a 95 %), mediana / máximo:

| | modelo | ingenua instantánea | ingenua último segundo |
|---|---|---|---|
| C1 | 0,4 / 8,8 | 13,6 / 331 | 4,0 / 8,8 |
| C2 | 5,4 / 10,1 | 17,3 / 82 | 5,1 / 12,3 |
| C3 | 0,7 / 14,0 | 49,9 / 383 | 11,8 / 142 |
| C4 | 58,9 / 110 | 40,8 / 270 | 32,7 / 918 |

- Raíces: los 4 métodos coinciden (< 10⁻¹⁰ s). e^(−t/τ) ≈ e^(−800) = 0 en punto flotante: f es una recta y S'/v∞ + τ es exacto. Punto fijo, Newton y secante: 2 iteraciones (la 2.ª confirma); bisección 43–44. El orden de convergencia no se ve con datos reales (sí en `tests/test_raices.py`).
- C4: la velocidad subió 28 % durante la descarga (4,24 → 5,42 MB/s por quintil de bytes); el modelo (v∞ constante) predice tarde. C1 y C3 estables; C2 sube un poco.
- Instantánea inservible por las caídas. Último segundo sigue los cambios pero salta cuando cae en una caída.

EDO (`metodos/edo.py`, `analisis/edo_modelo.py`, última sección del notebook). `dv/dt = (v∞ − v)/τ`, v(0) = 0, con v∞ y τ de cada corrida, contra v̂ en [0, 5 s]:

- Orden observado (h = 0,01–0,05 s): Euler 1,01–1,04; Heun 2,02–2,06; RK4 4,02–4,07.
- Estabilidad: Euler y Heun si h < 2τ, RK4 si h < 2,785τ. Con h = 1 s, en C1 y C2 los tres divergen (error 8 a 230 veces v∞); en C3 Euler y Heun divergen y RK4 da 0,63; en C4 (τ = 1,13 s) los tres son estables. Para simular el arranque hace falta h < τ.
- Heun = trapecios y RK4 = Simpson sobre las v medidas cada 1 s: diferencias de 10⁻⁶ B (redondeo).
- En los tests, el orden se mide con el error máximo en la grilla: con el error en un solo punto, RK4 tarda en mostrar orden 4 (cancelación parcial de términos).

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
- [x] Etapa 1: `analisis/velocimetro.py` con tests y sección en el notebook
- [x] Etapa 2: `analisis/volumen.py` con tests y sección en el notebook
- [x] Etapa 3: `errores_por_desfase` con tests y sección en el notebook
- [x] Etapa 4: `analisis/modelo.py` con tests y sección en el notebook
- [x] Etapa 5: `analisis/faltantes.py` con tests y sección en el notebook
- [x] Etapa 6: `analisis/prediccion.py` con tests y sección en el notebook
- [x] Opcional EDO: `metodos/edo.py` y `analisis/edo_modelo.py` con tests y sección en el notebook
- [ ] Informe y exposición

## Para retomar (23/10)

Hecho: las etapas 0 a 6 y la parte opcional de EDO, con tests y aplicadas a C1 a C4 en el notebook. 133 tests pasan.

Queda:
- **Informe y exposición.** Los resultados y conclusiones de cada etapa están arriba y en las celdas "Qué muestra…" del notebook.
- **Revisar `DEFENSA.md`** recién al final (ver memoria).

Cómo trabajar con el notebook:
- `analisis.ipynb` tiene `resultados[c]` (c = "C1"…"C4") con `t`, `B`, `S`, todo lo que devuelve `preparar_corrida` (`h`, `r_f`, `B_d`, `v`, `v_inf`, `i_ini`, `i_fin`, `t_ini`, `t_fin`, `p`) y `tau` (etapa 4). También `colores` (uno por corrida) y `p_C1`.
- Para regenerar las salidas: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace analisis.ipynb` (tarda ~30 s).
- Para editar celdas desde un script, usar `nbformat` (mantiene los ids de las celdas).
