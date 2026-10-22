# Análisis numérico de descargas

Proyecto de Análisis Numérico (Ingeniería en Sistemas de Información, UCC): estimar los datos descargados y el tiempo restante de una descarga a partir de mediciones reales del contador de bytes de la red, usando métodos numéricos implementados desde cero (derivación, integración, interpolación, regresión y búsqueda de raíces).

## Entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Ver interfaces de red y sus bytes recibidos
python captura/registrar.py --listar

# Registrar una corrida cada 0,1 s (Ctrl+C para terminar)
python captura/registrar.py --interfaz wlp0s20f3 --salida datos/c1.csv

# Revisar una corrida
python captura/graficar.py datos/c1.csv

# Correr los tests
pytest
```

## Estructura

```
captura/        registro del contador de bytes y revisión de corridas
datos/          un CSV por corrida
metodos/        métodos numéricos propios (NumPy, sin SciPy)
tests/          pytest; SciPy solo se usa acá para verificar
analisis.ipynb  experimentos, tablas y gráficos del informe
```
