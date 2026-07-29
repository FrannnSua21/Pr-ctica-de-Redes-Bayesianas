# Red Bayesiana - Analisis desde cero y descubrimiento causal

Interfaz grafica en Python (tkinter) para construir una Red Bayesiana desde cero con un dataset de 8 registros (x1, x2, x3, Y) y compararla con los algoritmos de descubrimiento causal PC y Hill Climbing.

## Funcionalidades

- Visualizacion del dataset
- Calculo de probabilidades marginales
- Calculo de probabilidades conjuntas y aplicacion del Teorema de Bayes por variable
- Grafo (DAG) manual, y grafos obtenidos con PC y Hill Climbing usando pgmpy
- Comparacion de estructuras con las metricas BIC, AIC y BDeu

## Requisitos

- Python 3.10 o superior
- pandas
- networkx
- matplotlib
- pgmpy
- tkinter (en Linux: `sudo apt install python3-tk`, en Windows y Mac ya viene incluido)

## Instalacion

```bash
pip install pandas networkx matplotlib pgmpy
```

## Uso

```bash
python red_bayesiana_gui.py
```

Se abre una ventana con 5 pestanas: Dataset, Probabilidades, Teorema de Bayes, Grafos y Algoritmos, y Metricas BIC AIC BDeu.

## Estructura

```
red_bayesiana_gui.py
```

Contiene las funciones de calculo (marginales, conjuntas, Bayes, PC, Hill Climbing, metricas) separadas de la interfaz grafica.
