import warnings
warnings.filterwarnings("ignore")

import itertools
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx as nx

from pgmpy.base import DAG
from pgmpy.estimators import PC, HillClimbSearch, BIC, AIC, BDeu


data = pd.DataFrame({
    "x1": [0, 0, 1, 1, 1, 0, 0, 1],
    "x2": [0, 0, 0, 1, 1, 1, 0, 1],
    "x3": [1, 1, 1, 0, 0, 1, 1, 0],
    "Y":  [1, 1, 1, 0, 0, 0, 1, 0]
})

N = len(data)
variables = ["x1", "x2", "x3"]


def calcular_marginales(df):
    lineas = []
    for col in list(df.columns):
        conteos = df[col].value_counts().sort_index()
        for valor, conteo in conteos.items():
            prob = conteo / N
            lineas.append("P({}={}) = {}/{} = {:.4f}".format(col, valor, conteo, N, prob))
        lineas.append("")
    return "\n".join(lineas)


def calcular_conjuntas(df, var):
    lineas = []
    combinaciones = list(itertools.product([0, 1], [0, 1]))
    for vx, vy in combinaciones:
        conteo = len(df[(df[var] == vx) & (df["Y"] == vy)])
        prob = conteo / N
        lineas.append("P({}={}, Y={}) = {}/{} = {:.4f}".format(var, vx, vy, conteo, N, prob))
    return "\n".join(lineas)


def calcular_bayes(df, var):
    lineas = []
    py1 = len(df[df["Y"] == 1]) / N
    py0 = len(df[df["Y"] == 0]) / N
    for vy, pY in [(1, py1), (0, py0)]:
        subset = df[df["Y"] == vy]
        for vx in [0, 1]:
            pxdady = len(subset[subset[var] == vx]) / len(subset) if len(subset) > 0 else 0
            px = len(df[df[var] == vx]) / N
            if px == 0:
                continue
            pydadx = (pxdady * pY) / px
            lineas.append(
                "P({}={}|Y={}) = {:.4f}   ->   P(Y={}|{}={}) = {:.4f}".format(
                    var, vx, vy, pxdady, vy, var, vx, pydadx
                )
            )
    return "\n".join(lineas)


def construir_dag_manual():
    modelo = DAG()
    modelo.add_nodes_from(["x1", "x2", "x3", "Y"])
    modelo.add_edges_from([("x1", "x3"), ("x2", "x3"), ("x3", "Y")])
    return modelo


def ejecutar_pc(df):
    pc = PC(df)
    modelo = pc.estimate(variant="stable", ci_test="chi_square", significance_level=0.05, show_progress=False)
    return modelo


def ejecutar_hill_climbing(df):
    hc = HillClimbSearch(df)
    scorer = BIC(df)
    modelo = hc.estimate(scoring_method=scorer, show_progress=False)
    return modelo


def calcular_metricas(df, modelos):
    bic = BIC(df)
    aic = AIC(df)
    bdeu = BDeu(df, equivalent_sample_size=1)
    filas = []
    for nombre, modelo in modelos.items():
        filas.append({
            "modelo": nombre,
            "aristas": str(list(modelo.edges())),
            "BIC": round(bic.score(modelo), 4),
            "AIC": round(aic.score(modelo), 4),
            "BDeu": round(bdeu.score(modelo), 4)
        })
    return pd.DataFrame(filas)


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Red Bayesiana - Analisis desde cero y descubrimiento causal")
        self.geometry("950x650")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_dataset = ttk.Frame(notebook)
        self.tab_prob = ttk.Frame(notebook)
        self.tab_bayes = ttk.Frame(notebook)
        self.tab_grafos = ttk.Frame(notebook)
        self.tab_metricas = ttk.Frame(notebook)

        notebook.add(self.tab_dataset, text="Dataset")
        notebook.add(self.tab_prob, text="Probabilidades")
        notebook.add(self.tab_bayes, text="Teorema de Bayes")
        notebook.add(self.tab_grafos, text="Grafos y Algoritmos")
        notebook.add(self.tab_metricas, text="Metricas BIC AIC BDeu")

        self.construir_tab_dataset()
        self.construir_tab_probabilidades()
        self.construir_tab_bayes()
        self.construir_tab_grafos()
        self.construir_tab_metricas()

    def construir_tab_dataset(self):
        frame = self.tab_dataset
        cols = list(data.columns)
        tabla = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        for c in cols:
            tabla.heading(c, text=c)
            tabla.column(c, width=80, anchor="center")
        for _, fila in data.iterrows():
            tabla.insert("", "end", values=list(fila))
        tabla.pack(padx=20, pady=20, fill="x")

        info = tk.Label(frame, text="N = {} registros, variables binarias x1 x2 x3 Y".format(N), font=("Arial", 11))
        info.pack(pady=10)

    def construir_tab_probabilidades(self):
        frame = self.tab_prob
        boton = ttk.Button(frame, text="Calcular probabilidades marginales", command=self.mostrar_marginales)
        boton.pack(pady=10)

        self.texto_prob = tk.Text(frame, height=25, width=100, font=("Consolas", 10))
        self.texto_prob.pack(padx=10, pady=10, fill="both", expand=True)

    def mostrar_marginales(self):
        resultado = calcular_marginales(data)
        self.texto_prob.delete("1.0", tk.END)
        self.texto_prob.insert(tk.END, resultado)

    def construir_tab_bayes(self):
        frame = self.tab_bayes
        top = ttk.Frame(frame)
        top.pack(pady=10)

        tk.Label(top, text="Variable:").pack(side="left", padx=5)
        self.combo_var = ttk.Combobox(top, values=variables, state="readonly", width=10)
        self.combo_var.current(0)
        self.combo_var.pack(side="left", padx=5)

        boton = ttk.Button(top, text="Calcular P(Y|X) con Bayes", command=self.mostrar_bayes)
        boton.pack(side="left", padx=10)

        self.texto_bayes = tk.Text(frame, height=25, width=100, font=("Consolas", 10))
        self.texto_bayes.pack(padx=10, pady=10, fill="both", expand=True)

    def mostrar_bayes(self):
        var = self.combo_var.get()
        conjunta = calcular_conjuntas(data, var)
        bayes = calcular_bayes(data, var)
        resultado = "Probabilidades conjuntas P({}, Y)\n\n{}\n\nAplicacion del Teorema de Bayes\n\n{}".format(
            var, conjunta, bayes
        )
        self.texto_bayes.delete("1.0", tk.END)
        self.texto_bayes.insert(tk.END, resultado)

    def construir_tab_grafos(self):
        frame = self.tab_grafos
        top = ttk.Frame(frame)
        top.pack(pady=10)

        ttk.Button(top, text="DAG Manual", command=lambda: self.mostrar_grafo("manual")).pack(side="left", padx=5)
        ttk.Button(top, text="Algoritmo PC", command=lambda: self.mostrar_grafo("pc")).pack(side="left", padx=5)
        ttk.Button(top, text="Hill Climbing", command=lambda: self.mostrar_grafo("hc")).pack(side="left", padx=5)

        self.frame_canvas = ttk.Frame(frame)
        self.frame_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_actual = None

        self.modelos_cache = {
            "manual": construir_dag_manual(),
            "pc": None,
            "hc": None
        }

    def mostrar_grafo(self, tipo):
        if tipo == "pc" and self.modelos_cache["pc"] is None:
            self.modelos_cache["pc"] = ejecutar_pc(data)
        if tipo == "hc" and self.modelos_cache["hc"] is None:
            self.modelos_cache["hc"] = ejecutar_hill_climbing(data)

        modelo = self.modelos_cache[tipo]
        titulos = {"manual": "DAG Manual (enunciado)", "pc": "Grafo obtenido con PC", "hc": "Grafo obtenido con Hill Climbing"}

        grafo = nx.DiGraph()
        grafo.add_nodes_from(["x1", "x2", "x3", "Y"])
        grafo.add_edges_from(list(modelo.edges()))

        figura = Figure(figsize=(6, 5))
        ax = figura.add_subplot(111)
        posiciones = nx.circular_layout(grafo)
        nx.draw(
            grafo, posiciones, ax=ax, with_labels=True, node_color="#9FE3D3",
            node_size=1800, font_size=12, font_weight="bold", arrowsize=25
        )
        ax.set_title(titulos[tipo])

        if self.canvas_actual is not None:
            self.canvas_actual.get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(figura, master=self.frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_actual = canvas

    def construir_tab_metricas(self):
        frame = self.tab_metricas
        boton = ttk.Button(frame, text="Calcular BIC AIC BDeu para los tres modelos", command=self.mostrar_metricas)
        boton.pack(pady=10)

        cols = ["modelo", "aristas", "BIC", "AIC", "BDeu"]
        self.tabla_metricas = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.tabla_metricas.heading(c, text=c)
            ancho = 300 if c == "aristas" else 100
            self.tabla_metricas.column(c, width=ancho, anchor="center")
        self.tabla_metricas.pack(padx=10, pady=10, fill="x")

        self.texto_analisis = tk.Text(frame, height=12, width=100, font=("Consolas", 10))
        self.texto_analisis.pack(padx=10, pady=10, fill="both", expand=True)

    def mostrar_metricas(self):
        manual = construir_dag_manual()
        pc = ejecutar_pc(data)
        hc = ejecutar_hill_climbing(data)

        modelos = {"Manual": manual, "PC": pc, "Hill Climbing": hc}
        tabla = calcular_metricas(data, modelos)

        for fila in self.tabla_metricas.get_children():
            self.tabla_metricas.delete(fila)
        for _, fila in tabla.iterrows():
            self.tabla_metricas.insert("", "end", values=list(fila))

        mejor = tabla.loc[tabla["BIC"].idxmax()]
        analisis = (
            "El modelo con mejor puntaje BIC es {} con {:.4f}.\n"
            "Con N={} la potencia estadistica es baja, por lo que PC puede dejar aristas sin "
            "direccion definida y los algoritmos automaticos pueden preferir estructuras mas "
            "simples que el DAG manual."
        ).format(mejor["modelo"], mejor["BIC"], N)
        self.texto_analisis.delete("1.0", tk.END)
        self.texto_analisis.insert(tk.END, analisis)


if __name__ == "__main__":
    app = Aplicacion()
    app.mainloop()