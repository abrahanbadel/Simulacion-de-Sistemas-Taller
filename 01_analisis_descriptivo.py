# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - Paso 1: ANÁLISIS DESCRIPTIVO DE LOS DATOS
Caso: Ejemplo 6.1 (lavado e inspección de piezas A, B y C)
Ejecute celda por celda en Spyder (Ctrl + Enter).
Para analizar otra variable cambie VARIABLE (ej.: "llegadas_A", "inspeccion_C").
"""

#%% 1. Librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

# Los .csv pueden estar en la subcarpeta datos/ o junto a los scripts
CARPETA_DATOS = "datos" if os.path.isdir("datos") else "."
print("Carpeta de trabajo:", os.getcwd())

os.makedirs("figuras", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"

#%% 2. Cargar los datos
VARIABLE = "lavado_B"                  # <- nombre del archivo de la carpeta datos/ (sin .csv)
df = pd.read_csv(f"{CARPETA_DATOS}/{VARIABLE}.csv")
x = df["tiempo_min"].to_numpy()        # convertimos la columna en un arreglo de números
n = len(x)
print("Número de datos:", n)
print("Primeros 10 datos:", x[:10])

#%% 3. Medidas de resumen
q1, mediana, q3 = np.percentile(x, [25, 50, 75])
resumen = pd.Series({
    "n": n,
    "Media": x.mean(),
    "Mediana": mediana,
    "Desviación estándar": x.std(ddof=1),      # ddof=1 -> desviación MUESTRAL (n-1)
    "Coef. de variación (CV)": x.std(ddof=1) / x.mean(),
    "Mínimo": x.min(),
    "Q1 (25 %)": q1,
    "Q3 (75 %)": q3,
    "Máximo": x.max(),
    "Rango": x.max() - x.min(),
    "Asimetría": stats.skew(x, bias=False),
    "Curtosis (exceso)": stats.kurtosis(x, bias=False),
})
print(resumen.round(3).to_string())

# Pistas para la Fase de ajuste:
#  CV ~ 1            -> sospechar Exponencial
#  CV < 1 y asimetría > 0 -> Gamma, Erlang, Lognormal, Weibull
#  asimetría ~ 0     -> Normal (o Uniforme si la curtosis es muy negativa)

#%% 4. Diagrama de caja (boxplot) y datos atípicos
ric = q3 - q1                          # rango intercuartílico (RIC o IQR)
lim_inf = q1 - 1.5 * ric
lim_sup = q3 + 1.5 * ric
atipicos = x[(x < lim_inf) | (x > lim_sup)]
print(f"RIC = {ric:.2f}  |  Límites: [{lim_inf:.2f} ; {lim_sup:.2f}]")
print("Datos atípicos:", np.sort(atipicos))

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.boxplot(x, vert=False, widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#D6E6EA", color=AZUL, linewidth=1.5),
           medianprops=dict(color=NARANJA, linewidth=2.5),
           whiskerprops=dict(color=AZUL, linewidth=1.5),
           capprops=dict(color=AZUL, linewidth=1.5),
           flierprops=dict(marker="o", markerfacecolor=NARANJA, markeredgecolor=NARANJA, markersize=7))
for valor, texto in [(q1, "Q1"), (mediana, "Mediana"), (q3, "Q3")]:
    ax.annotate(f"{texto}\n{valor:.2f}", (valor, 1.27), ha="center", fontsize=10, color=AZUL)
ax.axvline(lim_sup, color=GRIS, ls="--", lw=1)
ax.annotate(f"Límite superior\nQ3 + 1.5·RIC = {lim_sup:.2f}", (lim_sup - 0.2, 0.62), ha="right", fontsize=9, color=GRIS)
ax.set_yticks([])
ax.set_xlabel("Tiempo (min)")
ax.set_title(f"Diagrama de caja: {VARIABLE}", loc="left", fontsize=12)
ax.spines[["top", "right", "left"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/01_boxplot.png", dpi=200)
plt.show()

#%% 5. Histograma (número de clases por la regla de Sturges)
k = int(round(1 + 3.322 * np.log10(n)))
print("Número de clases (Sturges):", k)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(x, bins=k, color=AZUL, edgecolor="white", alpha=0.9)
ax.axvline(x.mean(), color=NARANJA, lw=2, label=f"Media = {x.mean():.2f}")
ax.axvline(mediana, color=GRIS, lw=2, ls="--", label=f"Mediana = {mediana:.2f}")
ax.set_xlabel("Tiempo (min)")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Histograma: {VARIABLE} (k = {k} clases)", loc="left", fontsize=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/01_histograma.png", dpi=200)
plt.show()

#%% 6. Gráfico de los datos en el ORDEN en que se tomaron
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(np.arange(1, n + 1), x, marker="o", ms=3.5, color=AZUL, lw=1)
ax.axhline(x.mean(), color=NARANJA, lw=1.5, label="Media")
ax.set_xlabel("Número de observación (orden de medición)")
ax.set_ylabel("Tiempo (min)")
ax.set_title(f"{VARIABLE} en orden de recolección: ¿hay tendencias o saltos?", loc="left", fontsize=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/01_serie.png", dpi=200)
plt.show()
