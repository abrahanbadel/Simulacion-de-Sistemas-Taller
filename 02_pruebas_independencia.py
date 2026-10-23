# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - Paso 2: PRUEBAS DE INDEPENDENCIA
  a) Gráfico de dispersión  (x_i  vs  x_i+1)
  b) Análisis de correlación (autocorrelograma)
  c) Análisis de pruebas    (rachas sobre/bajo la mediana, rachas
                              ascendentes/descendentes y Ljung-Box)
Si se instala por primera vez:  pip install statsmodels
"""

#%% 1. Librerías y datos
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Trabajar SIEMPRE en la carpeta donde está guardado este script ---
# (evita el error FileNotFoundError cuando Spyder tiene otra carpeta de trabajo)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
# Los .csv pueden estar en la subcarpeta datos/ o junto a los scripts
CARPETA_DATOS = "datos" if os.path.isdir("datos") else "."
print("Carpeta de trabajo:", os.getcwd())

os.makedirs("figuras", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"
ALFA = 0.05

VARIABLE = "lavado_B"                  # <- archivo de la carpeta datos/ (sin .csv)
x = pd.read_csv(f"{CARPETA_DATOS}/{VARIABLE}.csv")["tiempo_min"].to_numpy()
n = len(x)

#%% 2. (a) GRÁFICO DE DISPERSIÓN x_i vs x_i+1
# Para comparar, creamos una serie DEPENDIENTE a propósito (cada dato "arrastra" al anterior)
rng = np.random.default_rng(1)
dep = np.zeros(n); dep[0] = x.mean()
for i in range(1, n):
    dep[i] = 0.85 * dep[i - 1] + 0.15 * x.mean() + rng.normal(0, 1.2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, serie, titulo, color in [(axs[0], x, f"Nuestros datos ({VARIABLE})", AZUL),
                                 (axs[1], dep, "Ejemplo de datos DEPENDIENTES", NARANJA)]:
    r = np.corrcoef(serie[:-1], serie[1:])[0, 1]
    ax.scatter(serie[:-1], serie[1:], s=22, color=color, alpha=0.8)
    ax.set_title(f"{titulo}\nr = {r:.3f}", loc="left", fontsize=11)
    ax.set_xlabel("$x_i$"); ax.set_ylabel("$x_{i+1}$")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/02_dispersion.png", dpi=200)
plt.show()

r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
print(f"Correlación entre x_i y x_i+1: r = {r1:.3f}")
# Nube SIN forma (sin tendencia) -> los datos parecen independientes

#%% 3. (b) ANÁLISIS DE CORRELACIÓN: autocorrelograma
# r_k = correlación entre un dato y el que está k posiciones después.
# Bandas de confianza aprox. 95 %:  ± 1.96 / raíz(n)
banda = 1.96 / np.sqrt(n)
print(f"Bandas: ± {banda:.3f}")

fig, ax = plt.subplots(figsize=(9, 3.8))
plot_acf(x, lags=20, alpha=ALFA, ax=ax, title="", color=AZUL, vlines_kwargs={"colors": AZUL})
ax.axhline(banda, color=NARANJA, ls="--", lw=1); ax.axhline(-banda, color=NARANJA, ls="--", lw=1)
ax.set_title("Autocorrelograma (retardos 1 a 20)", loc="left", fontsize=12)
ax.set_xlabel("Retardo k"); ax.set_ylabel("$r_k$")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/02_autocorrelograma.png", dpi=200)
plt.show()

xc = x - x.mean()
rk = np.array([np.sum(xc[:-k] * xc[k:]) / np.sum(xc ** 2) for k in range(1, 21)])
fuera = np.where(np.abs(rk) > banda)[0] + 1
print("Retardos fuera de las bandas:", fuera if len(fuera) else "ninguno")
# OJO: con bandas del 95 %, por puro azar se espera que ~1 de cada 20 retardos
# quede afuera. Preocupa si salen varios, o si salen los primeros retardos (k = 1, 2).
print(f"Se esperan por azar ~{0.05 * 20:.0f} de 20 fuera de las bandas")

#%% 4. (c-1) PRUEBA DE RACHAS SOBRE Y BAJO LA MEDIANA
# H0: los datos son independientes (aleatorios)   H1: no lo son
mediana = np.median(x)
signos = np.where(x > mediana, "+", "-")[x != mediana]   # se descartan datos iguales a la mediana
n1, n2 = np.sum(signos == "+"), np.sum(signos == "-")
rachas = 1 + np.sum(signos[1:] != signos[:-1])           # cada cambio de signo inicia una racha
media_R = 2 * n1 * n2 / (n1 + n2) + 1
var_R = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
z_med = (rachas - media_R) / np.sqrt(var_R)
p_med = 2 * (1 - stats.norm.cdf(abs(z_med)))
print(f"Rachas = {rachas} | E[R] = {media_R:.2f} | Z = {z_med:.3f} | valor p = {p_med:.4f}")

#%% 5. (c-2) PRUEBA DE RACHAS ASCENDENTES Y DESCENDENTES
d = np.sign(np.diff(x)); d = d[d != 0]                    # +1 sube, -1 baja
rachas_ad = 1 + np.sum(d[1:] != d[:-1])
media_ad = (2 * n - 1) / 3
var_ad = (16 * n - 29) / 90
z_ad = (rachas_ad - media_ad) / np.sqrt(var_ad)
p_ad = 2 * (1 - stats.norm.cdf(abs(z_ad)))
print(f"Rachas = {rachas_ad} | E[R] = {media_ad:.2f} | Z = {z_ad:.3f} | valor p = {p_ad:.4f}")

#%% 6. (c-3) PRUEBA DE LJUNG-BOX (todas las autocorrelaciones hasta el retardo 10)
lb = acorr_ljungbox(x, lags=[10], return_df=True)
q_lb, p_lb = lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]
print(f"Q = {q_lb:.3f} | valor p = {p_lb:.4f}")

#%% 7. TABLA RESUMEN Y DECISIÓN
tabla = pd.DataFrame({
    "Prueba": ["Rachas sobre/bajo la mediana", "Rachas ascendentes/descendentes", "Ljung-Box (10 retardos)"],
    "Estadístico": [f"Z = {z_med:.3f}", f"Z = {z_ad:.3f}", f"Q = {q_lb:.3f}"],
    "Valor p": [p_med, p_ad, p_lb],
})
tabla["Decisión (α = 0.05)"] = np.where(tabla["Valor p"] >= ALFA, "No se rechaza H0", "Se rechaza H0")
print(tabla.round(4).to_string(index=False))

# La decisión formal se toma con las pruebas; los gráficos la respaldan
if (tabla["Valor p"] >= ALFA).all():
    print("\nCONCLUSIÓN: los datos SON independientes -> pasar a la selección de la distribución.")
else:
    print("\nCONCLUSIÓN: hay evidencia de dependencia -> revisar la recolección de datos.")
