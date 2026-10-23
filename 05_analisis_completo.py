# -*- coding: utf-8 -*-
"""
SIMULACION DE SISTEMAS - CECAR
Ejemplo 6.1 - ANALISIS COMPLETO (script unico)
================================================
Procesa en batch TODOS los archivos CSV de la carpeta datos/:
  - Pruebas de independencia  (equivalente al script 02)
  - Bondad de ajuste          (equivalente al script 03)
  - Validacion del modelo     (equivalente al script 04)

Genera:
  figuras/<variable>/
      02a_dispersion.png
      02b_autocorrelograma.png
      03a_densidades.png
      03b_acumuladas.png
      03c_pp_qq.png
  figuras/validacion/
      04_validacion.png
  resultados/
      independencia_resumen.csv
      bondad_ajuste_resumen.csv
      validacion_resumen.csv
      ranking_<variable>.csv      (uno por variable)
      ajuste_<variable>.json      (con expresion FlexSim)
"""

# ===========================================================================
# 0. LIBRERIAS
# ===========================================================================
import os
import sys
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # modo sin ventana: guarda las figuras en disco
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Fuerza UTF-8 en la consola de Windows para que los acentos se vean bien
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")   # silencia advertencias menores de scipy

# ===========================================================================
# 1. RUTAS Y CONFIGURACION GLOBAL
# ===========================================================================
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()          # compatibilidad con Jupyter / Spyder interactivo
os.chdir(BASE)

def _detectar_carpeta_datos():
    """Busca la carpeta con los CSV en varias ubicaciones posibles."""
    for ruta in ("datos/datos", "datos", "."):
        if os.path.isfile(os.path.join(BASE, ruta, "llegadas_A.csv")):
            return os.path.join(BASE, ruta)
    raise FileNotFoundError(
        "No se encontro la carpeta de datos. "
        "Asegurese de que este script este en la misma carpeta que la carpeta 'datos/'."
    )

CARPETA_DATOS = _detectar_carpeta_datos()
print("Carpeta de datos detectada:", CARPETA_DATOS)

os.makedirs(os.path.join(BASE, "figuras"),    exist_ok=True)
os.makedirs(os.path.join(BASE, "resultados"), exist_ok=True)

# Colores del tema
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"
COLORES_GRAF = [NARANJA, AZUL, "#5B8E7D", "#9A348E", "#B08D57", GRIS]
ALFA = 0.05   # nivel de significancia

# Variables de tiempo a procesar (nombre del archivo sin .csv)
VARIABLES_TIEMPO = [
    "llegadas_A",   "llegadas_B",   "llegadas_C",
    "lavado_A",     "lavado_B",     "lavado_C",
    "inspeccion_A", "inspeccion_B", "inspeccion_C",
]


def guardar_csv(df, ruta, **kwargs):
    """Guarda un DataFrame en CSV con manejo amigable de bloqueos de archivo (ej. abierto en Excel)."""
    try:
        df.to_csv(ruta, **kwargs)
    except PermissionError:
        nombre = os.path.basename(ruta)
        print(f"\n  [ADVERTENCIA] No se pudo sobrescribir '{nombre}' porque esta abierto en Excel u otro programa.")
        print(f"                Cierre el archivo para que los cambios se guarden en disco.\n")


def guardar_json(obj, ruta, **kwargs):
    """Guarda un diccionario en JSON con manejo de bloqueos."""
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(obj, f, **kwargs)
    except PermissionError:
        nombre = os.path.basename(ruta)
        print(f"\n  [ADVERTENCIA] No se pudo sobrescribir '{nombre}' porque esta bloqueado o abierto.\n")


# ===========================================================================
# SECCION A — PRUEBAS DE INDEPENDENCIA
# ===========================================================================
def pruebas_independencia(variable, x, carpeta_fig):
    """
    Aplica tres pruebas de independencia sobre el vector x:
      a) Grafico de dispersion x_i vs x_{i+1}
      b) Autocorrelograma (ACF)
      c1) Prueba de rachas sobre/bajo la mediana
      c2) Prueba de rachas ascendentes/descendentes
      c3) Prueba de Ljung-Box
    Guarda las figuras en carpeta_fig y retorna un dict con los resultados.
    """
    n = len(x)
    os.makedirs(carpeta_fig, exist_ok=True)

    # ------------------------------------------------------------------
    # (a) Grafico de dispersion x_i vs x_{i+1}
    # Muestra si existe correlacion entre observaciones consecutivas.
    # Si la nube no tiene forma, los datos parecen independientes.
    # ------------------------------------------------------------------
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[:-1], x[1:], s=22, color=AZUL, alpha=0.8)
    ax.set_title(
        f"Dispersion xi vs xi+1  [{variable}]\nr de Pearson = {r1:.3f}",
        fontsize=11, loc="left"
    )
    ax.set_xlabel("x_i")
    ax.set_ylabel("x_i+1")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "02a_dispersion.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # (b) Autocorrelograma (ACF)
    # r_k = correlacion entre x_i y x_{i+k}.
    # Bandas de confianza al 95%: +-1.96/sqrt(n).
    # Si ninguna barra sale de las bandas, no hay autocorrelacion.
    # ------------------------------------------------------------------
    banda = 1.96 / np.sqrt(n)
    max_lags = min(20, n // 2 - 1)   # scipy exige lags < n/2
    fig, ax = plt.subplots(figsize=(9, 3.8))
    plot_acf(x, lags=max_lags, alpha=ALFA, ax=ax, title="",
             color=AZUL, vlines_kwargs={"colors": AZUL})
    ax.axhline( banda, color=NARANJA, ls="--", lw=1, label=f"+- {banda:.3f}")
    ax.axhline(-banda, color=NARANJA, ls="--", lw=1)
    ax.set_title(f"Autocorrelograma [{variable}]", loc="left", fontsize=11)
    ax.set_xlabel("Retardo k")
    ax.set_ylabel("r_k")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "02b_autocorrelograma.png"), dpi=150)
    plt.close()

    # Retardos fuera de las bandas (calculo manual para imprimir)
    xc = x - x.mean()
    rk = np.array([np.sum(xc[:-k] * xc[k:]) / np.sum(xc**2) for k in range(1, max_lags + 1)])
    fuera = np.where(np.abs(rk) > banda)[0] + 1

    # ------------------------------------------------------------------
    # (c-1) Prueba de rachas sobre y bajo la mediana
    # H0: los datos son independientes (aleatorios)
    # Se cuentan las "rachas" (grupos consecutivos de + o -).
    # Un Z grande (valor p pequenio) indica falta de independencia.
    # ------------------------------------------------------------------
    mediana = np.median(x)
    signos = np.where(x > mediana, "+", "-")
    signos = signos[x != mediana]          # se descartan empates con la mediana
    n1 = int(np.sum(signos == "+"))
    n2 = int(np.sum(signos == "-"))
    rachas_med = 1 + int(np.sum(signos[1:] != signos[:-1]))
    media_R = 2 * n1 * n2 / (n1 + n2) + 1
    var_R   = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
               / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    z_med = (rachas_med - media_R) / np.sqrt(var_R)
    p_med = 2 * (1 - stats.norm.cdf(abs(z_med)))

    # ------------------------------------------------------------------
    # (c-2) Prueba de rachas ascendentes y descendentes
    # Cuenta cuantas veces la serie cambia de sentido (sube/baja).
    # ------------------------------------------------------------------
    d = np.sign(np.diff(x))
    d = d[d != 0]                          # elimina empates
    rachas_ad = 1 + int(np.sum(d[1:] != d[:-1]))
    media_ad  = (2 * n - 1) / 3
    var_ad    = (16 * n - 29) / 90
    z_ad  = (rachas_ad - media_ad) / np.sqrt(var_ad)
    p_ad  = 2 * (1 - stats.norm.cdf(abs(z_ad)))

    # ------------------------------------------------------------------
    # (c-3) Prueba de Ljung-Box
    # Prueba conjunta: H0: las primeras lb_lag autocorrelaciones son cero.
    # ------------------------------------------------------------------
    lb_lag = min(10, max_lags)
    lb_res = acorr_ljungbox(x, lags=[lb_lag], return_df=True)
    q_lb   = float(lb_res["lb_stat"].iloc[0])
    p_lb   = float(lb_res["lb_pvalue"].iloc[0])

    # ------------------------------------------------------------------
    # Conclusion global
    # ------------------------------------------------------------------
    independientes = all(p >= ALFA for p in [p_med, p_ad, p_lb])
    conclusion = "INDEPENDIENTES" if independientes else "DEPENDENCIA detectada"

    print(f"\n--- INDEPENDENCIA: {variable} (n={n}) ---")
    print(f"  Retardos ACF fuera de bandas: "
          f"{list(fuera) if len(fuera) else 'ninguno'}"
          f"  (se esperan ~{int(0.05*max_lags)} de {max_lags} por azar)")
    print(f"  (c1) Rachas mediana:    R={rachas_med}, E[R]={media_R:.2f}, "
          f"Z={z_med:.4f}, p={p_med:.4f}  "
          f"-> {'No se rechaza H0' if p_med >= ALFA else 'Se rechaza H0'}")
    print(f"  (c2) Rachas asc/des:   R={rachas_ad}, E[R]={media_ad:.2f}, "
          f"Z={z_ad:.4f}, p={p_ad:.4f}  "
          f"-> {'No se rechaza H0' if p_ad >= ALFA else 'Se rechaza H0'}")
    print(f"  (c3) Ljung-Box(k={lb_lag}): Q={q_lb:.4f}, p={p_lb:.4f}  "
          f"-> {'No se rechaza H0' if p_lb >= ALFA else 'Se rechaza H0'}")
    print(f"  ==> Conclusion: {conclusion}")

    return {
        "Variable":          variable,
        "n":                 n,
        "R_rachas_mediana":  rachas_med,
        "Z_rachas_mediana":  round(z_med, 4),
        "p_rachas_mediana":  round(p_med, 4),
        "dec_rachas_med":    "No se rechaza H0" if p_med >= ALFA else "Se rechaza H0",
        "R_rachas_ascdes":   rachas_ad,
        "Z_rachas_ascdes":   round(z_ad,  4),
        "p_rachas_ascdes":   round(p_ad,  4),
        "dec_rachas_ad":     "No se rechaza H0" if p_ad  >= ALFA else "Se rechaza H0",
        "Q_LjungBox":        round(q_lb,  4),
        "p_LjungBox":        round(p_lb,  4),
        "dec_LjungBox":      "No se rechaza H0" if p_lb  >= ALFA else "Se rechaza H0",
        "Conclusion":        conclusion,
    }


# ===========================================================================
# SECCION B — BONDAD DE AJUSTE
# ===========================================================================
CANDIDATAS = [
    "Normal", "Exponencial", "Gamma", "Erlang", "Lognormal",
    "Weibull", "Beta", "Triangular", "Uniforme", "Logistica",
]

def _ajustar(nombre, x, cota_inf, delta):
    """Ajusta la distribucion 'nombre' por maxima verosimilitud y retorna
    (distribucion_scipy_congelada, texto_parametros, num_parametros_estimados)."""
    if nombre == "Normal":
        m, s = stats.norm.fit(x)
        return stats.norm(m, s), f"mu={m:.3f}, sigma={s:.3f}", 2

    if nombre == "Exponencial":
        loc, esc = stats.expon.fit(x, floc=cota_inf)
        return stats.expon(loc, esc), f"media={esc:.3f}", 1

    if nombre == "Gamma":
        a, loc, esc = stats.gamma.fit(x, floc=cota_inf)
        return stats.gamma(a, loc, esc), f"alfa={a:.3f}, beta={esc:.3f}", 2

    if nombre == "Erlang":
        # Erlang es Gamma con parametro de forma ENTERO
        a, _loc, _esc = stats.gamma.fit(x, floc=cota_inf)
        k   = max(1, int(round(a)))
        esc = (x.mean() - cota_inf) / k
        return stats.gamma(k, cota_inf, esc), f"k={k}, beta={esc:.3f}", 2

    if nombre == "Lognormal":
        s, loc, esc = stats.lognorm.fit(x, floc=cota_inf)
        return stats.lognorm(s, loc, esc), f"mu={np.log(esc):.3f}, sigma={s:.3f}", 2

    if nombre == "Weibull":
        c, loc, esc = stats.weibull_min.fit(x, floc=cota_inf)
        return stats.weibull_min(c, loc, esc), f"forma={c:.3f}, escala={esc:.3f}", 2

    if nombre == "Beta":
        a_ = cota_inf
        b_ = x.max() + 0.05 * (x.max() - x.min())
        p, q, loc, esc = stats.beta.fit(x, floc=a_, fscale=b_ - a_)
        return stats.beta(p, q, loc, esc), f"p={p:.3f}, q={q:.3f}", 4

    if nombre == "Triangular":
        a_ = max(cota_inf, x.min() - delta)
        b_ = x.max() + delta
        c, loc, esc = stats.triang.fit(x, floc=a_, fscale=b_ - a_)
        moda = a_ + c * (b_ - a_)
        return (stats.triang(c, a_, b_ - a_),
                f"min={a_:.2f}, moda={moda:.2f}, max={b_:.2f}", 3)

    if nombre == "Uniforme":
        a_ = max(cota_inf, x.min() - delta)
        b_ = x.max() + delta
        return stats.uniform(a_, b_ - a_), f"min={a_:.2f}, max={b_:.2f}", 2

    if nombre == "Logistica":
        m, s = stats.logistic.fit(x)
        return stats.logistic(m, s), f"mu={m:.3f}, s={s:.3f}", 2

    raise ValueError(f"Distribucion desconocida: {nombre}")


def _prueba_chi2(dist, x, n_par):
    """Chi-cuadrado con clases EQUIPROBABLES (misma frecuencia esperada en cada clase)."""
    n = len(x)
    k = max(3, int(round(1 + 3.322 * np.log10(n))))   # numero de clases (Sturges)
    limites = dist.ppf(np.linspace(0, 1, k + 1))
    limites[0], limites[-1] = -np.inf, np.inf
    obs, _ = np.histogram(x, bins=limites)
    esp    = np.full(k, n / k)
    chi2   = np.sum((obs - esp) ** 2 / esp)
    gl     = k - 1 - n_par                             # grados de libertad
    p      = 1 - stats.chi2.cdf(chi2, gl) if gl > 0 else np.nan
    return chi2, gl, p


def _prueba_ks(dist, x):
    """Kolmogorov-Smirnov: distancia maxima entre la FDA empirica y la teorica."""
    res = stats.kstest(x, dist.cdf)
    return res.statistic, res.pvalue


def _prueba_ad(dist, x):
    """Anderson-Darling: da mas peso a las colas que K-S.
    No tiene valor p estandar aqui; se usa como indicador complementario."""
    n = len(x)
    F = np.clip(dist.cdf(np.sort(x)), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n


def _expr_flexsim(nombre, d):
    """Construye la cadena de texto para pegar en FlexSim."""
    a = d.args
    if nombre == "Normal":
        return f"normal({a[0]:.4f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Exponencial":
        return f"exponential({a[0]:.2f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Gamma":
        return f"gamma({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Erlang":
        return f"erlang({a[1]:.2f}, {a[2]:.4f}, {int(a[0])}, getstream(current))"
    if nombre == "Lognormal":
        return f"lognormal2({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Weibull":
        return f"weibull({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Beta":
        return f"beta({a[2]:.2f}, {a[2]+a[3]:.2f}, {a[0]:.4f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Triangular":
        return f"triangular({a[1]:.2f}, {a[1]+a[2]:.2f}, {a[1]+a[0]*a[2]:.2f}, getstream(current))"
    if nombre == "Uniforme":
        return f"uniform({a[0]:.2f}, {a[0]+a[1]:.2f}, getstream(current))"
    if nombre == "Logistica":
        return f"logistic({a[0]:.4f}, {a[1]:.4f}, getstream(current))"
    return "N/A"


def bondad_ajuste(variable, x_raw, carpeta_fig):
    """
    Ajusta 10 distribuciones candidatas y aplica Chi2, K-S y Anderson-Darling.
    Construye un ranking (100 = mejor), marca las aceptadas y grafica.
    Guarda ranking en CSV y parametros de la mejor en JSON.
    """
    x = np.sort(x_raw)
    n = len(x)
    cota_inf = 0.0
    delta    = (x.max() - x.min()) / max(n - 1, 1)
    os.makedirs(carpeta_fig, exist_ok=True)

    # --- Ajuste y pruebas de cada candidata ---
    filas   = []
    modelos = {}
    for nombre in CANDIDATAS:
        try:
            dist, txt, n_par = _ajustar(nombre, x, cota_inf, delta)
            chi2, gl, p_chi  = _prueba_chi2(dist, x, n_par)
            ks,   p_ks       = _prueba_ks(dist, x)
            ad               = _prueba_ad(dist, x)
            modelos[nombre]  = dist
            filas.append([nombre, txt, chi2, gl, p_chi, ks, p_ks, ad])
        except Exception as err:
            print(f"    [{nombre}] no se pudo ajustar: {err}")

    # --- Tabla de resultados ---
    cols  = ["Distribucion", "Parametros", "Chi2", "gl", "p_Chi2", "KS", "p_KS", "AD"]
    tabla = pd.DataFrame(filas, columns=cols)

    # Puntaje = promedio de los dos valores p (cuanto mas alto, mejor ajuste)
    tabla["Puntaje"] = tabla[["p_Chi2", "p_KS"]].mean(axis=1)
    pmax = tabla["Puntaje"].max()
    tabla["Ranking"] = (100 * tabla["Puntaje"] / pmax) if pmax > 0 else 0.0

    # Una distribucion se acepta si AMBAS pruebas tienen p >= alfa
    acepta = (tabla["p_Chi2"] >= ALFA) & (tabla["p_KS"] >= ALFA)
    tabla["Aceptacion"] = np.where(acepta, "No se rechaza", "Rechazada")
    tabla = tabla.sort_values("Ranking", ascending=False).reset_index(drop=True)

    # Guardar ranking completo
    guardar_csv(tabla, os.path.join(BASE, "resultados", f"ranking_{variable}.csv"), index=False)

    # Mejor distribucion = la primera del ranking que no se rechaza
    aceptadas = tabla.loc[tabla["Aceptacion"] == "No se rechaza", "Distribucion"].tolist()
    if aceptadas:
        mejor = aceptadas[0]
    else:
        mejor = tabla.loc[0, "Distribucion"]
        print(f"  [AVISO] Ninguna distribucion paso ambas pruebas en {variable}.")

    graficar = aceptadas[:4] if aceptadas else [mejor]

    # --- Imprimir tabla resumen ---
    print(f"\n--- BONDAD DE AJUSTE: {variable} (n={n}) ---")
    print(tabla[["Distribucion", "p_Chi2", "p_KS", "AD", "Ranking", "Aceptacion"]]
          .round({"p_Chi2": 4, "p_KS": 4, "AD": 3, "Ranking": 2})
          .to_string(index=False))
    print(f"  No se rechazan : {aceptadas if aceptadas else 'ninguna'}")
    print(f"  Mejor ajuste   : {mejor}")

    # --- Grafico 1: Histograma + curvas de densidad ---
    k_sturges = max(3, int(round(1 + 3.322 * np.log10(n))))
    malla = np.linspace(max(cota_inf, x.min() - 1), x.max() + 1, 400)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(x, bins=k_sturges, density=True,
            color="#D6E6EA", edgecolor="white", label="Datos")
    for nom, col in zip(graficar, COLORES_GRAF):
        try:
            ax.plot(malla, modelos[nom].pdf(malla), color=col, lw=2.2, label=nom)
        except Exception:
            pass
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"{variable}: histograma y distribuciones aceptadas", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "03a_densidades.png"), dpi=150)
    plt.close()

    # --- Grafico 2: FDA empirica vs teóricas ---
    Fn = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.step(x, Fn, where="post", color="black", lw=1.5, label="Empirica")
    for nom, col in zip(graficar, COLORES_GRAF):
        try:
            ax.plot(malla, modelos[nom].cdf(malla), color=col, lw=2, label=nom)
        except Exception:
            pass
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("F(x)")
    ax.set_title(f"{variable}: acumulada empirica vs modelos", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "03b_acumuladas.png"), dpi=150)
    plt.close()

    # --- Grafico 3: P-P y Q-Q del mejor modelo ---
    dist_mejor = modelos[mejor]
    p_emp = (np.arange(1, n + 1) - 0.5) / n
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))

    # P-P: probabilidad empirica vs probabilidad teorica
    axs[0].scatter(p_emp, dist_mejor.cdf(x), s=16, color=AZUL)
    axs[0].plot([0, 1], [0, 1], color=NARANJA, lw=1.5)
    axs[0].set_title(f"Grafico P-P: {mejor}", loc="left")
    axs[0].set_xlabel("Probabilidad empirica")
    axs[0].set_ylabel("Probabilidad teorica")

    # Q-Q: cuantil teorico vs dato observado
    q_teo = dist_mejor.ppf(p_emp)
    axs[1].scatter(q_teo, x, s=16, color=AZUL)
    lim = [min(q_teo.min(), x.min()), max(q_teo.max(), x.max())]
    axs[1].plot(lim, lim, color=NARANJA, lw=1.5)
    axs[1].set_title(f"Grafico Q-Q: {mejor}", loc="left")
    axs[1].set_xlabel("Cuantil teorico (min)")
    axs[1].set_ylabel("Dato observado (min)")

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "03c_pp_qq.png"), dpi=150)
    plt.close()

    # --- Expresion FlexSim y archivo JSON ---
    expr = _expr_flexsim(mejor, dist_mejor)
    fila_mejor = int(tabla.index[tabla["Distribucion"] == mejor][0])
    print(f"  Expresion FlexSim: {expr}")

    resultado_json = {
        "variable":    variable,
        "distribucion": mejor,
        "parametros":  tabla.loc[fila_mejor, "Parametros"],
        "scipy":       dist_mejor.dist.name,
        "args":        [float(v) for v in dist_mejor.args],
        "flexsim":     expr,
    }
    guardar_json(resultado_json, os.path.join(BASE, "resultados", f"ajuste_{variable}.json"),
                 ensure_ascii=False, indent=2)

    return {
        "Variable":       variable,
        "No_se_rechazan": ", ".join(aceptadas) if aceptadas else "ninguna",
        "Mejor_ajuste":   mejor,
        "Parametros":     tabla.loc[fila_mejor, "Parametros"],
        "FlexSim":        expr,
        "Ranking_mejor":  round(float(tabla.loc[0, "Ranking"]), 2),
    }


# ===========================================================================
# SECCION C — VALIDACION DEL MODELO
# ===========================================================================
def validacion(carpeta_fig):
    """
    Valida el modelo de simulacion comparando resultados reales vs simulados.
    Pasos:
      1. Prueba piloto -> estimar desviacion estandar s
      2. Calcular numero de corridas necesarias n = (s*Z/epsilon)^2
      3. Prueba F de igualdad de varianzas
      4. Prueba t de diferencia de medias (varianzas iguales o Welch)
      5. Prueba z (equivalente a Excel)
    """
    os.makedirs(carpeta_fig, exist_ok=True)

    real     = pd.read_csv(os.path.join(CARPETA_DATOS, "validacion_real.csv"))["piezas_turno"].to_numpy()
    piloto   = pd.read_csv(os.path.join(CARPETA_DATOS, "validacion_piloto.csv"))["piezas_turno"].to_numpy()
    simulado = pd.read_csv(os.path.join(CARPETA_DATOS, "validacion_corridas.csv"))["piezas_turno"].to_numpy()

    # ------------------------------------------------------------------
    # Paso 1 y 2: Prueba piloto -> numero de corridas
    # ------------------------------------------------------------------
    n0      = len(piloto)
    s0      = piloto.std(ddof=1)
    EPSILON = 1.5                             # error maximo aceptado: +-1.5 piezas
    z_val   = stats.norm.ppf(1 - ALFA / 2)   # Z_{alpha/2} = 1.96 con alfa=0.05
    n_z     = math.ceil((s0 * z_val / EPSILON) ** 2)
    t_val   = stats.t.ppf(1 - ALFA / 2, n0 - 1)
    n_t     = math.ceil((s0 * t_val / EPSILON) ** 2)

    # ------------------------------------------------------------------
    # Paso 4a: Prueba F de igualdad de varianzas
    # H0: sigma^2(modelo) = sigma^2(real)
    # ------------------------------------------------------------------
    v_sim  = simulado.var(ddof=1)
    v_real = real.var(ddof=1)
    F_stat = v_sim / v_real
    gl1, gl2 = len(simulado) - 1, len(real) - 1
    p_F = 2 * min(stats.f.cdf(F_stat, gl1, gl2),
                  1 - stats.f.cdf(F_stat, gl1, gl2))
    var_iguales = p_F >= ALFA

    # ------------------------------------------------------------------
    # Paso 4b: Prueba t de diferencia de medias
    # H0: mu(modelo) = mu(real)
    # ------------------------------------------------------------------
    res_t  = stats.ttest_ind(simulado, real, equal_var=var_iguales)
    t_crit = stats.t.ppf(1 - ALFA / 2, res_t.df)
    ic     = res_t.confidence_interval(confidence_level=1 - ALFA)

    # ------------------------------------------------------------------
    # Paso 4c: Prueba z (Excel: "Prueba z para medias de dos muestras")
    # ------------------------------------------------------------------
    dif   = simulado.mean() - real.mean()
    z_cal = dif / math.sqrt(v_sim / len(simulado) + v_real / len(real))
    p_z   = 2 * (1 - stats.norm.cdf(abs(z_cal)))

    valido = res_t.pvalue >= ALFA

    # --- Imprimir resumen ---
    print(f"\n{'='*60}")
    print("  VALIDACION DEL MODELO")
    print(f"{'='*60}")
    print(f"  Datos reales   : n={len(real)}, media={real.mean():.3f}, s={real.std(ddof=1):.3f}")
    print(f"  Datos simulados: n={len(simulado)}, media={simulado.mean():.3f}, s={simulado.std(ddof=1):.3f}")
    print(f"  --- Paso 1-2: Prueba piloto ---")
    print(f"  Piloto: n0={n0}, media={piloto.mean():.3f}, s={s0:.3f}")
    print(f"  epsilon={EPSILON}, Z={z_val:.3f}, t={t_val:.3f}")
    print(f"  Corridas necesarias: n_Z={n_z}  n_t={n_t}")
    print(f"  Corridas disponibles: {len(simulado)} "
          f"{'[OK]' if len(simulado) >= n_z else '[INSUFICIENTES]'}")
    print(f"  --- Paso 4a: Prueba F ---")
    print(f"  F={F_stat:.4f}, gl=({gl1},{gl2}), p={p_F:.4f}  "
          f"-> varianzas {'IGUALES' if var_iguales else 'DIFERENTES (Welch)'}")
    tipo_t = "varianzas iguales" if var_iguales else "Welch (varianzas desiguales)"
    print(f"  --- Paso 4b: Prueba t ({tipo_t}) ---")
    print(f"  t={res_t.statistic:.4f}, gl={res_t.df:.1f}, t_critico=+-{t_crit:.4f}, p={res_t.pvalue:.4f}")
    print(f"  IC 95%: [{ic.low:.3f} ; {ic.high:.3f}]  "
          f"({'contiene el 0' if ic.low <= 0 <= ic.high else 'NO contiene el 0'})")
    print(f"  --- Paso 4c: Prueba z ---")
    print(f"  z={z_cal:.4f}, z_critico=+-{z_val:.3f}, p={p_z:.4f}")
    print(f"  ==> {'MODELO VALIDO (no se rechaza H0)' if valido else 'MODELO NO VALIDO - revisar supuestos'}")

    # --- Grafico comparativo ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4),
                            gridspec_kw={"width_ratios": [1.1, 1]})

    # Boxplots lado a lado
    bp = axs[0].boxplot([real, simulado], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
    for caja, color in zip(bp["boxes"], [AZUL, NARANJA]):
        caja.set_facecolor(color)
        caja.set_alpha(0.75)
    axs[0].set_xticks([1, 2], ["Real", "Simulado"])
    axs[0].set_ylabel("Piezas terminadas por turno")
    axs[0].set_title("Distribucion de los resultados", loc="left")

    # Medias con IC 95%
    medias = [real.mean(), simulado.mean()]
    err = [
        stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m))
        for m in (real, simulado)
    ]
    axs[1].errorbar([0, 1], medias, yerr=err,
                    fmt="o", ms=9, capsize=8, color=GRIS, ecolor=GRIS)
    axs[1].scatter([0, 1], medias, s=90, color=[AZUL, NARANJA], zorder=3)
    axs[1].set_xticks([0, 1], ["Real", "Simulado"])
    axs[1].set_xlim(-0.6, 1.6)
    axs[1].set_title("Medias con IC 95%", loc="left")
    for i, m in enumerate(medias):
        axs[1].annotate(f"{m:.2f}", (i + 0.12, m), va="center")

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_fig, "04_validacion.png"), dpi=150)
    plt.close()

    return {
        "n_piloto":             n0,
        "s_piloto":             round(s0, 4),
        "corridas_necesarias_z": n_z,
        "corridas_necesarias_t": n_t,
        "corridas_disponibles": len(simulado),
        "F_stat":               round(F_stat, 4),
        "p_F":                  round(p_F, 4),
        "varianzas_iguales":    var_iguales,
        "t_stat":               round(res_t.statistic, 4),
        "gl_t":                 round(res_t.df, 1),
        "p_t":                  round(res_t.pvalue, 4),
        "IC_low":               round(ic.low, 4),
        "IC_high":              round(ic.high, 4),
        "z_stat":               round(z_cal, 4),
        "p_z":                  round(p_z, 4),
        "Conclusion":           "MODELO VALIDO" if valido else "MODELO NO VALIDO",
    }


# ===========================================================================
# EJECUCION PRINCIPAL
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ANALISIS COMPLETO — Ejemplo 6.1  (alfa = 0.05)")
    print("=" * 60)

    filas_indep  = []
    filas_bondad = []

    for var in VARIABLES_TIEMPO:
        csv_path = os.path.join(CARPETA_DATOS, f"{var}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [AVISO] No existe el archivo: {csv_path}")
            continue

        x = pd.read_csv(csv_path)["tiempo_min"].to_numpy()
        carpeta_fig = os.path.join(BASE, "figuras", var)

        # A) Independencia
        r_indep = pruebas_independencia(var, x, carpeta_fig)
        filas_indep.append(r_indep)

        # B) Bondad de ajuste
        r_bondad = bondad_ajuste(var, x, carpeta_fig)
        filas_bondad.append(r_bondad)

    # C) Validacion
    carpeta_fig_val = os.path.join(BASE, "figuras", "validacion")
    r_val = validacion(carpeta_fig_val)

    # Guardar tablas resumen consolidadas
    guardar_csv(pd.DataFrame(filas_indep),
                os.path.join(BASE, "resultados", "independencia_resumen.csv"), index=False)
    guardar_csv(pd.DataFrame(filas_bondad),
                os.path.join(BASE, "resultados", "bondad_ajuste_resumen.csv"), index=False)
    guardar_csv(pd.DataFrame([r_val]),
                os.path.join(BASE, "resultados", "validacion_resumen.csv"), index=False)

    print("\n" + "=" * 60)
    print("  ANALISIS TERMINADO")
    print("  Figuras  -> figuras/<variable>/")
    print("  Tablas   -> resultados/*.csv")
    print("  Parametros FlexSim -> resultados/ajuste_<variable>.json")
    print("=" * 60)
