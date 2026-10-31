"""Utilidades compartidas de los notebooks DAMMG – Módulo VI.

Misma convención que los widgets (docs/notacion.md):
    J(w, b) = (1/2m) Σ (f(x) - y)² + (λ/2m) ||w||²   (b no se regulariza)

Este archivo se descarga en Colab desde la release pública; no importa nada del repo privado.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

# Colores fijos por símbolo (iguales a packages/ui/src/tokens.css)
COLORES = {
    "papel": "#fdfdfd",
    "tinta": "#1a1a1a",
    "w": "#0847c4",
    "b": "#575380",
    "lambda": "#f27900",
    "error": "#fb2c36",
    "yhat": "#2e7d5b",
    "amarillo": "#ffcd42",
    "datos": "#dbe8ff",
}


def estilo_dammg():
    """Aplica el estilo brutalista a matplotlib (bordes gruesos, sin degradados)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": COLORES["papel"],
        "axes.facecolor": COLORES["papel"],
        "axes.edgecolor": COLORES["tinta"],
        "axes.linewidth": 2.0,
        "axes.grid": True,
        "grid.color": "#e4e4e4",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "scatter.edgecolors": COLORES["tinta"],
        "legend.frameon": True,
        "legend.edgecolor": COLORES["tinta"],
        "legend.fancybox": False,
        "axes.prop_cycle": mpl.cycler(color=[COLORES["w"], COLORES["error"], COLORES["yhat"],
                                             COLORES["lambda"], COLORES["b"]]),
    })


# ---------- regresión lineal ----------
def costo_lineal(x, y, w, b, lam=0.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = len(y)
    e = w * x + b - y
    return float((e @ e) / (2 * m) + lam / (2 * m) * np.sum(np.square(w)))


def gradiente_lineal(x, y, w, b, lam=0.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = len(y)
    e = w * x + b - y
    return float((e @ x) / m + lam / m * w), float(e.sum() / m)


def descenso_gradiente(x, y, w0, b0, alpha, iters, lam=0.0):
    w, b = float(w0), float(b0)
    camino = [(w, b, costo_lineal(x, y, w, b, lam))]
    for _ in range(iters):
        dw, db = gradiente_lineal(x, y, w, b, lam)
        w, b = w - alpha * dw, b - alpha * db
        J = costo_lineal(x, y, w, b, lam)
        camino.append((w, b, J))
        if not np.isfinite(J) or abs(J) > 1e12:
            break
    return camino


def ols_1d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx = np.sum((x - x.mean()) ** 2)
    if sxx == 0:
        return 0.0, float(y.mean())
    w = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    return w, float(y.mean() - w * x.mean())


def ridge_svd(X, y, lam):
    """Ridge sin intercepto vía SVD: w = V diag(d/(d²+λ)) Uᵀ y. Con λ=0 es la pseudo-inversa."""
    U, d, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    denom = d ** 2 + lam
    f = np.where(denom > 1e-12, d / np.where(denom > 1e-12, denom, 1.0), 0.0)
    return Vt.T @ (f * (U.T @ np.asarray(y, float)))


def factores_encogimiento(X, lam):
    """d²/(d²+λ) por dirección principal: cuánto sobrevive cada dirección."""
    d = np.linalg.svd(np.asarray(X, float), compute_uv=False)
    return d ** 2 / (d ** 2 + lam)


# ---------- logística ----------
def sigmoide(z):
    z = np.asarray(z, float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def log_loss(y, p, eps=EPS):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------- árboles ----------
def entropia(conteos):
    c = np.asarray(conteos, float)
    n = c.sum()
    if n == 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def gini(conteos):
    c = np.asarray(conteos, float)
    n = c.sum()
    return 0.0 if n == 0 else float(1 - np.sum((c / n) ** 2))


def ganancia_info(padre, izq, der, impureza=entropia):
    n, nl, nr = (float(np.sum(v)) for v in (padre, izq, der))
    if n == 0:
        return 0.0
    return impureza(padre) - nl / n * impureza(izq) - nr / n * impureza(der)


# ---------- SVM ----------
def hinge(y, f):
    return np.maximum(0.0, 1 - np.asarray(y, float) * np.asarray(f, float))


def phi_poli2(x):
    r2 = np.sqrt(2)
    x0, x1 = x
    return np.array([x0 * x0, x1 * x1, r2 * x0 * x1, r2 * x0, r2 * x1, 1.0])


# ---------- ejemplos de juguete (Ng) ----------
CASAS_NG = {
    "x": [1.0, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.1, 2.3, 2.5, 2.7, 3.0],
    "y": [245, 280, 300, 312, 340, 365, 400, 395, 440, 470, 500, 560],
}


def entorno_listo():
    """Celda L0: verificación del entorno."""
    import sklearn

    version = tuple(int(v) for v in np.__version__.split(".")[:2])
    assert version >= (1, 24), f"numpy {np.__version__} es demasiado viejo: reinicia el entorno y vuelve a correr L0"
    print(f"✅ entorno listo · numpy {np.__version__} · scikit-learn {sklearn.__version__}")


def _nombres_globales(codigo):
    """Nombres globales que LEE un objeto código (no atributos), incluidas funciones anidadas y lambdas."""
    import dis

    nombres = {i.argval for i in dis.get_instructions(codigo) if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
    for c in codigo.co_consts:
        if hasattr(c, "co_code"):
            nombres |= _nombres_globales(c)
    return nombres


def revisa_aislada(f, graded=()):
    """Autocomprobación: la función no debe depender de nada que el evaluador no vea.

    El evaluador solo extrae las celdas `graded` y solo permite numpy y math. Si `f` usa `du.`,
    una variable del notebook (datos, m, alpha...) o un módulo como pandas, en tu Colab funciona
    pero en el evaluador truena con NameError. `graded` son las otras funciones evaluables que sí
    puede usar. Una función auxiliar tuya solo genera un aviso: asegúrate de que esté en una celda graded.
    """
    import inspect
    import types

    globales = f.__globals__
    usadas = sorted(n for n in _nombres_globales(f.__code__) if n in globales and n not in graded)
    malos, avisos = [], []
    for n in usadas:
        v = globales[n]
        if isinstance(v, types.ModuleType):
            if v.__name__.split(".")[0] not in ("numpy", "math"):
                malos.append(f"`{n}` (módulo {v.__name__})")
        elif callable(v) and getattr(v, "__module__", None) == "__main__" and inspect.isfunction(v):
            avisos.append(n)
        elif not callable(v):
            malos.append(f"`{n}` (variable del notebook)")
        else:
            malos.append(f"`{n}`")
    assert not malos, (f"`{f.__name__}` usa {', '.join(malos)}, que está fuera de las celdas graded. "
                       "En el evaluador eso da NameError: usa solo los argumentos de la función, numpy y math.")
    for n in avisos:
        print(f"⚠ `{f.__name__}` llama a `{n}`: si no está en una celda graded, el evaluador no la verá.")
