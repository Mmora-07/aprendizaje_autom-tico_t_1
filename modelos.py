"""Modelos lineales implementados desde cero con NumPy (Taller 1).

Los tres notebooks importan los modelos desde este archivo para no
reimplementarlos en cada experimento.

Convención de las pérdidas (igual a la del enunciado):

    z = X w + b 1
    J_reg(w, b) = (1 / 2N) ||z - y||²
    J_log(w, b) = -(1/N) Σ [y log p + (1 - y) log(1 - p)],   p = σ(z)
    F(w, b)     = J(w, b) + λ1 ||w||₁ + (λ2 / 2) ||w||²

El intercepto b nunca se penaliza.

Contenido
---------
- Funciones base: sigmoide, pérdida logística, objetivos, gradientes y umbral suave.
- RegresionOLS:          mínimos cuadrados con np.linalg.lstsq.
- RegresionRidge:        Ridge resolviendo un sistema lineal (np.linalg.solve).
- RegresionLinealGD:     descenso por gradiente para regresión lineal (opción L2).
- LassoProximal:         gradiente proximal para Lasso lineal.
- RegresionLogisticaGD:  descenso por gradiente para regresión logística (opción L2).
"""

import json

import numpy as np

SEMILLA = 2026


# =============================================================================
# Funciones base
# =============================================================================

def sigmoid(z):
    """Sigmoide σ(z) = 1 / (1 + e^{-z}) evaluada de forma estable.

    Parameters
    ----------
    z : float o numpy.ndarray
        Scores reales.

    Returns
    -------
    numpy.ndarray
        Probabilidades en [0, 1] con la misma forma que z.
    """
    # Reutilizado de clase: la expresión exp(-logaddexp(0, -z)).
    # Mejoras: solo se documenta por qué es estable.
    # σ(z) = exp(-log(1 + e^{-z})). np.logaddexp(0, -z) calcula log(1 + e^{-z})
    # sin formar e^{-z}, así que no desborda para z = -1000; el exp final da
    # a lo sumo un subdesbordamiento a 0, que sigue siendo una probabilidad válida.
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def perdida_logistica(y, z):
    """Pérdida logística media J_log calculada directamente desde los scores.

    Usa la forma equivalente  ℓ(y, z) = log(1 + e^{z}) - y z  (problema 3a),
    que nunca evalúa log(0) y no requiere recortar probabilidades.

    Parameters
    ----------
    y : numpy.ndarray de forma (N,)
        Etiquetas 0 o 1.
    z : numpy.ndarray de forma (N,)
        Scores lineales z = Xw + b.

    Returns
    -------
    float
        Pérdida media.
    """
    # Reutilizado de clase: calcular la pérdida desde z con np.logaddexp.
    # Mejoras: se usa la forma log(1 + e^z) - y z, la misma que se deriva en el
    # problema 3a, en lugar de elegir el signo con np.where.
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def objetivo_lineal(w, b, X, y, lam1=0.0, lam2=0.0):
    """Objetivo completo F = J_reg + λ1 ||w||₁ + (λ2/2) ||w||² para regresión."""
    residuo = X @ w + b - y
    return float(residuo @ residuo / (2 * len(y))
                 + lam1 * np.sum(np.abs(w)) + lam2 / 2 * (w @ w))


def gradiente_lineal(w, b, X, y, lam2=0.0):
    """Gradiente de la parte diferenciable J_reg + (λ2/2)||w||².

    Returns
    -------
    grad_w : numpy.ndarray de forma (D,)
        (1/N) Xᵀ(z - y) + λ2 w
    grad_b : float
        (1/N) Σ (z_n - y_n)   (el intercepto no lleva penalización)
    """
    error = X @ w + b - y
    return X.T @ error / len(y) + lam2 * w, float(np.mean(error))


def objetivo_logistico(w, b, X, y, lam2=0.0):
    """Objetivo completo F = J_log + (λ2/2) ||w||² para clasificación."""
    return perdida_logistica(y, X @ w + b) + lam2 / 2 * float(w @ w)


def gradiente_logistico(w, b, X, y, lam2=0.0):
    """Gradiente de J_log + (λ2/2)||w||².

    Returns
    -------
    grad_w : numpy.ndarray de forma (D,)
        (1/N) Xᵀ(p - y) + λ2 w
    grad_b : float
        (1/N) Σ (p_n - y_n)
    """
    error = sigmoid(X @ w + b) - y
    return X.T @ error / len(y) + lam2 * w, float(np.mean(error))


def umbral_suave(u, t):
    """Operador de umbral suave S_t(u) = sign(u) · max(|u| - t, 0).

    Es el operador proximal de t·||·||₁ (problema 2a): las coordenadas con
    |u_j| <= t quedan exactamente en cero y las demás se encogen t hacia cero.
    """
    return np.sign(u) * np.maximum(np.abs(u) - t, 0.0)


# =============================================================================
# Validaciones
# =============================================================================

def _validar_X(X, n_caracteristicas=None):
    """Convierte X a arreglo float y verifica que sea 2D, no vacío y finito."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz 2D; tiene {X.ndim} dimensiones.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X no puede estar vacía.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene valores no finitos (NaN o inf).")
    if n_caracteristicas is not None and X.shape[1] != n_caracteristicas:
        raise ValueError(f"X tiene {X.shape[1]} columnas; el modelo se ajustó "
                         f"con {n_caracteristicas}.")
    return X


def _validar_Xy(X, y, binaria=False):
    """Valida X e y juntos: dimensiones compatibles, valores finitos y etiquetas 0/1."""
    X = _validar_X(X)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y debe ser un vector 1D; tiene {y.ndim} dimensiones.")
    if len(y) != X.shape[0]:
        raise ValueError(f"X tiene {X.shape[0]} filas pero y tiene {len(y)}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene valores no finitos (NaN o inf).")
    if binaria and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("En clasificación y solo puede contener 0 y 1.")
    return X, y


def _validar_no_negativo(nombre, valor):
    if not (np.isscalar(valor) and np.isfinite(valor) and valor >= 0):
        raise ValueError(f"{nombre} debe ser un número finito >= 0; se recibió {valor!r}.")


def _validar_positivo(nombre, valor):
    if not (np.isscalar(valor) and np.isfinite(valor) and valor > 0):
        raise ValueError(f"{nombre} debe ser un número finito > 0; se recibió {valor!r}.")


def _validar_entero_positivo(nombre, valor):
    if not (isinstance(valor, (int, np.integer)) and valor > 0):
        raise ValueError(f"{nombre} debe ser un entero > 0; se recibió {valor!r}.")


def _matriz_aumentada(X):
    """X̃ = [1_N  X]: agrega la columna de unos del intercepto."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _matriz_diseno(X, ajustar_intercepto):
    """Matriz con la que se ajusta: X̃ si hay intercepto, X si b se fija en 0."""
    return _matriz_aumentada(X) if ajustar_intercepto else X


# =============================================================================
# Clase base
# =============================================================================

class _ModeloLineal:
    """Comportamiento común: predicción, objetivo y guardado de parámetros.

    Atributos que deja `fit` en todos los modelos
    ---------------------------------------------
    coef_ : numpy.ndarray de forma (D,)
    intercept_ : float
    objetivo_final_ : float
        Valor del objetivo completo F en la solución.
    n_iter_ : int
        Iteraciones realizadas (0 en las soluciones directas).
    motivo_parada_ : str
        "solucion directa", "convergencia", "max_iter" o "divergencia".

    Todos los modelos aceptan `ajustar_intercepto` (True por defecto). Con
    False el intercepto se fija en b = 0 y no se estima (problema 5).
    """

    def _verificar_ajustado(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("El modelo no está ajustado: llame primero a fit.")

    def decision_function(self, X):
        """Scores lineales z = Xw + b."""
        self._verificar_ajustado()
        X = _validar_X(X, len(self.coef_))
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predicciones continuas Xw + b (en regresión coincide con el score)."""
        return self.decision_function(X)

    def objetivo(self, X, y):
        """Evalúa el objetivo completo F del modelo con los parámetros actuales."""
        self._verificar_ajustado()
        X, y = _validar_Xy(X, y)
        return self._objetivo(self.coef_, self.intercept_, X, y)

    def _guardar_theta(self, theta):
        """Separa θ en intercepto y coeficientes según `ajustar_intercepto`."""
        if self.ajustar_intercepto:
            self.intercept_, self.coef_ = float(theta[0]), theta[1:]
        else:
            self.intercept_, self.coef_ = 0.0, theta

    def guardar_parametros(self, ruta):
        """Guarda coeficientes, intercepto e hiperparámetros en un archivo JSON.

        JSON escribe los float con la representación más corta que se lee de
        vuelta exactamente igual, así que las predicciones no cambian al recargar.
        """
        self._verificar_ajustado()
        datos = {"modelo": type(self).__name__,
                 "hiperparametros": self._hiperparametros(),
                 "coef": self.coef_.tolist(),
                 "intercept": float(self.intercept_)}
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, indent=2)

    def cargar_parametros(self, ruta):
        """Carga coeficientes e intercepto guardados con `guardar_parametros`."""
        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        if datos["modelo"] != type(self).__name__:
            raise ValueError(f"El archivo es de {datos['modelo']}, no de {type(self).__name__}.")
        self.coef_ = np.array(datos["coef"], dtype=float)
        self.intercept_ = float(datos["intercept"])
        return self


# =============================================================================
# Soluciones directas: OLS y Ridge
# =============================================================================

class RegresionOLS(_ModeloLineal):
    """Mínimos cuadrados ordinarios resueltos con np.linalg.lstsq.

    lstsq usa la SVD de X̃, por lo que también funciona si X̃ no tiene rango
    completo (p. ej., columnas duplicadas). En ese caso hay infinitas
    soluciones y lstsq devuelve la de norma mínima; las predicciones X̃θ son
    las mismas para todas.

    Parameters
    ----------
    ajustar_intercepto : bool
        Si es False se usa X en lugar de X̃ y b = 0.

    Atributos adicionales
    ---------------------
    rango_ : int
        Rango numérico de la matriz de diseño. Si es menor que su número de
        columnas, la solución no es única.
    """

    def __init__(self, ajustar_intercepto=True):
        self.ajustar_intercepto = ajustar_intercepto

    def _hiperparametros(self):
        return {"ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_lineal(w, b, X, y)

    def fit(self, X, y):
        """Ajusta θ = (b, w) minimizando (1/2N)||X̃θ - y||²."""
        X, y = _validar_Xy(X, y)
        A = _matriz_diseno(X, self.ajustar_intercepto)
        theta, _, self.rango_, _ = np.linalg.lstsq(A, y, rcond=None)
        self._guardar_theta(theta)
        self.objetivo_final_ = self._objetivo(self.coef_, self.intercept_, X, y)
        self.n_iter_, self.motivo_parada_ = 0, "solucion directa"
        return self


class RegresionRidge(_ModeloLineal):
    """Ridge con intercepto no penalizado, resolviendo las ecuaciones normales.

    Minimiza F = (1/2N)||X̃θ - y||² + (λ2/2) θᵀPθ, con P = diag(0, 1, ..., 1).
    Igualando el gradiente a cero (problema 1a):

        (X̃ᵀX̃ + N λ2 P) θ = X̃ᵀ y

    Con λ2 > 0 la matriz es definida positiva aunque X tenga columnas
    dependientes (problema 1b), así que np.linalg.solve encuentra la solución
    única sin calcular ninguna inversa.

    Parameters
    ----------
    lam2 : float > 0
        Intensidad de la penalización L2.
    ajustar_intercepto : bool
        Si es False se resuelve (XᵀX + Nλ2 I) w = Xᵀy con b = 0.
    """

    def __init__(self, lam2=1.0, ajustar_intercepto=True):
        self.lam2, self.ajustar_intercepto = lam2, ajustar_intercepto

    def _hiperparametros(self):
        return {"lam2": self.lam2, "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_lineal(w, b, X, y, lam2=self.lam2)

    def fit(self, X, y):
        """Ajusta θ = (b, w) resolviendo (X̃ᵀX̃ + Nλ2P)θ = X̃ᵀy."""
        _validar_positivo("lam2", self.lam2)
        X, y = _validar_Xy(X, y)
        A = _matriz_diseno(X, self.ajustar_intercepto)
        P = np.eye(A.shape[1])
        if self.ajustar_intercepto:
            P[0, 0] = 0.0  # el intercepto no se penaliza
        theta = np.linalg.solve(A.T @ A + len(y) * self.lam2 * P, A.T @ y)
        self._guardar_theta(theta)
        self.objetivo_final_ = self._objetivo(self.coef_, self.intercept_, X, y)
        self.n_iter_, self.motivo_parada_ = 0, "solucion directa"
        return self


# =============================================================================
# Métodos iterativos: descenso por gradiente y gradiente proximal
# =============================================================================

def _gradiente_proximal(gradiente, objetivo, w, b, lam1, eta, tol, max_iter,
                        ajustar_intercepto=True):
    """Bucle común de los modelos iterativos.

    Minimiza f(w, b) + λ1||w||₁, donde f es la parte diferenciable, con

        w⁺ = S_{ηλ1}(w - η ∇_w f)      (umbral suave)
        b⁺ = b - η ∂f/∂b               (el intercepto no se penaliza)

    Si λ1 = 0, el umbral suave es la identidad y el bucle es descenso por
    gradiente ordinario. Si ajustar_intercepto es False, b no se actualiza.

    Criterio de parada: ||θ⁺ - θ||₂ / η < tol, con θ = (b, w). Para λ1 = 0
    es la norma del gradiente; para λ1 > 0 es el residuo de la actualización
    proximal, que vale cero solo en un minimizador de F (no basta con que
    el gradiente de la parte suave sea cero).

    Returns
    -------
    w, b : parámetros finales
    historial : list[float]
        Objetivo completo F en el punto inicial y tras cada iteración.
    residuo : float
        Último valor de ||θ⁺ - θ||₂ / η.
    n_iter : int
    motivo : str
        "convergencia", "max_iter" o "divergencia".
    """
    # Reutilizado de clase: la actualización de fit_regularized (paso de gradiente
    # + umbral suave en w, paso normal en b) y su criterio ||Δθ|| / η.
    # Mejoras: se registra el objetivo completo en cada iteración, el residuo
    # final y el motivo de terminación, y se detecta la divergencia.
    historial = [objetivo(w, b)]
    residuo, motivo = np.inf, "max_iter"
    for n_iter in range(1, max_iter + 1):
        grad_w, grad_b = gradiente(w, b)
        w_nuevo = umbral_suave(w - eta * grad_w, eta * lam1)
        b_nuevo = b - eta * grad_b if ajustar_intercepto else b
        residuo = np.sqrt(np.sum((w_nuevo - w) ** 2) + (b_nuevo - b) ** 2) / eta
        w, b = w_nuevo, b_nuevo
        historial.append(objetivo(w, b))
        if not np.isfinite(historial[-1]):
            motivo = "divergencia"
            break
        if residuo < tol:
            motivo = "convergencia"
            break
    return w, b, historial, residuo, n_iter, motivo


class _ModeloIterativo(_ModeloLineal):
    """Parte común de los modelos que se ajustan iterando.

    Parameters
    ----------
    eta : float > 0 o None
        Tamaño de paso. Si es None se usa η = 1/L, donde L es la constante de
        Lipschitz del gradiente de la parte suave (ver `_lipschitz`). Con ese
        paso el descenso (y el gradiente proximal) disminuye el objetivo en
        cada iteración.
    tol : float > 0
        Tolerancia del criterio ||θ⁺ - θ||₂ / η < tol.
    max_iter : int > 0
        Máximo de iteraciones.
    warm_start : bool
        Si es False (por defecto), cada llamada a fit reinicia en w = 0, b = 0.
        Si es True y el modelo ya estaba ajustado con el mismo número de
        columnas, fit continúa desde los parámetros anteriores.
    ajustar_intercepto : bool
        Si es False, b se mantiene en 0 durante todo el ajuste.

    Atributos adicionales
    ---------------------
    eta_ : float
        Paso usado en el último ajuste.
    historial_objetivo_ : list[float]
        Objetivo completo F por iteración (incluye el punto inicial).
    residuo_final_ : float
        Último valor de ||θ⁺ - θ||₂ / η.
    """

    def __init__(self, eta=None, tol=1e-8, max_iter=100_000, warm_start=False,
                 ajustar_intercepto=True):
        self.eta, self.tol, self.max_iter, self.warm_start = eta, tol, max_iter, warm_start
        self.ajustar_intercepto = ajustar_intercepto

    def _validar_hiperparametros(self):
        if self.eta is not None:
            _validar_positivo("eta", self.eta)
        _validar_positivo("tol", self.tol)
        _validar_entero_positivo("max_iter", self.max_iter)

    def _parametros_iniciales(self, n_caracteristicas):
        """Arranque en w = 0, b = 0, salvo arranque en caliente declarado."""
        if self.warm_start and hasattr(self, "coef_") and len(self.coef_) == n_caracteristicas:
            return self.coef_.copy(), float(self.intercept_)
        return np.zeros(n_caracteristicas), 0.0

    def _ajustar(self, X, y, lam1=0.0):
        """Ejecuta el bucle iterativo y guarda los atributos del ajuste."""
        if self.eta is None:
            self.eta_ = 1.0 / self._lipschitz(X)
        else:
            self.eta_ = float(self.eta)
        w, b = self._parametros_iniciales(X.shape[1])
        (self.coef_, self.intercept_, self.historial_objetivo_,
         self.residuo_final_, self.n_iter_, self.motivo_parada_) = _gradiente_proximal(
            gradiente=lambda w, b: self._gradiente(w, b, X, y),
            objetivo=lambda w, b: self._objetivo(w, b, X, y),
            w=w, b=b, lam1=lam1, eta=self.eta_, tol=self.tol, max_iter=self.max_iter,
            ajustar_intercepto=self.ajustar_intercepto)
        self.objetivo_final_ = self.historial_objetivo_[-1]
        return self


class RegresionLinealGD(_ModeloIterativo):
    """Regresión lineal por descenso por gradiente, con opción L2.

    Minimiza F = J_reg + (λ2/2)||w||². Con λ2 = 0 es OLS; con λ2 > 0 es Ridge.

    Parameters
    ----------
    lam2 : float >= 0
        Intensidad de la penalización L2.
    eta, tol, max_iter, warm_start, ajustar_intercepto : ver `_ModeloIterativo`.
    """

    # Reutilizado de clase: LinearRegressionScratch (gradientes de MSE/2).
    # Mejoras: opción L2, paso η = 1/L automático, validaciones, objetivo
    # completo por iteración, motivo de parada y arranque en caliente opcional.

    def __init__(self, lam2=0.0, eta=None, tol=1e-8, max_iter=100_000, warm_start=False,
                 ajustar_intercepto=True):
        super().__init__(eta, tol, max_iter, warm_start, ajustar_intercepto)
        self.lam2 = lam2

    def _hiperparametros(self):
        return {"lam2": self.lam2, "eta": self.eta, "tol": self.tol, "max_iter": self.max_iter,
                "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_lineal(w, b, X, y, lam2=self.lam2)

    def _gradiente(self, w, b, X, y):
        return gradiente_lineal(w, b, X, y, lam2=self.lam2)

    def _lipschitz(self, X):
        # El Hessiano de la parte suave es X̃ᵀX̃/N + λ2P; su mayor valor propio
        # está acotado por σ_max(X̃)²/N + λ2 (σ_max: mayor valor singular).
        A = _matriz_diseno(X, self.ajustar_intercepto)
        return np.linalg.norm(A, 2) ** 2 / X.shape[0] + self.lam2

    def fit(self, X, y):
        _validar_no_negativo("lam2", self.lam2)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y)
        return self._ajustar(X, y)


class LassoProximal(RegresionLinealGD):
    """Lasso lineal por gradiente proximal (ISTA).

    Minimiza F = J_reg + λ1||w||₁. Cada iteración da un paso de gradiente
    sobre J_reg y luego aplica el umbral suave con umbral ηλ1, de modo que
    los ceros aparecen dentro del algoritmo y no por redondear al final.
    El criterio de parada es el residuo proximal ||θ⁺ - θ||₂/η.

    Parameters
    ----------
    lam1 : float >= 0
        Intensidad de la penalización L1.
    eta, tol, max_iter, warm_start, ajustar_intercepto : ver `_ModeloIterativo`.
    """

    def __init__(self, lam1=1.0, eta=None, tol=1e-8, max_iter=100_000, warm_start=False,
                 ajustar_intercepto=True):
        super().__init__(lam2=0.0, eta=eta, tol=tol, max_iter=max_iter, warm_start=warm_start,
                         ajustar_intercepto=ajustar_intercepto)
        self.lam1 = lam1

    def _hiperparametros(self):
        return {"lam1": self.lam1, "eta": self.eta, "tol": self.tol, "max_iter": self.max_iter,
                "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_lineal(w, b, X, y, lam1=self.lam1)

    def fit(self, X, y):
        _validar_no_negativo("lam1", self.lam1)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y)
        return self._ajustar(X, y, lam1=self.lam1)


class RegresionLogisticaGD(_ModeloIterativo):
    """Regresión logística binaria por descenso por gradiente, con opción L2.

    Minimiza F = J_log + (λ2/2)||w||².

    Parameters
    ----------
    lam2 : float >= 0
        Intensidad de la penalización L2.
    eta, tol, max_iter, warm_start, ajustar_intercepto : ver `_ModeloIterativo`.
    """

    # Reutilizado de clase: LogisticRegressionScratch (gradiente X̃ᵀ(p - y)/N,
    # predict_proba con dos columnas y predict con umbral).
    # Mejoras: opción L2, paso η = 1/L, validaciones, objetivo completo por
    # iteración, motivo de parada y arranque en caliente opcional.

    def __init__(self, lam2=0.0, eta=None, tol=1e-8, max_iter=100_000, warm_start=False,
                 ajustar_intercepto=True):
        super().__init__(eta, tol, max_iter, warm_start, ajustar_intercepto)
        self.lam2 = lam2

    def _hiperparametros(self):
        return {"lam2": self.lam2, "eta": self.eta, "tol": self.tol, "max_iter": self.max_iter,
                "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_logistico(w, b, X, y, lam2=self.lam2)

    def _gradiente(self, w, b, X, y):
        return gradiente_logistico(w, b, X, y, lam2=self.lam2)

    def _lipschitz(self, X):
        # Hessiano = X̃ᵀ diag(p(1-p)) X̃ / N + λ2P y p(1-p) <= 1/4.
        A = _matriz_diseno(X, self.ajustar_intercepto)
        return np.linalg.norm(A, 2) ** 2 / (4 * X.shape[0]) + self.lam2

    def fit(self, X, y):
        _validar_no_negativo("lam2", self.lam2)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y, binaria=True)
        return self._ajustar(X, y)

    def objetivo(self, X, y):
        self._verificar_ajustado()
        X, y = _validar_Xy(X, y, binaria=True)
        return self._objetivo(self.coef_, self.intercept_, X, y)

    def predict_proba(self, X):
        """Probabilidades de las dos clases: columnas [P(Y=0|x), P(Y=1|x)]."""
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, tau=0.5):
        """Etiquetas ŷ = 1{p >= τ}. Cambiar τ no modifica predict_proba."""
        if not (np.isscalar(tau) and 0.0 <= tau <= 1.0):
            raise ValueError(f"tau debe estar en [0, 1]; se recibió {tau!r}.")
        return (self.predict_proba(X)[:, 1] >= tau).astype(int)
