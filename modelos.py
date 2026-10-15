"""Modelos lineales implementados desde cero con NumPy (Taller 1).

Los tres notebooks importan los modelos desde este archivo, así cada modelo
se programa una sola vez y se reutiliza en todos los experimentos.

Convención de las pérdidas (la misma del enunciado):

    z = X w + b                                   (score lineal)
    J_reg(w, b) = (1 / 2N) ||z - y||²             (regresión)
    J_log(w, b) = -(1/N) Σ [y log p + (1 - y) log(1 - p)],  p = σ(z)   (clasificación)
    F(w, b)     = J(w, b) + λ1 ||w||₁ + (λ2 / 2) ||w||²                (objetivo completo)

El intercepto b nunca se penaliza.

Contenido
---------
- Funciones base: sigmoide, pérdida logística, objetivos, gradientes y umbral suave.
- RegresionOLS:          mínimos cuadrados con np.linalg.lstsq.
- RegresionRidge:        Ridge resolviendo un sistema lineal con np.linalg.solve.
- RegresionLinealGD:     descenso por gradiente para regresión lineal (opción L2).
- LassoProximal:         gradiente proximal para Lasso lineal.
- RegresionLogisticaGD:  descenso por gradiente para regresión logística
                         (opción L2 y pesos por observación).

Todos los modelos tienen la misma interfaz: `fit` entrena, `predict` predice
y `objetivo` evalúa la función de costo con los parámetros aprendidos.
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
        Scores reales (pueden ser muy grandes, por ejemplo ±1000).

    Returns
    -------
    numpy.ndarray
        Probabilidades en [0, 1] con la misma forma que z.
    """
    # Reutilizado de clase: la expresión exp(-logaddexp(0, -z)).
    # La fórmula directa 1 / (1 + exp(-z)) desborda cuando z = -1000, porque
    # exp(1000) no cabe en un float. Aquí usamos σ(z) = exp(-log(1 + e^{-z})):
    # np.logaddexp(0, -z) calcula log(1 + e^{-z}) sin formar e^{-z}, y el exp
    # final, en el peor caso, da 0, que sigue siendo una probabilidad válida.
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def _escala_pesos(pesos, n):
    """Factor que multiplica la pérdida de cada observación.

    Sin pesos, cada observación pesa 1/N (promedio simple). Con pesos a_n,
    cada una pesa a_n / Σ a_n (promedio ponderado). En ambos casos los
    factores suman 1.
    """
    if pesos is None:
        return np.full(n, 1.0 / n)
    return pesos / np.sum(pesos)


def perdida_logistica(y, z, pesos=None):
    """Pérdida logística media J_log calculada directamente desde los scores.

    Usa la forma equivalente  ℓ(y, z) = log(1 + e^{z}) - y z  (problema 3a).
    Como no pasa por log(p), nunca evalúa log(0) y no hace falta recortar
    las probabilidades.

    Parameters
    ----------
    y : numpy.ndarray de forma (N,)
        Etiquetas 0 o 1.
    z : numpy.ndarray de forma (N,)
        Scores lineales z = Xw + b.
    pesos : numpy.ndarray de forma (N,) o None
        Pesos a_n > 0 por observación. Con pesos la pérdida es el promedio
        ponderado Σ a_n ℓ_n / Σ a_n; sin pesos, la media simple.

    Returns
    -------
    float
        Pérdida media (ponderada si se dan pesos).
    """
    # Reutilizado de clase: calcular la pérdida desde z con np.logaddexp.
    # Mejoras: se usa la forma log(1 + e^z) - y z, la misma que se deriva en el
    # problema 3a, en lugar de elegir el signo con np.where; opción de pesos.
    perdidas = np.logaddexp(0.0, z) - y * z          # pérdida de cada observación
    return float(np.sum(_escala_pesos(pesos, len(y)) * perdidas))


def objetivo_lineal(w, b, X, y, lam1=0.0, lam2=0.0):
    """Objetivo completo de regresión F = J_reg + λ1 ||w||₁ + (λ2/2) ||w||²."""
    residuo = X @ w + b - y
    error_cuadratico = residuo @ residuo / (2 * len(y))     # J_reg
    penalizacion_l1 = lam1 * np.sum(np.abs(w))
    penalizacion_l2 = lam2 / 2 * (w @ w)
    return float(error_cuadratico + penalizacion_l1 + penalizacion_l2)


def gradiente_lineal(w, b, X, y, lam2=0.0):
    """Gradiente de la parte derivable del objetivo lineal, J_reg + (λ2/2)||w||².

    Returns
    -------
    grad_w : numpy.ndarray de forma (D,)
        (1/N) Xᵀ(z - y) + λ2 w
    grad_b : float
        (1/N) Σ (z_n - y_n)   (el intercepto no lleva penalización)
    """
    error = X @ w + b - y                     # predicción menos observación
    grad_w = X.T @ error / len(y) + lam2 * w
    grad_b = float(np.mean(error))
    return grad_w, grad_b


def objetivo_logistico(w, b, X, y, lam2=0.0, pesos=None):
    """Objetivo completo de clasificación F = J_log + (λ2/2) ||w||².

    Con pesos, J_log es el promedio ponderado Σ a_n ℓ_n / Σ a_n. La
    penalización (λ2/2)||w||² es la misma con o sin pesos.
    """
    return perdida_logistica(y, X @ w + b, pesos) + lam2 / 2 * float(w @ w)


def gradiente_logistico(w, b, X, y, lam2=0.0, pesos=None):
    """Gradiente de J_log + (λ2/2)||w||².

    Tiene la misma forma que en regresión lineal, cambiando la predicción z
    por la probabilidad p = σ(z).

    Returns
    -------
    grad_w : numpy.ndarray de forma (D,)
        Σ c_n (p_n - y_n) x_n + λ2 w,  con c_n = 1/N o c_n = a_n / Σ a_n
    grad_b : float
        Σ c_n (p_n - y_n)   (el intercepto no lleva penalización)
    """
    error = _escala_pesos(pesos, len(y)) * (sigmoid(X @ w + b) - y)
    grad_w = X.T @ error + lam2 * w
    grad_b = float(np.sum(error))
    return grad_w, grad_b


def umbral_suave(u, t):
    """Operador de umbral suave S_t(u) = signo(u) · max(|u| - t, 0).

    Es la solución de Lasso en una dimensión (problema 2a): si |u| <= t el
    resultado es exactamente 0; si no, u se acerca t unidades hacia cero.
    """
    return np.sign(u) * np.maximum(np.abs(u) - t, 0.0)


# =============================================================================
# Validaciones
# =============================================================================

def _validar_X(X, n_caracteristicas=None):
    """Convierte X a arreglo float y verifica que sea 2D, no vacío y finito.

    Si se da `n_caracteristicas`, también verifica que X tenga ese número de
    columnas (sirve para `predict`, que debe recibir las mismas columnas que `fit`).
    """
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
    """Lanza ValueError si `valor` no es un número finito >= 0 (por ejemplo, λ)."""
    if not (np.isscalar(valor) and np.isfinite(valor) and valor >= 0):
        raise ValueError(f"{nombre} debe ser un número finito >= 0; se recibió {valor!r}.")


def _validar_positivo(nombre, valor):
    """Lanza ValueError si `valor` no es un número finito > 0 (por ejemplo, η o tol)."""
    if not (np.isscalar(valor) and np.isfinite(valor) and valor > 0):
        raise ValueError(f"{nombre} debe ser un número finito > 0; se recibió {valor!r}.")


def _validar_entero_positivo(nombre, valor):
    """Lanza ValueError si `valor` no es un entero > 0 (por ejemplo, max_iter)."""
    if not (isinstance(valor, (int, np.integer)) and valor > 0):
        raise ValueError(f"{nombre} debe ser un entero > 0; se recibió {valor!r}.")


def _matriz_aumentada(X):
    """X̃ = [1  X]: agrega a la izquierda la columna de unos del intercepto."""
    return np.column_stack([np.ones(X.shape[0]), X])


def _matriz_diseno(X, ajustar_intercepto):
    """Matriz con la que se ajusta: X̃ si se estima b, o X si b se fija en 0."""
    if ajustar_intercepto:
        return _matriz_aumentada(X)
    return X


# =============================================================================
# Clase base
# =============================================================================

class _ModeloLineal:
    """Comportamiento común a todos los modelos: predecir, evaluar y guardar.

    Atributos que deja `fit` en todos los modelos
    ---------------------------------------------
    coef_ : numpy.ndarray de forma (D,)
        Coeficientes w.
    intercept_ : float
        Intercepto b.
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
        """Lanza un error claro si se intenta predecir antes de entrenar."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("El modelo no está ajustado: llame primero a fit.")

    def decision_function(self, X):
        """Scores lineales z = Xw + b."""
        self._verificar_ajustado()
        X = _validar_X(X, len(self.coef_))
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predicciones continuas Xw + b (en regresión coinciden con el score)."""
        return self.decision_function(X)

    def objetivo(self, X, y):
        """Evalúa el objetivo completo F con los parámetros aprendidos."""
        self._verificar_ajustado()
        X, y = _validar_Xy(X, y)
        return self._objetivo(self.coef_, self.intercept_, X, y)

    def _guardar_theta(self, theta):
        """Separa θ en intercepto y coeficientes según `ajustar_intercepto`.

        Con intercepto, θ = (b, w): el primer elemento es b. Sin intercepto,
        θ = w y b queda en 0.
        """
        if self.ajustar_intercepto:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

    def guardar_parametros(self, ruta):
        """Guarda coeficientes, intercepto e hiperparámetros en un archivo JSON.

        JSON escribe cada float con la representación más corta que se lee de
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

    lstsq usa la descomposición SVD de X̃, así que también funciona si X̃ no
    tiene rango completo (por ejemplo, con una columna duplicada). En ese caso
    hay infinitas soluciones y lstsq devuelve la de norma mínima; todas dan
    las mismas predicciones X̃θ.

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
        self.n_iter_ = 0
        self.motivo_parada_ = "solucion directa"
        return self


class RegresionRidge(_ModeloLineal):
    """Ridge con intercepto no penalizado, resolviendo las ecuaciones normales.

    Minimiza F = (1/2N)||X̃θ - y||² + (λ2/2) θᵀPθ, con P = diag(0, 1, ..., 1).
    El 0 de P deja sin penalizar al intercepto. Igualando el gradiente a cero
    (problema 1a) se obtiene el sistema lineal

        (X̃ᵀX̃ + N λ2 P) θ = X̃ᵀ y

    Con λ2 > 0 la matriz del sistema es invertible aunque X tenga columnas
    dependientes (problema 1b), así que np.linalg.solve encuentra la única
    solución sin calcular ninguna inversa explícita.

    Parameters
    ----------
    lam2 : float > 0
        Intensidad de la penalización L2.
    ajustar_intercepto : bool
        Si es False se resuelve (XᵀX + Nλ2 I) w = Xᵀy con b = 0.
    """

    def __init__(self, lam2=1.0, ajustar_intercepto=True):
        self.lam2 = lam2
        self.ajustar_intercepto = ajustar_intercepto

    def _hiperparametros(self):
        return {"lam2": self.lam2, "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_lineal(w, b, X, y, lam2=self.lam2)

    def fit(self, X, y):
        """Ajusta θ = (b, w) resolviendo (X̃ᵀX̃ + Nλ2P)θ = X̃ᵀy."""
        _validar_positivo("lam2", self.lam2)
        X, y = _validar_Xy(X, y)
        A = _matriz_diseno(X, self.ajustar_intercepto)

        # P es la identidad, salvo un 0 en la posición del intercepto.
        P = np.eye(A.shape[1])
        if self.ajustar_intercepto:
            P[0, 0] = 0.0

        matriz = A.T @ A + len(y) * self.lam2 * P
        theta = np.linalg.solve(matriz, A.T @ y)
        self._guardar_theta(theta)
        self.objetivo_final_ = self._objetivo(self.coef_, self.intercept_, X, y)
        self.n_iter_ = 0
        self.motivo_parada_ = "solucion directa"
        return self


# =============================================================================
# Métodos iterativos: descenso por gradiente y gradiente proximal
# =============================================================================

def _gradiente_proximal(gradiente, objetivo, w, b, lam1, eta, tol, max_iter,
                        ajustar_intercepto=True):
    """Bucle común de todos los modelos iterativos.

    Minimiza f(w, b) + λ1||w||₁, donde f es la parte derivable del objetivo.
    En cada iteración:

        w⁺ = S_{ηλ1}(w - η ∇_w f)      paso de gradiente y luego umbral suave
        b⁺ = b - η ∂f/∂b               paso de gradiente normal (b no se penaliza)

    Si λ1 = 0, el umbral suave no cambia nada y el bucle es descenso por
    gradiente ordinario. Si ajustar_intercepto es False, b no se actualiza.

    Criterio de parada: ||θ⁺ - θ||₂ / η < tol, con θ = (b, w). Para λ1 = 0
    esa cantidad es la norma del gradiente. Para λ1 > 0 es el "residuo
    proximal", que vale cero solo en un minimizador de F; no basta con que el
    gradiente de la parte suave sea cero, porque la parte L1 no es derivable.

    Parameters
    ----------
    gradiente : función (w, b) -> (grad_w, grad_b)
        Gradiente de la parte derivable f.
    objetivo : función (w, b) -> float
        Objetivo completo F, solo para registrar el historial.
    w, b : parámetros iniciales.
    lam1, eta, tol, max_iter : penalización L1, paso, tolerancia y máximo de iteraciones.
    ajustar_intercepto : bool
        Si es False, b se mantiene fijo.

    Returns
    -------
    w, b : parámetros finales
    historial : list[float]
        Objetivo completo F en el punto inicial y después de cada iteración.
    residuo : float
        Último valor de ||θ⁺ - θ||₂ / η.
    n_iter : int
        Iteraciones realizadas.
    motivo : str
        "convergencia", "max_iter" o "divergencia".
    """
    # Reutilizado de clase: la actualización de fit_regularized (paso de gradiente
    # + umbral suave en w, paso normal en b) y su criterio ||Δθ|| / η.
    # Mejoras: se registra el objetivo completo en cada iteración, el residuo
    # final y el motivo de terminación, y se detecta la divergencia.
    historial = [objetivo(w, b)]
    residuo = np.inf
    motivo = "max_iter"            # si el bucle se agota sin converger, queda este motivo

    for n_iter in range(1, max_iter + 1):
        grad_w, grad_b = gradiente(w, b)

        # Paso de gradiente en w y, enseguida, umbral suave (pone ceros exactos).
        w_nuevo = umbral_suave(w - eta * grad_w, eta * lam1)
        # El intercepto solo da el paso de gradiente, sin umbral.
        if ajustar_intercepto:
            b_nuevo = b - eta * grad_b
        else:
            b_nuevo = b

        # Tamaño del cambio de todos los parámetros (incluido b), dividido por η.
        cambio = np.sum((w_nuevo - w) ** 2) + (b_nuevo - b) ** 2
        residuo = np.sqrt(cambio) / eta

        w, b = w_nuevo, b_nuevo
        historial.append(objetivo(w, b))

        if not np.isfinite(historial[-1]):     # el objetivo explotó: paso demasiado grande
            motivo = "divergencia"
            break
        if residuo < tol:                      # los parámetros ya casi no cambian
            motivo = "convergencia"
            break

    return w, b, historial, residuo, n_iter, motivo


class _ModeloIterativo(_ModeloLineal):
    """Parte común de los modelos que se ajustan iterando.

    Parameters
    ----------
    eta : float > 0 o None
        Tamaño de paso. Si es None se usa η = 1/L, donde L es una cota de la
        constante de Lipschitz del gradiente (qué tan rápido puede cambiar el
        gradiente; ver `_lipschitz`). Con ese paso el objetivo baja en cada
        iteración.
    tol : float > 0
        Tolerancia del criterio de parada ||θ⁺ - θ||₂ / η < tol.
    max_iter : int > 0
        Máximo de iteraciones.
    warm_start : bool
        Si es False (por defecto), cada llamada a fit empieza en w = 0, b = 0.
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
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.ajustar_intercepto = ajustar_intercepto

    def _validar_hiperparametros(self):
        """Revisa que η (si se dio), tol y max_iter tengan valores válidos."""
        if self.eta is not None:
            _validar_positivo("eta", self.eta)
        _validar_positivo("tol", self.tol)
        _validar_entero_positivo("max_iter", self.max_iter)

    def _parametros_iniciales(self, n_caracteristicas):
        """Punto de arranque: w = 0, b = 0, salvo arranque en caliente declarado."""
        ya_ajustado = hasattr(self, "coef_") and len(self.coef_) == n_caracteristicas
        if self.warm_start and ya_ajustado:
            return self.coef_.copy(), float(self.intercept_)
        return np.zeros(n_caracteristicas), 0.0

    def _ajustar(self, X, y, lam1=0.0):
        """Elige el paso, ejecuta el bucle iterativo y guarda los resultados."""
        if self.eta is None:
            self.eta_ = 1.0 / self._lipschitz(X)
        else:
            self.eta_ = float(self.eta)

        w_inicial, b_inicial = self._parametros_iniciales(X.shape[1])

        # Funciones que el bucle llama en cada iteración; "fijan" X e y.
        def gradiente(w, b):
            return self._gradiente(w, b, X, y)

        def objetivo(w, b):
            return self._objetivo(w, b, X, y)

        w, b, historial, residuo, n_iter, motivo = _gradiente_proximal(
            gradiente, objetivo, w_inicial, b_inicial, lam1=lam1, eta=self.eta_,
            tol=self.tol, max_iter=self.max_iter, ajustar_intercepto=self.ajustar_intercepto)

        self.coef_ = w
        self.intercept_ = b
        self.historial_objetivo_ = historial
        self.residuo_final_ = residuo
        self.n_iter_ = n_iter
        self.motivo_parada_ = motivo
        self.objetivo_final_ = historial[-1]
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
        """Cota L de la curvatura del objetivo lineal: σ_max(X̃)²/N + λ2.

        El Hessiano de la parte suave es X̃ᵀX̃/N + λ2P. Su mayor valor propio es
        a lo sumo σ_max(X̃)²/N + λ2, donde σ_max es el mayor valor singular.
        """
        A = _matriz_diseno(X, self.ajustar_intercepto)
        return np.linalg.norm(A, 2) ** 2 / X.shape[0] + self.lam2

    def fit(self, X, y):
        """Valida las entradas y ajusta el modelo por descenso por gradiente."""
        _validar_no_negativo("lam2", self.lam2)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y)
        return self._ajustar(X, y)


class LassoProximal(RegresionLinealGD):
    """Lasso lineal por gradiente proximal (ISTA).

    Minimiza F = J_reg + λ1||w||₁. Cada iteración da un paso de gradiente
    sobre J_reg y luego aplica el umbral suave con umbral ηλ1. Así los ceros
    aparecen dentro del algoritmo, no por redondear coeficientes al final.
    El criterio de parada es el residuo proximal ||θ⁺ - θ||₂/η.

    Hereda de RegresionLinealGD (con λ2 = 0) el gradiente y la constante L,
    porque la parte derivable es la misma J_reg.

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
        """Valida las entradas y ajusta Lasso por gradiente proximal."""
        _validar_no_negativo("lam1", self.lam1)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y)
        return self._ajustar(X, y, lam1=self.lam1)


class RegresionLogisticaGD(_ModeloIterativo):
    """Regresión logística binaria por descenso por gradiente, con opción L2.

    Minimiza F = J_log + (λ2/2)||w||². Si `fit` recibe pesos a_n, J_log es
    el promedio ponderado Σ a_n ℓ_n / Σ a_n (ponderación de clases, problema 7)
    y la penalización se mantiene igual.

    Parameters
    ----------
    lam2 : float >= 0
        Intensidad de la penalización L2.
    eta, tol, max_iter, warm_start, ajustar_intercepto : ver `_ModeloIterativo`.
    """

    # Reutilizado de clase: LogisticRegressionScratch (gradiente X̃ᵀ(p - y)/N,
    # predict_proba con dos columnas y predict con umbral).
    # Mejoras: opción L2, pesos por observación, paso η = 1/L, validaciones,
    # objetivo completo por iteración, motivo de parada y arranque en caliente.

    def __init__(self, lam2=0.0, eta=None, tol=1e-8, max_iter=100_000, warm_start=False,
                 ajustar_intercepto=True):
        super().__init__(eta, tol, max_iter, warm_start, ajustar_intercepto)
        self.lam2 = lam2
        self._pesos = None   # pesos del ajuste en curso; los fija fit

    def _hiperparametros(self):
        return {"lam2": self.lam2, "eta": self.eta, "tol": self.tol, "max_iter": self.max_iter,
                "ajustar_intercepto": self.ajustar_intercepto}

    def _objetivo(self, w, b, X, y):
        return objetivo_logistico(w, b, X, y, lam2=self.lam2, pesos=self._pesos)

    def _gradiente(self, w, b, X, y):
        return gradiente_logistico(w, b, X, y, lam2=self.lam2, pesos=self._pesos)

    def _lipschitz(self, X):
        """Cota L de la curvatura del objetivo logístico.

        El Hessiano es X̃ᵀ diag(c_n p_n(1-p_n)) X̃ + λ2P, y p(1-p) <= 1/4 siempre.
        Por eso L = σ_max(diag(√c) X̃)² / 4 + λ2, con c_n = 1/N sin pesos.
        """
        A = _matriz_diseno(X, self.ajustar_intercepto)
        raiz_c = np.sqrt(_escala_pesos(self._pesos, X.shape[0]))
        A_escalada = raiz_c[:, None] * A          # multiplica cada fila n por √c_n
        return np.linalg.norm(A_escalada, 2) ** 2 / 4 + self.lam2

    def fit(self, X, y, pesos=None):
        """Valida las entradas y ajusta la logística.

        Parameters
        ----------
        X : numpy.ndarray de forma (N, D)
        y : numpy.ndarray de forma (N,) con etiquetas 0/1
        pesos : numpy.ndarray de forma (N,) o None
            Pesos a_n > 0 por observación (por ejemplo, pesos de clase).
        """
        _validar_no_negativo("lam2", self.lam2)
        self._validar_hiperparametros()
        X, y = _validar_Xy(X, y, binaria=True)
        if pesos is not None:
            pesos = np.asarray(pesos, dtype=float)
            forma_ok = pesos.shape == y.shape
            valores_ok = np.all(np.isfinite(pesos)) and np.all(pesos > 0)
            if not (forma_ok and valores_ok):
                raise ValueError("pesos debe ser un vector de la longitud de y con valores finitos > 0.")
        self._pesos = pesos
        return self._ajustar(X, y)

    def objetivo(self, X, y, pesos=None):
        """Evalúa F en (X, y), con pesos opcionales."""
        self._verificar_ajustado()
        X, y = _validar_Xy(X, y, binaria=True)
        return objetivo_logistico(self.coef_, self.intercept_, X, y, lam2=self.lam2, pesos=pesos)

    def predict_proba(self, X):
        """Probabilidades de las dos clases: columnas [P(Y=0|x), P(Y=1|x)]."""
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, tau=0.5):
        """Etiquetas ŷ = 1{p >= τ}. Cambiar τ no modifica predict_proba."""
        if not (np.isscalar(tau) and 0.0 <= tau <= 1.0):
            raise ValueError(f"tau debe estar en [0, 1]; se recibió {tau!r}.")
        return (self.predict_proba(X)[:, 1] >= tau).astype(int)
