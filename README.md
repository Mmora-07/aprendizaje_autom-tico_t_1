# Taller 1 — Aprendizaje Automático (2026-2S)

**Datos, modelos lineales y decisiones bajo incertidumbre**

Este repositorio contiene la parte práctica del Taller 1 (problemas 4 a 7, 70 puntos). Los modelos lineales están implementados desde cero con NumPy, se verifican numéricamente y se usan en tres experimentos:

- una simulación de sesgo, varianza y selección de variables;
- una regresión reproducible sobre el conjunto **Airfoil Self-Noise**;
- una clasificación temporal de retrasos de vuelos, con una política de alertas.

La parte teórica (problemas 1 a 3) se responde en el informe PDF. Algunos de sus resultados se contrastan aquí: la estabilidad de Ridge (problema 1) en el problema 5 y el umbral de costo (problema 3) en el problema 7.

---

## Contenido

| Archivo | Problema | Qué se hace |
|---|---|---|
| `modelos.py` | — | Código común: OLS, Ridge, Lasso (gradiente proximal) y regresión logística con L2. Los tres notebooks lo importan, así que ningún modelo se implementa dos veces. |
| `01_implementacion_simulacion.ipynb` | 4 y 5 | Pruebas numéricas de los modelos (gradientes, estabilidad, condiciones de optimalidad), comparación con scikit-learn y simulación de sesgo y varianza con ρ = 0 y ρ = 0.9. |
| `02_regresion_airfoil.ipynb` | 6 | Auditoría de los datos, partición 80/20, EDA sobre el conjunto de desarrollo, validación cruzada de 5 pliegues y una única evaluación final en prueba. |
| `03_clasificacion_vuelos.ipynb` | 7 | Partición temporal 60/20/20 con separación de 24 filas, CV expansiva, contraexperimento de fuga, calibración, tabla de umbrales y evaluación final en prueba. |

Los problemas 4 y 5 comparten notebook porque los dos usan datos sintéticos y las mismas implementaciones. Por eso son tres notebooks para cuatro problemas.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── modelos.py                          # Modelos desde cero (código común)
├── 01_implementacion_simulacion.ipynb  # Problemas 4 y 5
├── 02_regresion_airfoil.ipynb          # Problema 6
├── 03_clasificacion_vuelos.ipynb       # Problema 7
├── data/
│   └── airfoil_self_noise.dat          # Datos crudos (no se modifican)
└── resultados/                         # Salidas estructuradas (CSV / JSON), prefijo por problema
    └── parametros/                     # Parámetros de modelos guardados (JSON)
```

---

## Instalación

Requiere **Python 3.10**. Desde la raíz del repositorio:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencias

| Paquete | Versión | Uso |
|---|---|---|
| numpy | 2.2.6 | Implementación de los modelos (`solve`, `lstsq`) |
| pandas | 2.2.3 | Tablas, auditoría y exportación de resultados |
| scikit-learn | 1.6.1 | Solo preprocesamiento, particiones, métricas y contrastes |
| matplotlib | 3.10.3 | Figuras |
| jupyter | 1.1.1 | Ejecución de los notebooks |

> La versión de scikit-learn está fijada porque la equivalencia entre λ₁, λ₂ y los parámetros `alpha` y `C` (problema 4c) depende de la versión instalada.

## Ejecución

Los notebooks se ejecutan **en orden**, cada uno desde un kernel limpio (*Restart Kernel and Run All*):

1. `01_implementacion_simulacion.ipynb`
2. `02_regresion_airfoil.ipynb`: si `data/airfoil_self_noise.dat` no existe, lo descarga.
3. `03_clasificacion_vuelos.ipynb`

Para ejecutarlos todos desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_implementacion_simulacion.ipynb 02_regresion_airfoil.ipynb 03_clasificacion_vuelos.ipynb
```

| Notebook | Tiempo aproximado |
|---|---|
| 01 | _por medir_ |
| 02 | _por medir_ |
| 03 | _por medir_ |

Tiempos medidos en: _por completar (procesador, RAM, sistema operativo)_.

---

## Datos

| Caso | Procedencia |
|---|---|
| Airfoil Self-Noise | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise). Lo descarga el notebook 02 en `data/airfoil_self_noise.dat`, con la misma fuente del notebook de regresión de clase. |
| Vuelos sintéticos | `generar_vuelos(n=9600, seed=2026)`, tomada del notebook de clasificación de clase sin cambiar su mecanismo. |
| Simulación controlada | Generador del problema 5, definido en el notebook 01. |

## Semillas

- **Semilla maestra: 2026.** Se usa para la partición 80/20 de Airfoil, los pliegues con mezcla y el generador de vuelos.
- **Problema 5:** las subsemillas de los diseños (X, X\*) y de los ruidos ε⁽ʳ⁾ se derivan de la maestra y quedan registradas en `resultados/`.

## Resultados

La carpeta `resultados/` contiene los archivos que pide el enunciado:

| Problema | Archivos |
|---|---|
| 4 | `p4_pruebas.csv` (tabla de pruebas), `p4_historial_convergencia.csv`, `p4_comparacion_sklearn.csv` (objetivos equivalentes) |
| 5 | Sesgo², varianza y error por escenario, modelo y λ; frecuencias de selección de Lasso |
| 6 | Particiones por `id`, auditoría, resultados por pliegue, registro de selección, predicciones finales |
| 7 | Rangos e IDs temporales, resultados por pliegue, tabla de umbrales, manifiesto de la política, predicciones finales (con fecha y decisión) |

---

## Criterios metodológicos

- El conjunto de prueba **no interviene** en la elección de variables, hiperparámetros, algoritmos ni umbrales, y se evalúa una sola vez.
- El preprocesamiento (escalado, imputación, codificación) se ajusta **solo con los datos de entrenamiento** de cada partición o pliegue.
- Todos los modelos se comparan con las **mismas particiones y observaciones**.
- Los algoritmos se implementan desde cero: no se usan inversas explícitas, diferenciación automática, optimizadores genéricos ni estimadores de biblioteca en su lugar. scikit-learn solo sirve para comparar y para utilidades.

---

## Atribución y uso de IA

**Código adaptado de clase**

En el código, cada función o celda adaptada lleva un comentario que indica qué se reutilizó y qué se cambió:

```python
# Reutilizado de clase: <qué parte se tomó>
# Mejoras: <qué se modificó o agregó y por qué>
```

| Función o celda | Fuente | Cambios realizados |
|---|---|---|
| _por completar_ | _notebook de clase_ | _por completar_ |

**Responsabilidades:** _por completar_.

**Uso de IA generativa:** _por completar: para qué se usó, qué se verificó y qué se corrigió._
