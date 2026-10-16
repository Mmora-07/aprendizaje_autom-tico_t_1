# Taller 1 — Aprendizaje Automático (2026-2S)
# Mateo Sebastián Mora Montero, Diego Carvajal González

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
| `modelos.py` | — | Código común: OLS, Ridge, Lasso (gradiente proximal) y regresión logística con L2 y pesos de clase opcionales. Los tres notebooks lo importan, así que ningún modelo se implementa dos veces. |
| `01_implementacion_simulacion.ipynb` | 4 y 5 | Pruebas numéricas de los modelos (gradientes, estabilidad, condiciones de optimalidad), comparación con scikit-learn y simulación de sesgo y varianza con ρ = 0 y ρ = 0.9. |
| `02_regresion_airfoil.ipynb` | 6 | Auditoría de los datos, partición 80/20, EDA sobre el conjunto de desarrollo, validación cruzada de 5 pliegues y una única evaluación final en prueba. |
| `03_clasificacion_vuelos.ipynb` | 7 | Partición temporal 60/20/20 con separación de 24 filas, CV expansiva, contraexperimento de fuga, calibración, tabla de umbrales y evaluación final en prueba. |

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
| pandas | 2.3.2 | Tablas, auditoría y exportación de resultados |
| scikit-learn | 1.7.2 | Solo preprocesamiento, particiones, métricas y contrastes |
| matplotlib | 3.10.6 | Figuras |
| ipykernel | 6.30.1 | Kernel de Python para los notebooks |
| nbconvert | 7.16.6 | Ejecución de los notebooks desde la terminal |

> La versión de scikit-learn está fijada porque la equivalencia entre λ₁, λ₂ y los parámetros `alpha` y `C` (problema 4c) depende de la versión instalada.

## Ejecución

Los notebooks se ejecutan **en orden**, cada uno desde un kernel limpio (*Restart Kernel and Run All*):

1. `01_implementacion_simulacion.ipynb`
2. `02_regresion_airfoil.ipynb`: si `data/airfoil_self_noise.dat` no existe, lo descarga.
3. `03_clasificacion_vuelos.ipynb`

Para ejecutarlos todos desde la terminal:

```bash
python -m nbconvert --to notebook --execute --inplace 01_implementacion_simulacion.ipynb 02_regresion_airfoil.ipynb 03_clasificacion_vuelos.ipynb
```

| Notebook | Tiempo aproximado |
|---|---|
| 01 | ~30 s |
| 02 | ~15 s |
| 03 | ~1 min |

Tiempos medidos en: AMD Ryzen 7 5700U, 24 GB de RAM, Windows 11, Python 3.10.0.

---

## Datos

| Caso | Procedencia |
|---|---|
| Airfoil Self-Noise | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise). Lo descarga el notebook 02 en `data/airfoil_self_noise.dat`, con la misma fuente que se usó en el notebook de regresión de clase. |
| Vuelos sintéticos | `generar_vuelos(n=9600, seed=2026)`, tomada del notebook de clasificación de clase sin cambiar su mecanismo. |
| Simulación controlada | Generador del problema 5, definido en el notebook 01. |

## Semillas

- **Semilla maestra: 2026.** Se usa para la partición 80/20 de Airfoil, los pliegues con mezcla y el generador de vuelos.
- **Problema 5:** las subsemillas de los diseños (X, X\*) y de los ruidos ε⁽ʳ⁾ se derivan de la maestra y quedan registradas en `resultados/p5_semillas.csv`.

## Resultados

La carpeta `resultados/` contiene los archivos que pide el enunciado:

| Problema | Archivos |
|---|---|
| 4 | `p4_pruebas.csv` (tabla de pruebas), `p4_historial_convergencia.csv`, `p4_comparacion_sklearn.csv` (objetivos equivalentes) |
| 5 | `p5_sesgo_varianza.csv` (tabla por escenario, modelo y λ, con métricas de soporte de Lasso), `p5_frecuencias_seleccion.csv`, `p5_semillas.csv` |
| 6 | `p6_particiones.csv` (manifiesto por `id`), `p6_auditoria.csv`, `p6_validacion_cv.csv` (resultados por pliegue), `p6_seleccion.json` (registro de la selección), `p6_evaluacion_final.csv`, `p6_predicciones_prueba.csv` |
| 7 | `p7_particiones.csv` (conjunto de cada `id_vuelo`), `p7_validacion_cv.csv` (resultados por pliegue), `p7_seleccion.json` (historial de selección), `p7_tabla_umbrales.csv`, `p7_manifiesto_politica.json`, `p7_evaluacion_final.csv`, `p7_predicciones_prueba.csv` (id, fecha, probabilidad y decisión por política) |

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
| `sigmoid` (`modelos.py`) | Notebook de regresión lineal, logística y regularización | Se conserva la forma estable `exp(-logaddexp(0, -z))`; solo se documenta por qué no desborda |
| `perdida_logistica` (`modelos.py`) | Notebook de regresión lineal, logística y regularización | Se calcula como `log(1 + e^z) - y z` (la forma del problema 3a) y admite pesos por observación, normalizados por Σa |
| `_gradiente_proximal` (`modelos.py`) | Notebook de regresión lineal, logística y regularización (`fit_regularized`) | Bucle común a todos los modelos iterativos; registra el objetivo completo por iteración, el residuo ‖θ⁺ − θ‖/η y el motivo de parada, y detecta divergencia |
| `RegresionLinealGD` (`modelos.py`) | Notebook de regresión lineal, logística y regularización (`LinearRegressionScratch`) | Opción L2, paso η = 1/L automático, validación de entradas e hiperparámetros, arranque en caliente opcional y opción sin intercepto |
| `RegresionLogisticaGD` (`modelos.py`) | Notebook de regresión lineal, logística y regularización (`LogisticRegressionScratch`) | Opción L2, pesos de clase, paso η = 1/L, validaciones, arranque en caliente opcional y opción sin intercepto |
| Descarga y lectura de Airfoil (notebook 02) | Notebook de regresión lineal, logística y regularización | Se agrega un `id` estable por número de línea y el hash SHA-256 del archivo crudo |
| `generar_vuelos` (notebook 03) | Notebook de clasificación de vuelos | Copiada sin cambios (el enunciado exige no modificarla); fuera de la función se agregan `id_vuelo` y la etiqueta |
| Partición temporal (notebook 03) | Notebook de clasificación de vuelos (`separar_por_tiempo`) | Mismos cortes y exclusión de 24 filas; se registra el conjunto de cada `id_vuelo`, incluidas las filas excluidas, y los rangos de cada período |
| Preprocesador (notebook 03) | Notebook de clasificación de vuelos (`construir_pipeline`) | Mismas ramas numérica y categórica; el estimador final es la logística propia en lugar de la de scikit-learn |

`RegresionOLS`, `RegresionRidge`, `LassoProximal`, las pruebas del problema 4, la simulación del problema 5 y el resto de los análisis son implementaciones nuevas.

**Uso de IA generativa:** _por completar: Redacción y escritura en Latex, revisión teórica de resultados del informe, organización de código, comentarios, docstrings, redacción general y creación de tests. Todo el contenido generado/modificado por IA fue revisado posteriormente por nosotros.
