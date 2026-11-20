# Hiper-heurística de Selección para TSP

Proyecto final: dado una instancia nueva de TSP, predecir cuál de tres algoritmos
(2-opt hill climbing, recocido simulado con 2-opt, MST + 2-opt) obtendrá el tour de menor costo.

El reporte completo está en [`report/REPORTE.md`](report/REPORTE.md).

## Resultados principales

| | Baseline (siempre 2-opt HC) | Hiper-heurística (KNN, k = 31) |
|---|---|---|
| Accuracy en prueba | 50.3 % | 55.7 % |
| Regret promedio en prueba | 0.0428 | 0.0357 |

Balance de clases de las 500 instancias con nuestras corridas: 2-opt HC 59.8 %, SA + 2-opt 23.6 %, MST + 2-opt 16.6 %.

## Estructura

| Ruta | Contenido |
|---|---|
| `data/instances/`, `data/train.csv`, `data/test.csv` | Dataset provisto por el curso (sin etiquetas). |
| `data/labeled/` | `train_labeled.csv` y `test_labeled.csv`: el dataset con los costos y `best_algorithm` generados por nuestras corridas. |
| `src/tarea2_algorithms.py` | Algoritmos de Tarea 2, copiados sin cambios de [`abyssace/algoritmia`](https://github.com/abyssace/algoritmia). |
| `src/label_instances.py` | Parte 1: corre los tres algoritmos sobre cada instancia y genera las etiquetas. |
| `src/train_model.py` | Parte 2: selección de modelo, evaluación en prueba, regret, importancia de características y figuras. |
| `src/build_report.py` | Convierte el reporte a PDF (el PDF no se versiona). |
| `results/` | Métricas (`metrics.json`, CSV) y figuras. |
| `report/REPORTE.md` | Reporte. |

## Cómo reproducir

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.label_instances    # Parte 1 (unos 15 s)
python -m src.train_model        # Parte 2
python -m src.build_report       # opcional: report/REPORTE.pdf
```

Todo usa semillas fijas, así que los resultados son reproducibles. Probado con Python 3.12.

## Parámetros de SA usados en el etiquetado

Enfriamiento exponencial, T0 = 0.056125, Tf = 0.0002, 15 000 iteraciones, movimiento 2-opt aleatorio.
Son los valores de Tarea 2 (T0 = 5.6125, Tf = 0.02) divididos entre 100, porque Tarea 2 usó
coordenadas en [0, 100]² y este dataset usa [0, 1]². Detalles en el reporte.
