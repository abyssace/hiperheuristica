"""Parte 1: corre los tres algoritmos de Tarea 2 sobre cada instancia y etiqueta.

Uso:
    python -m src.label_instances

Lee data/train.csv y data/test.csv (sin etiquetas), corre sobre cada instancia
2-opt hill climbing, SA con movimiento 2-opt y MST + 2-opt (implementaciones de
Tarea 2, copiadas sin cambios en src/tarea2_algorithms.py) y escribe
data/labeled/train_labeled.csv y data/labeled/test_labeled.csv.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import random
import time
from pathlib import Path

from .tarea2_algorithms import (
    mst_preorder_tour, random_tour, simulated_annealing, tour_cost, two_opt,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "labeled"

# --- Parámetros de SA (son parte del etiquetado) -----------------------------
# Tarea 2 trabajó con coordenadas en [0, 100]^2; este dataset usa [0, 1]^2.
# Las temperaturas se dividen entre COORD_SCALE para conservar exactamente la
# misma probabilidad de aceptación exp(-delta / T) que en Tarea 2 (equivale a
# multiplicar las coordenadas por 100 y correr con las temperaturas originales).
COORD_SCALE = 100.0
SA_START_TEMP_TAREA2 = 5.612504500232145   # nivel "low" seleccionado en el DOE de Tarea 2
SA_END_TEMP_TAREA2 = 0.02                  # temperatura final del DOE de Tarea 2
SA_START_TEMP = SA_START_TEMP_TAREA2 / COORD_SCALE
SA_END_TEMP = SA_END_TEMP_TAREA2 / COORD_SCALE
SA_SCHEDULE = "exponential"                # T_k = T0 * (Tf / T0) ** (k / (K - 1))
SA_ITERATIONS = 15_000                     # presupuesto del experimento primario del DOE

SEED = 20261119
# Dos costos se consideran empatados si difieren en menos de TIE_TOL; el empate
# se resuelve por este orden fijo (el mismo orden de las columnas de costo).
TIE_TOL = 1e-9
ALGORITHMS = ("2opt_hill_climbing", "sa_2opt", "mst_2opt")


def load_points(instance_id: str) -> list[tuple[float, float]]:
    with (DATA / "instances" / f"{instance_id}.csv").open(newline="") as file:
        return [(float(row["x"]), float(row["y"])) for row in csv.DictReader(file)]


def best_algorithm(costs: dict[str, float]) -> tuple[str, int]:
    """Algoritmo de menor costo y cuántos algoritmos empatan en ese mínimo."""
    minimum = min(costs.values())
    tied = [name for name in ALGORITHMS if costs[name] - minimum <= TIE_TOL]
    return tied[0], len(tied)


def run_instance(instance_id: str) -> dict[str, object]:
    points = load_points(instance_id)
    n = len(points)
    number = int(instance_id.split("_")[1])
    rng = random.Random(SEED + number)
    # Igual que en Tarea 2: búsqueda local y SA parten del mismo tour aleatorio.
    initial = random_tour(n, rng)

    started = time.perf_counter()
    hill_tour, _, _ = two_opt(initial, points)
    t_hill = time.perf_counter() - started

    started = time.perf_counter()
    if n >= 2:
        sa_cost = simulated_annealing(
            points, initial, SA_SCHEDULE, SA_ITERATIONS,
            random.Random(SEED + 1_000_000 + number), SA_START_TEMP, SA_END_TEMP,
        ).best_cost
    else:
        sa_cost = tour_cost(initial, points)
    t_sa = time.perf_counter() - started

    started = time.perf_counter()
    mst_tour, _, _ = two_opt(mst_preorder_tour(points), points)
    t_mst = time.perf_counter() - started

    costs = {
        "2opt_hill_climbing": tour_cost(hill_tour, points),
        "sa_2opt": sa_cost,
        "mst_2opt": tour_cost(mst_tour, points),
    }
    winner, tied = best_algorithm(costs)
    return {
        "instance_id": instance_id,
        "cost_2opt_hill_climbing": costs["2opt_hill_climbing"],
        "cost_sa_2opt": costs["sa_2opt"],
        "cost_mst_2opt": costs["mst_2opt"],
        "best_algorithm": winner,
        "n_tied_best": tied,
        "time_2opt_hill_climbing": t_hill,
        "time_sa_2opt": t_sa,
        "time_mst_2opt": t_mst,
    }


def label_split(name: str, pool: mp.pool.Pool) -> None:
    with (DATA / f"{name}.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    results = pool.map(run_instance, [row["instance_id"] for row in rows], chunksize=4)
    extra = [key for key in results[0] if key != "instance_id"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_labeled.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) + extra)
        writer.writeheader()
        for row, result in zip(rows, results):
            assert row["instance_id"] == result["instance_id"]
            writer.writerow({**row, **{key: result[key] for key in extra}})
    counts = {a: sum(r["best_algorithm"] == a for r in results) for a in ALGORITHMS}
    print(f"{name}: {len(results)} instancias -> {counts}")


def main() -> None:
    started = time.perf_counter()
    with mp.Pool() as pool:
        for name in ("train", "test"):
            label_split(name, pool)
    print(f"Listo en {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
