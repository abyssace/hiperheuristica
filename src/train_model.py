"""Parte 2: entrena y evalúa la hiper-heurística de selección.

Uso:
    python -m src.train_model

Lee data/labeled/{train,test}_labeled.csv (generados por src.label_instances),
elige modelo e hiperparámetros con validación cruzada SOLO sobre entrenamiento,
evalúa una vez en prueba y escribe métricas y figuras en results/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20261119

ALGORITHMS = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
LABELS = {"2opt_hill_climbing": "2-opt HC", "sa_2opt": "SA + 2-opt", "mst_2opt": "MST + 2-opt"}
COLORS = {"2opt_hill_climbing": "#2a78d6", "sa_2opt": "#eb6834", "mst_2opt": "#1baf7a"}
DISTRIBUTIONS = ["uniform", "clustered", "border"]
PROVIDED = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
ONE_HOT = [f"dist_{name}" for name in DISTRIBUTIONS]
COMPUTED = ["nn_dist_cv", "mst_length", "bbox_area"]
FEATURE_SETS = {
    "provistas": PROVIDED + ONE_HOT,
    "provistas+calculadas": PROVIDED + ONE_HOT + COMPUTED,
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 9.5, "grid.color": GRID, "grid.linewidth": 0.8,
})


# --- Características calculadas a partir de las coordenadas -------------------
def coordinate_features(instance_id: str) -> dict[str, float]:
    points = pd.read_csv(DATA / "instances" / f"{instance_id}.csv").to_numpy()
    n = len(points)
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    nearest = np.where(np.eye(n, dtype=bool), np.inf, dist).min(axis=1)
    # Longitud del MST (Prim): cota inferior del tour y medida de "estructura".
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    mst_length = 0.0
    for _ in range(n - 1):
        best[in_tree] = np.inf
        nxt = int(best.argmin())
        mst_length += float(best[nxt])
        in_tree[nxt] = True
        best = np.minimum(best, dist[nxt])
    span = points.max(axis=0) - points.min(axis=0)
    return {
        "nn_dist_cv": float(nearest.std() / nearest.mean()) if nearest.mean() > 0 else 0.0,
        "mst_length": mst_length,
        "bbox_area": float(span[0] * span[1]),
    }


def load(split: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / "labeled" / f"{split}_labeled.csv")
    for name in DISTRIBUTIONS:
        frame[f"dist_{name}"] = (frame["distribution"] == name).astype(int)
    extra = pd.DataFrame([coordinate_features(i) for i in frame["instance_id"]])
    return pd.concat([frame, extra], axis=1)


# --- Regret -------------------------------------------------------------------
def regret(frame: pd.DataFrame, chosen) -> np.ndarray:
    """Costo del algoritmo elegido menos el mejor costo real de cada instancia."""
    costs = frame[[f"cost_{a}" for a in ALGORITHMS]].to_numpy()
    index = np.array([ALGORITHMS.index(a) for a in chosen])
    return costs[np.arange(len(frame)), index] - costs.min(axis=1)


def regret_summary(frame: pd.DataFrame, chosen) -> dict[str, float]:
    values = regret(frame, chosen)
    best = frame[[f"cost_{a}" for a in ALGORITHMS]].min(axis=1).to_numpy()
    return {
        "regret_mean": float(values.mean()),
        "regret_median": float(np.median(values)),
        "regret_max": float(values.max()),
        "regret_rel_mean_pct": float((values / best).mean() * 100),
        "zero_regret_pct": float((values <= 1e-9).mean() * 100),
    }


# --- Selección de modelo (solo con entrenamiento) -----------------------------
def candidates() -> dict[str, tuple[object, dict]]:
    return {
        "Árbol de decisión": (
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 3, 5, 10, 20],
             "criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
            {"knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
             "knn__weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED),
            {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10],
             "class_weight": [None, "balanced"]},
        ),
    }


def select_model(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], GridSearchCV]]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, searches = [], {}
    for set_name, columns in FEATURE_SETS.items():
        for model_name, (estimator, grid) in candidates().items():
            search = GridSearchCV(estimator, grid, cv=folds, scoring="accuracy", n_jobs=-1)
            search.fit(train[columns], train["best_algorithm"])
            std = search.cv_results_["std_test_score"][search.best_index_]
            searches[(model_name, set_name)] = search
            rows.append({"modelo": model_name, "caracteristicas": set_name,
                         "cv_accuracy": search.best_score_, "cv_std": std,
                         "mejores_hiperparametros": json.dumps(search.best_params_)})
    return pd.DataFrame(rows), searches


# --- Figuras ------------------------------------------------------------------
def fig_class_balance(full: pd.DataFrame) -> None:
    share = full["best_algorithm"].value_counts(normalize=True).reindex(ALGORITHMS) * 100
    counts = full["best_algorithm"].value_counts().reindex(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.barh([LABELS[a] for a in ALGORITHMS], share, height=0.55,
                   color=[COLORS[a] for a in ALGORITHMS])
    for bar, a in zip(bars, ALGORITHMS):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{share[a]:.1f} %  ({counts[a]})", va="center", color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 75); ax.set_xlabel("% de las 500 instancias")
    ax.set_title("Balance de clases: algoritmo ganador"); ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RESULTS / "class_balance.png", dpi=170); plt.close(fig)


def fig_winner_patterns(full: pd.DataFrame) -> None:
    sizes = sorted(full["n"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, name in zip(axes, DISTRIBUTIONS):
        sub = full[full["distribution"] == name]
        table = pd.crosstab(sub["n"], sub["best_algorithm"], normalize="index").reindex(
            index=sizes, columns=ALGORITHMS, fill_value=0) * 100
        bottom = np.zeros(len(sizes))
        for a in ALGORITHMS:
            ax.bar(range(len(sizes)), table[a], bottom=bottom, width=0.72, color=COLORS[a],
                   edgecolor="#fcfcfb", linewidth=1.5, label=LABELS[a])
            bottom += table[a].to_numpy()
        ax.set_xticks(range(len(sizes)), sizes); ax.set_xlabel("n (ciudades)")
        ax.set_title(name); ax.set_ylim(0, 100)
    axes[0].set_ylabel("% de instancias que gana")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    fig.suptitle("Algoritmo ganador por tamaño y distribución (500 instancias)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(RESULTS / "winner_by_n_distribution.png", dpi=170,
                                    bbox_inches="tight"); plt.close(fig)


def fig_cost_gap(full: pd.DataFrame) -> None:
    best = full[[f"cost_{a}" for a in ALGORITHMS]].min(axis=1)
    sizes = sorted(full["n"].unique())
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for a in ALGORITHMS:
        gap = ((full[f"cost_{a}"] - best) / best * 100).groupby(full["n"]).mean().reindex(sizes)
        ax.plot(range(len(sizes)), gap, "o-", color=COLORS[a], linewidth=2, markersize=5,
                label=LABELS[a])
        ax.annotate(LABELS[a], (len(sizes) - 1, gap.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK)
    ax.set_xticks(range(len(sizes)), sizes); ax.set_xlim(-0.3, len(sizes) + 0.9)
    ax.set_xlabel("n (ciudades)"); ax.set_ylabel("Brecha media vs. mejor de los tres (%)")
    ax.set_title("Qué tan lejos queda cada algoritmo del mejor, por tamaño")
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(RESULTS / "cost_gap_by_n.png", dpi=170); plt.close(fig)


def fig_confusion(matrix: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.15)
    names = [LABELS[a] for a in ALGORITHMS]
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real (mejor algoritmo)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                    color="#ffffff" if matrix[i, j] > matrix.max() * 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0); ax.set_title(title)
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrix.png", dpi=170); plt.close(fig)


def fig_importance(importance: pd.DataFrame, tree_importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ordered = importance.sort_values("permutation_importance")
    axes[0].barh(ordered["feature"], ordered["permutation_importance"],
                 xerr=ordered["permutation_std"], color="#2a78d6", height=0.6,
                 error_kw={"ecolor": MUTED, "elinewidth": 1})
    axes[0].set_title("Modelo elegido (permutación, prueba)")
    axes[0].set_xlabel("Caída media de accuracy al permutar (± 1 d.e.)")
    ordered = tree_importance.sort_values("importance")
    axes[1].barh(ordered["feature"], ordered["importance"], color="#2a78d6", height=0.6)
    axes[1].set_title("Árbol complementario (impureza)")
    axes[1].set_xlabel("Fracción de la reducción de impureza (entrenamiento)")
    for ax in axes:
        ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RESULTS / "feature_importance.png", dpi=170); plt.close(fig)


def fig_regret(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    colors = ["#2a78d6" if s.startswith("Hiper") else "#c3c2b7" for s in table["estrategia"]]
    bars = ax.barh(table["estrategia"], table["regret_mean"], color=colors, height=0.6)
    for bar, value, rel in zip(bars, table["regret_mean"], table["regret_rel_mean_pct"]):
        ax.text(bar.get_width() + table["regret_mean"].max() * 0.015,
                bar.get_y() + bar.get_height() / 2, f"{value:.4f}  ({rel:.2f} %)",
                va="center", color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, table["regret_mean"].max() * 1.32)
    ax.set_xlabel("Regret promedio en prueba (unidades de costo; % relativo entre paréntesis)")
    ax.set_title("Regret promedio en prueba por estrategia")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RESULTS / "regret_comparison.png", dpi=170); plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train, test = load("train"), load("test")
    full = pd.concat([train, test], ignore_index=True)

    # 1) Balance de clases y baseline trivial (clase mayoritaria de entrenamiento).
    balance = pd.DataFrame({
        "total": full["best_algorithm"].value_counts(),
        "total_pct": full["best_algorithm"].value_counts(normalize=True) * 100,
        "train": train["best_algorithm"].value_counts(),
        "test": test["best_algorithm"].value_counts(),
    }).reindex(ALGORITHMS)
    balance.to_csv(RESULTS / "class_balance.csv", index_label="best_algorithm")
    majority = train["best_algorithm"].value_counts().idxmax()
    baseline_pred = np.full(len(test), majority)
    baseline_acc = accuracy_score(test["best_algorithm"], baseline_pred)

    # 2) Selección de modelo con validación cruzada estratificada (solo train).
    cv_table, searches = select_model(train)
    cv_table = cv_table.sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
    cv_table.to_csv(RESULTS / "model_selection_cv.csv", index=False)
    chosen_model, chosen_set = cv_table.loc[0, ["modelo", "caracteristicas"]]
    search = searches[(chosen_model, chosen_set)]
    columns = FEATURE_SETS[chosen_set]
    model = search.best_estimator_

    # 3) Evaluación única en prueba.
    pred = model.predict(test[columns])
    accuracy = accuracy_score(test["best_algorithm"], pred)
    matrix = confusion_matrix(test["best_algorithm"], pred, labels=ALGORITHMS)
    recalls = recall_score(test["best_algorithm"], pred, labels=ALGORITHMS, average=None,
                           zero_division=0)
    pd.DataFrame(matrix, index=[f"real_{a}" for a in ALGORITHMS],
                 columns=[f"pred_{a}" for a in ALGORITHMS]).to_csv(RESULTS / "confusion_matrix.csv")

    strategies = {"Hiper-heurística (modelo)": pred}
    strategies.update({f"Siempre {LABELS[a]}": np.full(len(test), a) for a in ALGORITHMS})
    rng = np.random.default_rng(SEED)
    regret_rows = [{"estrategia": name, **regret_summary(test, chosen)}
                   for name, chosen in strategies.items()]
    random_regret = np.mean([regret(test, rng.choice(ALGORITHMS, len(test))).mean()
                             for _ in range(1000)])
    regret_table = pd.DataFrame(regret_rows)
    regret_table.to_csv(RESULTS / "regret_test.csv", index=False)

    per_instance = test[["instance_id", "n", "distribution", "best_algorithm", "n_tied_best"]
                        + [f"cost_{a}" for a in ALGORITHMS]].copy()
    per_instance["predicted"] = pred
    per_instance["regret"] = regret(test, pred)
    per_instance.to_csv(RESULTS / "test_predictions.csv", index=False)

    # Otras configuraciones en prueba: solo informativo, no se usó para elegir.
    others = []
    for (model_name, set_name), other in searches.items():
        other_pred = other.best_estimator_.predict(test[FEATURE_SETS[set_name]])
        others.append({
            "modelo": model_name, "caracteristicas": set_name,
            "cv_accuracy": other.best_score_,
            "test_accuracy": accuracy_score(test["best_algorithm"], other_pred),
            "test_balanced_accuracy": balanced_accuracy_score(test["best_algorithm"], other_pred),
            "test_regret_mean": float(regret(test, other_pred).mean()),
        })
    pd.DataFrame(others).sort_values("cv_accuracy", ascending=False).to_csv(
        RESULTS / "all_models_test.csv", index=False)

    # 4) Importancia de características.
    perm = permutation_importance(model, test[columns], test["best_algorithm"],
                                  n_repeats=200, random_state=SEED, scoring="accuracy")
    importance = pd.DataFrame({"feature": columns, "permutation_importance": perm.importances_mean,
                               "permutation_std": perm.importances_std})
    importance.sort_values("permutation_importance", ascending=False).to_csv(
        RESULTS / "feature_importance.csv", index=False)

    # 5) Patrones por tamaño / distribución.
    pd.crosstab([full["distribution"], full["n"]], full["best_algorithm"]).reindex(
        columns=ALGORITHMS, fill_value=0).to_csv(RESULTS / "winners_by_n_distribution.csv")
    by_group = per_instance.assign(correct=per_instance["predicted"] == per_instance["best_algorithm"])
    by_group.groupby("n").agg(instancias=("correct", "size"), accuracy=("correct", "mean"),
                              regret_mean=("regret", "mean")).to_csv(RESULTS / "test_by_n.csv")
    by_group.groupby("distribution").agg(
        instancias=("correct", "size"), accuracy=("correct", "mean"),
        regret_mean=("regret", "mean")).to_csv(RESULTS / "test_by_distribution.csv")

    # 6) Figuras.
    fig_class_balance(full); fig_winner_patterns(full); fig_cost_gap(full)
    fig_confusion(matrix, f"Matriz de confusión en prueba ({chosen_model})")
    fig_regret(regret_table)

    # Árbol complementario (el mejor árbol según CV): solo para interpretar reglas.
    tree_key = max((k for k in searches if k[0] == "Árbol de decisión"),
                   key=lambda k: searches[k].best_score_)
    tree, tree_columns = searches[tree_key].best_estimator_, FEATURE_SETS[tree_key[1]]
    tree_importance = pd.DataFrame({"feature": tree_columns, "importance": tree.feature_importances_})
    tree_importance.sort_values("importance", ascending=False).to_csv(
        RESULTS / "tree_feature_importance.csv", index=False)
    (RESULTS / "decision_tree.txt").write_text(export_text(tree, feature_names=tree_columns))
    fig, ax = plt.subplots(figsize=(max(8, tree.get_n_leaves() * 1.9), 2 + tree.get_depth() * 1.5))
    plot_tree(tree, feature_names=tree_columns, class_names=[LABELS[c] for c in tree.classes_],
              filled=False, rounded=True, impurity=False, fontsize=8, ax=ax)
    fig.tight_layout(); fig.savefig(RESULTS / "decision_tree.png", dpi=170); plt.close(fig)
    fig_importance(importance, tree_importance)

    summary = {
        "n_train": len(train), "n_test": len(test),
        "baseline_class": majority, "baseline_accuracy_test": baseline_acc,
        "model": chosen_model, "feature_set": chosen_set, "features": columns,
        "hyperparameters": search.best_params_, "cv_accuracy": float(search.best_score_),
        "test_accuracy": accuracy,
        "test_balanced_accuracy": balanced_accuracy_score(test["best_algorithm"], pred),
        "test_recall": dict(zip(ALGORITHMS, map(float, recalls))),
        "test_predicted_counts": {a: int((pred == a).sum()) for a in ALGORITHMS},
        "accuracy_ci95_halfwidth": 1.96 * math.sqrt(accuracy * (1 - accuracy) / len(test)),
        "regret": {row["estrategia"]: {k: v for k, v in row.items() if k != "estrategia"}
                   for row in regret_rows},
        "regret_random_choice_mean": float(random_regret),
        "complementary_tree": {"feature_set": tree_key[1],
                               "hyperparameters": searches[tree_key].best_params_},
        "tied_best_pct_all": float((full["n_tied_best"] > 1).mean() * 100),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(balance.round(1).to_string(), "\n")
    print(cv_table.drop(columns="mejores_hiperparametros").round(4).to_string(), "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
