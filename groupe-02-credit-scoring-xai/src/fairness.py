"""
fairness.py — Audit de biais et fairness (Fairlearn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference, equalized_odds_difference,
)

DOCS_DIR = Path(__file__).parent.parent / "docs"
BIAS_THRESHOLD = 0.10  # seuil d'alerte


def run_fairness_audit(y_test, y_pred, y_proba, X_test: pd.DataFrame) -> dict:
    """Calcule les métriques de fairness par groupe d'âge.

    Returns:
        dict avec les métriques clés
    """
    age = X_test["age"].values
    age_groups = pd.cut(
        age, bins=[0, 30, 50, 100],
        labels=["Jeunes (<30)", "Adultes (30-50)", "Seniors (>50)"],
    )

    # Demographic Parity
    dp_diff = demographic_parity_difference(
        y_test.values, y_pred, sensitive_features=age_groups
    )

    # Equalized Odds
    eo_diff = equalized_odds_difference(
        y_test.values, y_pred, sensitive_features=age_groups
    )

    # Métriques par groupe
    print("\n=== Audit de biais — par groupe d'âge ===")
    print(f"{'Groupe':20s} | {'N':>4} | {'Taux accord':>11} | {'AUC-ROC':>7}")
    print("-" * 52)
    group_metrics = {}
    for group in age_groups.dtype.categories:
        mask = (age_groups == group)
        if mask.sum() < 5:
            continue
        rate = y_pred[mask].mean()
        auc = roc_auc_score(y_test.values[mask], y_proba[mask])
        group_metrics[group] = {"n": mask.sum(), "rate": rate, "auc": auc}
        print(f"  {group:18s} | {mask.sum():4d} | {rate:11.1%} | {auc:7.4f}")

    print(f"\nDemographic Parity Difference : {dp_diff:.4f}  {'⚠️' if abs(dp_diff) > BIAS_THRESHOLD else '✅'}")
    print(f"Equalized Odds Difference     : {eo_diff:.4f}  {'⚠️' if abs(eo_diff) > BIAS_THRESHOLD else '✅'}")
    print(f"(seuil d'alerte : >{BIAS_THRESHOLD})")

    _plot_fairness(group_metrics, dp_diff, eo_diff)

    return {
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "group_metrics": group_metrics,
        "age_groups": age_groups,
    }


def _plot_fairness(group_metrics: dict, dp_diff: float, eo_diff: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Audit de biais — XGBoost", fontsize=13, fontweight="bold")

    # Taux d'accord par groupe
    ax = axes[0]
    groups = list(group_metrics.keys())
    rates = [v["rate"] for v in group_metrics.values()]
    counts = [v["n"] for v in group_metrics.values()]
    overall_rate = sum(v["rate"] * v["n"] for v in group_metrics.values()) / sum(counts)
    colors = ["#D97706", "#2563EB", "#7C3AED"]

    bars = ax.bar(groups, rates, color=colors[:len(groups)], alpha=0.85, edgecolor="white")
    ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1.5,
               label=f"Taux global ({overall_rate:.1%})")
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}\n(n={n})", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Taux d'accord")
    ax.set_title("Taux d'accord par groupe d'âge", fontweight="bold")
    ax.legend(fontsize=9)

    # Métriques de fairness
    ax = axes[1]
    metric_labels = ["Demographic\nParity Diff.", "Equalized\nOdds Diff."]
    metric_values = [abs(dp_diff), abs(eo_diff)]
    bar_colors = ["#DC2626" if v > BIAS_THRESHOLD else "#16A34A" for v in metric_values]

    bars2 = ax.bar(metric_labels, metric_values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(BIAS_THRESHOLD, color="orange", linestyle="--", linewidth=1.5,
               label=f"Seuil d'alerte ({BIAS_THRESHOLD})")
    for bar, val in zip(bars2, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Différence (0 = équitable)")
    ax.set_title("Métriques de fairness (groupe d'âge)", fontweight="bold")
    ax.set_ylim(0, max(metric_values) * 1.5 + 0.05)

    ok = mpatches.Patch(color="#16A34A", alpha=0.85, label="Acceptable")
    warn = mpatches.Patch(color="#DC2626", alpha=0.85, label="Préoccupant")
    ax.legend(handles=[ok, warn, ax.get_lines()[0]], fontsize=8)

    plt.tight_layout()
    out = DOCS_DIR / "fairness_audit.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")
