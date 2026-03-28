"""
train.py — Entraînement et évaluation des modèles de credit scoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42
DOCS_DIR = Path(__file__).parent.parent / "docs"


def train_models(X_train_proc, y_train) -> dict:
    """Entraîne XGBoost, LightGBM et Decision Tree.

    Returns:
        dict {nom: modèle entraîné}
    """
    print("Entraînement des modèles...")
    class_weight = {0: 2.0, 1: 1.0}

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2.0, eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight=class_weight, random_state=RANDOM_STATE, verbosity=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight=class_weight, random_state=RANDOM_STATE,
        ),
    }

    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        print(f"  ✓ {name}")

    return models


def evaluate_models(models: dict, X_test_proc, y_test) -> dict:
    """Calcule AUC-ROC et AUPRC pour chaque modèle.

    Returns:
        dict {nom: {model, y_pred, y_proba, AUC-ROC, AUPRC}}
    """
    print("\nÉvaluation :")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        results[name] = dict(model=model, y_pred=y_pred, y_proba=y_proba,
                              **{"AUC-ROC": auc, "AUPRC": auprc})
        print(f"  {name:15s} | AUC-ROC: {auc:.4f} | AUPRC: {auprc:.4f}")

    print("\n" + classification_report(
        y_test, results["XGBoost"]["y_pred"],
        target_names=["Mauvais (0)", "Bon (1)"]
    ))
    return results


def plot_model_comparison(results: dict, y_test):
    """Courbes ROC/PR et matrice de confusion."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des modèles", fontsize=13, fontweight="bold")
    colors = ["#2563EB", "#16A34A", "#D97706"]

    for (name, res), color in zip(results.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, res["y_proba"], ax=axes[0],
            name=f"{name} ({res['AUC-ROC']:.3f})", color=color,
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, res["y_proba"], ax=axes[1],
            name=f"{name} ({res['AUPRC']:.3f})", color=color,
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("Courbes ROC", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Courbes Précision-Rappel", fontweight="bold")
    axes[1].legend(fontsize=8)

    ConfusionMatrixDisplay.from_predictions(
        y_test, results["XGBoost"]["y_pred"],
        display_labels=["Mauvais (0)", "Bon (1)"],
        ax=axes[2], colorbar=False, cmap="Blues",
    )
    axes[2].set_title("Matrice de confusion — XGBoost", fontweight="bold")

    plt.tight_layout()
    out = DOCS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")


def print_decision_tree_rules(X_train_proc, y_train, feature_names: list):
    """Affiche les règles d'un Decision Tree lisible (depth=3)."""
    dt = DecisionTreeClassifier(max_depth=3, class_weight={0: 2.0, 1: 1.0},
                                 random_state=RANDOM_STATE)
    dt.fit(X_train_proc, y_train)
    print("\n=== Règles du Decision Tree (depth=3, interprétable) ===")
    print(export_text(dt, feature_names=feature_names, max_depth=3))


def save_models(results: dict, preprocessor, feature_names: list):
    """Sérialise le pipeline XGBoost principal."""
    import json
    model_dir = DOCS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["XGBoost"]["model"], model_dir / "xgb_model.pkl")
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump({"all_encoded": feature_names}, f, indent=2)

    print(f"  Modèles sauvegardés dans {model_dir}")
