"""
explain.py — Explicabilité : SHAP, LIME, contrefactuels (DiCE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import lime
import lime.lime_tabular
from pathlib import Path

RANDOM_STATE = 42
DOCS_DIR = Path(__file__).parent.parent / "docs"


# ─── SHAP ────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test_proc) -> tuple:
    """Calcule les SHAP values via TreeExplainer.

    Returns:
        explainer, shap_values, shap_explanation
    """
    print("Calcul des SHAP values...")
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(X_test_proc)
    shap_explanation = explainer(X_test_proc)
    print(f"  Shape : {shap_values.shape}")
    return explainer, shap_values, shap_explanation


def plot_shap_global(shap_values, X_test_proc, feature_names: list):
    """Summary plot (beeswarm) et bar plot."""
    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names,
                      plot_type="dot", max_display=20, show=False)
    plt.title("SHAP Summary — Impact global des features (XGBoost)",
              fontweight="bold", pad=15)
    plt.tight_layout()
    out = DOCS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")

    # Bar
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names,
                      plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Feature Importance (|valeur moyenne|)", fontweight="bold", pad=15)
    plt.tight_layout()
    out = DOCS_DIR / "shap_importance.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")


def plot_shap_local(shap_explanation, y_pred, y_test, y_proba):
    """Waterfall plots pour un client accepté et un client refusé."""
    accepted = np.where((y_pred == 1) & (y_test.values == 1))[0][0]
    refused = np.where((y_pred == 0) & (y_test.values == 0))[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plt.sca(axes[0])
    shap.waterfall_plot(shap_explanation[accepted], max_display=12, show=False)
    axes[0].set_title(
        f"Client ACCEPTÉ — Probabilité : {y_proba[accepted]:.1%}",
        fontweight="bold", color="#16A34A",
    )

    plt.sca(axes[1])
    shap.waterfall_plot(shap_explanation[refused], max_display=12, show=False)
    axes[1].set_title(
        f"Client REFUSÉ — Probabilité : {y_proba[refused]:.1%}",
        fontweight="bold", color="#DC2626",
    )

    plt.tight_layout()
    out = DOCS_DIR / "shap_waterfall_local.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")

    return accepted, refused


def plot_shap_dependence(shap_values, X_test_proc, feature_names: list):
    """Dependence plots pour l'âge et le montant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, feat in zip(axes, ["age", "credit_amount"]):
        plt.sca(ax)
        shap.dependence_plot(feat, shap_values, X_test_proc,
                             feature_names=feature_names, ax=ax, show=False)
        ax.set_title(f"Effet de « {feat} » (SHAP)", fontweight="bold")

    plt.tight_layout()
    out = DOCS_DIR / "shap_dependence.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")


# ─── LIME ────────────────────────────────────────────────────────────────────

def build_lime_explainer(X_train_proc, feature_names: list):
    """Initialise l'explainer LIME."""
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_proc,
        feature_names=feature_names,
        class_names=["Refusé (0)", "Accordé (1)"],
        mode="classification",
        random_state=RANDOM_STATE,
    )


def explain_lime_local(lime_explainer, model, X_test_proc, refused_idx: int):
    """Génère et sauvegarde l'explication LIME pour le client refusé."""
    predict_fn = lambda x: model.predict_proba(x)
    exp = lime_explainer.explain_instance(
        data_row=X_test_proc[refused_idx],
        predict_fn=predict_fn,
        num_features=12,
        num_samples=1000,
    )
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 5)
    fig.axes[0].set_title("LIME — Explication locale — Client refusé", fontweight="bold")
    fig.tight_layout()
    out = DOCS_DIR / "lime_refused.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out}")
    return exp


def compare_shap_lime(shap_values, lime_exp, feature_names: list, refused_idx: int):
    """Affiche côte-à-côte les top features SHAP et LIME."""
    print("\n=== Comparaison SHAP vs LIME — Client refusé ===")

    shap_series = pd.Series(shap_values[refused_idx], index=feature_names)
    print("\nTop 5 features SHAP (|impact|) :")
    for feat, val in shap_series.abs().nlargest(5).items():
        direction = "↑ Refus" if shap_series[feat] < 0 else "↑ Accord"
        print(f"  {feat:45s} | {shap_series[feat]:+.4f} | {direction}")

    print("\nTop 5 features LIME (poids) :")
    for feat, weight in lime_exp.as_list()[:5]:
        direction = "↑ Accord" if weight > 0 else "↑ Refus"
        print(f"  {feat:55s} | {weight:+.4f} | {direction}")


# ─── Contrefactuels (DiCE) ───────────────────────────────────────────────────

def generate_counterfactuals(X_train, y_train, X_test, pipeline,
                              numerical_cols: list, refused_idx: int):
    """Génère des scénarios contrefactuels avec DiCE."""
    try:
        import dice_ml

        train_df = pd.concat([
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True)
        ], axis=1)

        d = dice_ml.Data(
            dataframe=train_df,
            continuous_features=numerical_cols,
            outcome_name="target",
        )
        m = dice_ml.Model(model=pipeline, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        client = X_test.reset_index(drop=True).iloc[[refused_idx]]
        cf = exp.generate_counterfactuals(
            client,
            total_CFs=4,
            desired_class="opposite",
            features_to_vary=["duration", "credit_amount", "employment",
                               "age", "installment_commitment"],
        )
        _plot_counterfactuals(cf, client, numerical_cols)
        return cf

    except Exception as e:
        print(f"  DiCE non disponible ou erreur : {e}")
        return None


def _plot_counterfactuals(cf, client, numerical_cols):
    cols_of_interest = ["age", "duration", "credit_amount", "installment_commitment"]
    try:
        cf_df = cf.cf_examples_list[0].final_cfs_df
        fig, axes = plt.subplots(1, len(cols_of_interest), figsize=(14, 4))
        fig.suptitle("Contrefactuels : que changer pour être accepté ?",
                     fontsize=13, fontweight="bold")

        for ax, col in zip(axes, cols_of_interest):
            original_val = float(client[col].values[0])
            cf_vals = cf_df[col].astype(float).values if col in cf_df.columns else []
            scenarios = ["Actuel"] + [f"CF {i+1}" for i in range(len(cf_vals))]
            values = [original_val] + list(cf_vals)
            colors = ["#DC2626"] + ["#16A34A"] * len(cf_vals)
            bars = ax.bar(scenarios, values, color=colors, alpha=0.85, edgecolor="white")
            ax.set_title(col.replace("_", " ").title(), fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                        f"{val:.0f}", ha="center", va="center",
                        fontweight="bold", color="white", fontsize=9)
            ax.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        out = DOCS_DIR / "counterfactuals.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Sauvegardé : {out}")
    except Exception as e:
        print(f"  Visualisation contrefactuels : {e}")
