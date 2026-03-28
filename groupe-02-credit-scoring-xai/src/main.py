"""
main.py — Orchestrateur du pipeline Credit Scoring XAI
ECE Paris — Ing4 Finance IA — Projet 2 — Groupe 02

Usage :
    python main.py               # pipeline complet
    python main.py --skip-dice   # sans contrefactuels (plus rapide)
    python dashboard.py          # dashboard interactif (séparé)
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# Assurer que src/ est dans le path si lancé depuis la racine
sys.path.insert(0, str(Path(__file__).parent))

from data import load_dataset, split_and_preprocess
from train import (
    train_models, evaluate_models, plot_model_comparison,
    print_decision_tree_rules, save_models,
)
from explain import (
    compute_shap, plot_shap_global, plot_shap_local,
    plot_shap_dependence, build_lime_explainer, explain_lime_local,
    compare_shap_lime, generate_counterfactuals,
)
from fairness import run_fairness_audit

DOCS_DIR = Path(__file__).parent.parent / "docs"


def main(skip_dice: bool = False):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Données ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. CHARGEMENT DES DONNÉES")
    print("=" * 60)
    X, y = load_dataset()
    data = split_and_preprocess(X, y)

    # ── 2. Modèles ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)
    models = train_models(data["X_train_proc"], data["y_train"])
    results = evaluate_models(models, data["X_test_proc"], data["y_test"])
    plot_model_comparison(results, data["y_test"])
    print_decision_tree_rules(data["X_train_proc"], data["y_train"], data["feature_names"])
    save_models(results, data["preprocessor"], data["feature_names"])

    xgb_result = results["XGBoost"]

    # ── 3. SHAP ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. EXPLICABILITÉ — SHAP")
    print("=" * 60)
    explainer, shap_values, shap_explanation = compute_shap(
        xgb_result["model"], data["X_test_proc"]
    )
    plot_shap_global(shap_values, data["X_test_proc"], data["feature_names"])
    accepted_idx, refused_idx = plot_shap_local(
        shap_explanation,
        xgb_result["y_pred"], data["y_test"], xgb_result["y_proba"],
    )
    plot_shap_dependence(shap_values, data["X_test_proc"], data["feature_names"])

    # ── 4. LIME ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. EXPLICABILITÉ — LIME")
    print("=" * 60)
    lime_explainer = build_lime_explainer(data["X_train_proc"], data["feature_names"])
    lime_exp = explain_lime_local(
        lime_explainer, xgb_result["model"], data["X_test_proc"], refused_idx
    )
    compare_shap_lime(shap_values, lime_exp, data["feature_names"], refused_idx)

    # ── 5. Contrefactuels ─────────────────────────────────────────────────────
    if not skip_dice:
        print("\n" + "=" * 60)
        print("5. CONTREFACTUELS (DiCE)")
        print("=" * 60)
        import joblib
        from sklearn.pipeline import Pipeline
        import xgboost as xgb_lib

        # Reconstruction pipeline complet pour DiCE (travaille sur X original)
        from data import build_preprocessor
        full_pipeline = Pipeline([
            ("preprocessor", build_preprocessor(data["numerical_cols"], data["categorical_cols"])),
            ("classifier", xgb_lib.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=2.0, eval_metric="auc",
                random_state=42, verbosity=0,
            )),
        ])
        full_pipeline.fit(data["X_train"], data["y_train"])

        generate_counterfactuals(
            data["X_train"], data["y_train"], data["X_test"],
            full_pipeline, data["numerical_cols"], refused_idx,
        )
    else:
        print("\n(Contrefactuels ignorés — --skip-dice)")

    # ── 6. Fairness ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. AUDIT DE BIAIS (FAIRLEARN)")
    print("=" * 60)
    run_fairness_audit(
        data["y_test"],
        xgb_result["y_pred"],
        xgb_result["y_proba"],
        data["X_test"],
    )

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  XGBoost AUC-ROC : {xgb_result['AUC-ROC']:.4f}")
    print(f"  XGBoost AUPRC   : {xgb_result['AUPRC']:.4f}")
    print(f"\n  Graphiques générés dans : {DOCS_DIR.resolve()}/")
    print("  Modèles dans : docs/models/")
    print("\nPour lancer le dashboard : python src/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring XAI pipeline")
    parser.add_argument("--skip-dice", action="store_true",
                        help="Ignorer l'étape contrefactuels (DiCE)")
    args = parser.parse_args()
    main(skip_dice=args.skip_dice)
