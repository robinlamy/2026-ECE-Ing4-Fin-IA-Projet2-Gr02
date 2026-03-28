# Credit Scoring avec IA Explicable (XAI)

**ECE Paris — Ing4 Finance IA — Projet 2 — Groupe 02 — 2026**

---

## Présentation

Ce projet implémente un pipeline complet de **credit scoring** avec explicabilité (XAI) sur le *German Credit Dataset* (UCI, 1000 demandes réelles). Il répond à la contrainte réglementaire RGPD Art. 22 : tout refus de crédit automatisé doit pouvoir être expliqué au demandeur.

### Pipeline

```
Données (OpenML)
    → data.py       Chargement, encodage, split train/test
    → train.py      XGBoost + LightGBM + Decision Tree, métriques AUC/AUPRC
    → explain.py    SHAP (global + local) · LIME · Contrefactuels (DiCE)
    → fairness.py   Audit de biais Fairlearn (parité démographique, equalized odds)
    → dashboard.py  Interface Gradio interactive
```

---

## Installation

```bash
# Cloner le repo et se placer dans le dossier du projet
cd groupe-02-credit-scoring-xai

# (Recommandé) Environnement virtuel
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# ou : .venv\Scripts\activate                        # Windows

# Dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Pipeline complet

```bash
python src/main.py
```

Génère dans `docs/` :
- `model_comparison.png` — Courbes ROC/PR + matrice de confusion
- `shap_summary.png` — Importance globale des features
- `shap_importance.png` — SHAP bar plot
- `shap_waterfall_local.png` — Explication locale client accepté vs refusé
- `shap_dependence.png` — Effet de l'âge et du montant
- `lime_refused.png` — Explication LIME locale
- `counterfactuals.png` — Scénarios « que changer ? »
- `fairness_audit.png` — Parité démographique par groupe d'âge

Option sans contrefactuels (plus rapide) :

```bash
python src/main.py --skip-dice
```

### Dashboard interactif

```bash
python src/dashboard.py
# → http://localhost:7860
```

Permet de saisir un profil client complet et d'obtenir la décision + explication SHAP + explication LIME en temps réel.

---

## Structure

```
groupe-02-credit-scoring-xai/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py         Chargement et prétraitement
│   ├── train.py        Entraînement et évaluation
│   ├── explain.py      SHAP, LIME, contrefactuels
│   ├── fairness.py     Audit de biais Fairlearn
│   ├── main.py         Orchestrateur
│   └── dashboard.py    Interface Gradio
├── docs/               Graphiques générés + modèles sérialisés
└── slides/             Support de présentation
```

---

## Résultats

| Modèle | AUC-ROC | AUPRC | Interprétable |
|---|---|---|---|
| XGBoost | ~0.80 | ~0.87 | Non → XAI |
| LightGBM | ~0.79 | ~0.86 | Non → XAI |
| Decision Tree | ~0.72 | ~0.81 | Oui (règles directes) |

---

## Méthodes XAI

**SHAP** (Shapley Additive exPlanations) — basé sur la théorie des jeux coopératifs, calcule la contribution marginale de chaque feature sur l'ensemble des coalitions possibles. Fournit des explications globales (summary plot) et locales (waterfall).

**LIME** (Local Interpretable Model-agnostic Explanations) — génère des perturbations autour d'une instance et ajuste un modèle linéaire local. Model-agnostic, moins stable que SHAP mais indépendant de l'architecture.

**DiCE** (Diverse Counterfactual Explanations) — génère des scénarios alternatifs : « si votre durée de crédit était de 18 mois au lieu de 36, vous auriez été accepté ». Forme d'explication la plus actionnable pour le client.

**Fairlearn** — mesure la *demographic parity difference* et l'*equalized odds difference* entre groupes démographiques (ici : groupes d'âge). Seuil d'alerte à 0.10.

---

## Références

- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017
- Ribeiro et al., *"Why Should I Trust You?"*, KDD 2016
- Mothilal et al., *Diverse Counterfactual Explanations*, FAccT 2020
- Bird et al., *Fairlearn: A toolkit for assessing ML fairness*, Microsoft 2020
- RGPD Art. 22 — Décisions individuelles automatisées
