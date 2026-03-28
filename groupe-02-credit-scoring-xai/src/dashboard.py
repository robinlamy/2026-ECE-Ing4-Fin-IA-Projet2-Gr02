"""
dashboard.py — Dashboard Gradio interactif
ECE Paris — Ing4 Finance IA — Projet 2 — Groupe 02

Lancement : python src/dashboard.py
Puis ouvrir http://localhost:7860
"""

import warnings
warnings.filterwarnings("ignore")
import sys, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json
import shap
import lime.lime_tabular
import gradio as gr

from data import load_dataset, split_and_preprocess
from sklearn.pipeline import Pipeline
import xgboost as xgb

RANDOM_STATE = 42
DOCS_DIR = Path(__file__).parent.parent / "docs"
MODEL_DIR = DOCS_DIR / "models"

# ─── Traductions ─────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "<0":           "Compte en découvert",
    "0<=X<200":     "Solde entre 0 et 200 DM",
    ">=200":        "Solde supérieur à 200 DM",
    "no checking":  "Pas de compte courant",
    "no credits/all paid":            "Aucun crédit antérieur",
    "all paid":                       "Tous les crédits remboursés",
    "existing paid":                  "Crédits en cours remboursés",
    "delayed previously":             "Retards de paiement passés",
    "critical/other existing credit": "Situation critique ou autres crédits",
    "new car":             "Voiture neuve",
    "used car":            "Voiture d'occasion",
    "furniture/equipment": "Mobilier ou équipement",
    "radio/tv":            "Électronique ou TV",
    "domestic appliance":  "Électroménager",
    "repairs":             "Travaux ou réparations",
    "education":           "Formation ou études",
    "vacation":            "Vacances",
    "retraining":          "Reconversion professionnelle",
    "business":            "Projet professionnel",
    "other":               "Autre",
    "<100":             "Moins de 100 DM d'épargne",
    "100<=X<500":       "Entre 100 et 500 DM d'épargne",
    "500<=X<1000":      "Entre 500 et 1 000 DM d'épargne",
    ">=1000":           "Plus de 1 000 DM d'épargne",
    "no known savings": "Épargne inconnue",
    "unemployed": "Sans emploi",
    "<1":         "Moins d'un an dans l'emploi actuel",
    "1<=X<4":     "Entre 1 et 4 ans dans l'emploi actuel",
    "4<=X<7":     "Entre 4 et 7 ans dans l'emploi actuel",
    ">=7":        "Plus de 7 ans dans l'emploi actuel",
    "male div/sep":       "Homme — divorcé ou séparé",
    "female div/dep/mar": "Femme — divorcée ou mariée",
    "male single":        "Homme — célibataire",
    "male mar/wid":       "Homme — marié ou veuf",
    "female single":      "Femme — célibataire",
    "none":          "Aucun",
    "co applicant":  "Co-emprunteur",
    "guarantor":     "Garant",
    "real estate":       "Bien immobilier",
    "life insurance":    "Assurance-vie",
    "car":               "Véhicule",
    "no known property": "Aucun bien connu",
    "bank":   "Crédit bancaire",
    "stores": "Crédit magasin",
    "rent": "Locataire",
    "free": "Logé gratuitement",
    "own":  "Propriétaire",
    "unskilled resident":        "Ouvrier non qualifié (résident)",
    "unskilled non-res":         "Ouvrier non qualifié (non résident)",
    "skilled":                   "Employé qualifié",
    "high qualif/self emp/mgmt": "Cadre, indépendant ou dirigeant",
    "yes": "Oui",
    "no":  "Non",
}
REVERSE = {v: k for k, v in TRANSLATIONS.items()}

def t(v):      return TRANSLATIONS.get(v, v)
def r(v):      return REVERSE.get(v, v)
def opts(lst): return [t(v) for v in lst]

CAT_RAW = {
    "checking_status":     ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history":      ["no credits/all paid", "all paid", "existing paid",
                            "delayed previously", "critical/other existing credit"],
    "purpose":             ["new car", "used car", "furniture/equipment", "radio/tv",
                            "domestic appliance", "repairs", "education", "vacation",
                            "retraining", "business", "other"],
    "savings_status":      ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment":          ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status":     ["male div/sep", "female div/dep/mar", "male single",
                             "male mar/wid", "female single"],
    "other_parties":       ["none", "co applicant", "guarantor"],
    "property_magnitude":  ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing":             ["rent", "free", "own"],
    "job":                 ["unskilled resident", "unskilled non-res", "skilled",
                            "high qualif/self emp/mgmt"],
    "own_telephone":       ["none", "yes"],
    "foreign_worker":      ["yes", "no"],
}

FEATURE_LABELS = {
    "checking_status": "Statut compte courant",
    "duration": "Durée du crédit",
    "credit_history": "Historique crédit",
    "purpose": "Objet du crédit",
    "credit_amount": "Montant du crédit",
    "savings_status": "Épargne",
    "employment": "Ancienneté emploi",
    "installment_commitment": "Taux d'endettement",
    "personal_status": "Situation personnelle",
    "other_parties": "Co-emprunteur",
    "residence_since": "Ancienneté adresse",
    "property_magnitude": "Bien principal",
    "age": "Age",
    "other_payment_plans": "Autres crédits",
    "housing": "Logement",
    "existing_credits": "Nombre de crédits",
    "job": "Catégorie professionnelle",
    "num_dependents": "Personnes à charge",
    "own_telephone": "Téléphone fixe",
    "foreign_worker": "Travailleur étranger",
}

def prettify_feature(raw: str) -> str:
    for key, label in FEATURE_LABELS.items():
        if raw == key:
            return label
        if raw.startswith(key + "_"):
            suffix = raw[len(key)+1:]
            return f"{label} — {TRANSLATIONS.get(suffix, suffix.replace('_', ' '))}"
    return raw.replace("_", " ").title()

def clean_lime_label(s: str) -> str:
    """Nettoie un label LIME : remplace les opérateurs mathématiques par du texte naturel."""
    s = re.sub(r'(\S+)\s*>\s*[\d.-]+',  lambda m: prettify_feature(m.group(1)) + " (élevé)", s)
    s = re.sub(r'(\S+)\s*<=\s*[\d.-]+', lambda m: prettify_feature(m.group(1)) + " (faible)", s)
    s = re.sub(r'(\S+)\s*<\s*[\d.-]+',  lambda m: prettify_feature(m.group(1)) + " (faible)", s)
    s = re.sub(r'(\S+)\s*>=\s*[\d.-]+', lambda m: prettify_feature(m.group(1)) + " (élevé)", s)
    return s.replace("_", " ")[:60]


# ─── Initialisation ──────────────────────────────────────────────────────────
def _init():
    model_path = MODEL_DIR / "xgb_model.pkl"
    prep_path  = MODEL_DIR / "preprocessor.pkl"
    meta_path  = MODEL_DIR / "feature_names.json"

    if model_path.exists() and prep_path.exists() and meta_path.exists():
        print("Chargement du modèle existant...")
        model        = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        with open(meta_path) as f:
            feature_names = json.load(f)["all_encoded"]
        X, y = load_dataset()
        data = split_and_preprocess(X, y)
    else:
        print("Entraînement du modèle (premier lancement)...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        X, y = load_dataset()
        data = split_and_preprocess(X, y)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=2.0, eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0,
        )
        model.fit(data["X_train_proc"], data["y_train"])
        preprocessor  = data["preprocessor"]
        feature_names = data["feature_names"]
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, prep_path)
        with open(meta_path, "w") as f:
            json.dump({"all_encoded": feature_names}, f)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    shap_exp = shap.TreeExplainer(model)
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=data["X_train_proc"],
        feature_names=feature_names,
        class_names=["Refusé", "Accordé"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    return model, preprocessor, pipeline, shap_exp, lime_exp, feature_names, data

print("Initialisation du dashboard...")
MODEL, PREP, PIPELINE, SHAP_EXP, LIME_EXP, FEATURE_NAMES, DATA = _init()
print("Pret !")


# ─── Palette graphiques (thème clair sobre) ───────────────────────────────────
PLT_BG    = "#ffffff"
PLT_AXES  = "#f8f9fb"
PLT_TEXT  = "#1e293b"
PLT_MUTED = "#64748b"
PLT_GRID  = "#e2e8f0"
PLT_GREEN = "#16a34a"
PLT_RED   = "#dc2626"
PLT_TITLE_ACC = "#15803d"
PLT_TITLE_REF = "#b91c1c"


def make_bar_chart(values, labels, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(PLT_BG)
    ax.set_facecolor(PLT_AXES)

    colors = [PLT_GREEN if v > 0 else PLT_RED for v in values]
    ax.barh(range(len(values)), values[::-1],
            color=colors[::-1], alpha=0.85, edgecolor=PLT_BG, height=0.58)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels[::-1], fontsize=9, color=PLT_TEXT)
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Impact sur la décision", fontsize=9, color=PLT_MUTED)

    ax.tick_params(colors=PLT_MUTED, length=0)
    ax.grid(axis="x", color=PLT_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLT_GRID)
        spine.set_linewidth(0.8)

    p1 = mpatches.Patch(color=PLT_GREEN, alpha=0.85, label="Favorise l'accord")
    p2 = mpatches.Patch(color=PLT_RED,   alpha=0.85, label="Défavorise l'accord")
    legend = ax.legend(handles=[p1, p2], fontsize=8, facecolor=PLT_BG,
                       edgecolor=PLT_GRID, loc="lower right")
    for text in legend.get_texts():
        text.set_color(PLT_TEXT)

    # Titre en haut à gauche, sous-titre en dessous — réservation d'espace explicite
    top_margin = 0.88 if subtitle else 0.92
    fig.subplots_adjust(top=top_margin, left=0.28, right=0.97, bottom=0.12)

    fig.text(0.01, 0.98, title, fontweight="bold", fontsize=11, color=PLT_TEXT,
             ha="left", va="top", transform=fig.transFigure)
    if subtitle:
        fig.text(0.01, 0.93, subtitle, fontsize=9, color=PLT_MUTED,
                 ha="left", va="top", transform=fig.transFigure)
    return fig


# ─── Prédiction ──────────────────────────────────────────────────────────────
def predict_and_explain(*args_fr):
    keys = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment", "installment_commitment", "personal_status",
        "other_parties", "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits", "job",
        "num_dependents", "own_telephone", "foreign_worker",
    ]
    values = [r(v) if k in CAT_RAW else v for k, v in zip(keys, args_fr)]
    client = pd.DataFrame([dict(zip(keys, values))])
    proba  = PIPELINE.predict_proba(client)[0, 1]
    client_proc = PREP.transform(client)

    decision = "Accordé" if proba >= 0.5 else "Refusé"
    dec_color = "#15803d" if proba >= 0.5 else "#b91c1c"
    dec_bg    = "#f0fdf4" if proba >= 0.5 else "#fef2f2"
    dec_border= "#bbf7d0" if proba >= 0.5 else "#fecaca"

    # ── SHAP — regroupé par variable parente ────────────────────────────────
    shap_vals   = SHAP_EXP.shap_values(client_proc)[0]
    shap_raw    = pd.Series(shap_vals, index=FEATURE_NAMES)

    # Sommer les contributions de toutes les modalités one-hot d une même variable
    def parent(name):
        for key in FEATURE_LABELS:
            if name == key or name.startswith(key + "_"):
                return key
        return name

    grouped = {}
    for feat, val in shap_raw.items():
        p = parent(feat)
        grouped[p] = grouped.get(p, 0.0) + val

    shap_series  = pd.Series(grouped)  # utilisé plus bas pour le résumé HTML
    top_grouped  = shap_series[shap_series.abs().nlargest(12).index]
    labels_shap  = [FEATURE_LABELS.get(f, f.replace("_", " ").title())
                    for f in top_grouped.index]

    fig_shap = make_bar_chart(
        values=list(top_grouped.values),
        labels=labels_shap,
        title="Explication SHAP",
        subtitle=f"Probabilité d'accord : {proba:.1%}  —  {decision}",
    )

    # ── LIME ─────────────────────────────────────────────────────────────────
    lime_result = LIME_EXP.explain_instance(
        data_row=client_proc[0],
        predict_fn=lambda x: MODEL.predict_proba(x),
        num_features=10, num_samples=800,
    )
    lime_feats   = lime_result.as_list()
    lime_labels  = [clean_lime_label(f) for f, _ in lime_feats]
    lime_weights = [w for _, w in lime_feats]

    fig_lime = make_bar_chart(
        values=lime_weights,
        labels=lime_labels,
        title="Explication LIME",
        subtitle="Approximation linéaire locale autour du profil analysé",
    )

    # ── Résumé HTML ──────────────────────────────────────────────────────────
    top3_for     = shap_series.nlargest(3)
    top3_against = shap_series.nsmallest(3)

    rows_for = "".join(
        f'<tr><td style="padding:5px 12px 5px 0;color:#374151;font-size:13px;">'
        f'{prettify_feature(f)}</td>'
        f'<td style="padding:5px 0;color:#15803d;font-size:13px;font-weight:600;">'
        f'+{v:.3f}</td></tr>'
        for f, v in top3_for.items()
    )
    rows_against = "".join(
        f'<tr><td style="padding:5px 12px 5px 0;color:#374151;font-size:13px;">'
        f'{prettify_feature(f)}</td>'
        f'<td style="padding:5px 0;color:#b91c1c;font-size:13px;font-weight:600;">'
        f'{v:.3f}</td></tr>'
        for f, v in top3_against.items()
    )

    html = f"""
    <div style="font-family:'DM Sans',sans-serif;">

      <div style="padding:16px 20px;background:{dec_bg};border:1px solid {dec_border};
                  border-radius:10px;margin-bottom:18px;">
        <div style="font-size:11px;font-weight:600;color:{dec_color};
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
          Décision du modèle
        </div>
        <div style="font-size:26px;font-weight:700;color:{dec_color};line-height:1.2;">
          {decision}
        </div>
        <div style="margin-top:6px;font-size:13px;color:#374151;">
          Probabilité d'accord : <strong>{proba:.1%}</strong>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          Probabilité de refus : <strong>{1-proba:.1%}</strong>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;">

        <div style="padding:14px 16px;background:#f8f9fb;border:1px solid #e2e8f0;border-radius:8px;">
          <div style="font-size:10px;font-weight:600;color:#6b7280;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            Facteurs favorables
          </div>
          <table style="border-collapse:collapse;width:100%;">{rows_for}</table>
        </div>

        <div style="padding:14px 16px;background:#f8f9fb;border:1px solid #e2e8f0;border-radius:8px;">
          <div style="font-size:10px;font-weight:600;color:#6b7280;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
            Facteurs défavorables
          </div>
          <table style="border-collapse:collapse;width:100%;">{rows_against}</table>
        </div>

      </div>

      <div style="padding:12px 16px;background:#fffbeb;border:1px solid #fde68a;
                  border-radius:8px;font-size:12px;color:#78350f;line-height:1.6;">
        <strong>Recommandation :</strong> Pour améliorer les chances d'acceptation,
        réduire la durée ou le montant du crédit, améliorer le statut du compte courant,
        ou constituer une épargne préalable.
      </div>

    </div>
    """

    return html, fig_shap, fig_lime


# ─── CSS minimal — thème clair, lisible, professionnel ───────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }

body, .gradio-container {
    background: #f1f5f9 !important;
    color: #1e293b !important;
}

/* Cartes / blocs */
.block, .gr-group, .gr-form {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* Labels */
label > span, .label-wrap span, .svelte-1b6s6s {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Inputs, selects */
input, select, textarea {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
input:focus, select:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
}

/* Dropdown items */
.dropdown-arrow, li span {
    color: #1e293b !important;
}

/* Sliders */
input[type=range] { accent-color: #2563eb !important; }

/* Bouton principal */
.primary {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
.primary:hover { background: #1e40af !important; }

/* Onglets */
.tab-nav button {
    color: #64748b !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #1d4ed8 !important;
    border-bottom-color: #1d4ed8 !important;
}

/* Markdown */
.prose, .prose p, .prose li, .md { color: #1e293b !important; }
.prose h1,.prose h2,.prose h3 { color: #0f172a !important; }

/* Barre de défilement */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
"""


# ─── Interface ───────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Analyse de demande de crédit") as demo:

    gr.HTML("""
    <div style="padding:24px 8px 20px;border-bottom:1px solid #e2e8f0;margin-bottom:24px;">
      <h1 style="margin:0 0 4px;font-size:20px;font-weight:700;color:#0f172a;
                 letter-spacing:-0.3px;">
        Analyse de demande de crédit
      </h1>
      <p style="margin:0;font-size:12px;color:#64748b;">
        Modèle XGBoost · German Credit Dataset (UCI, 1 000 demandes)
        · Explicabilité SHAP et LIME · Conformité RGPD Art. 22
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Formulaire ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):

            with gr.Group():
                gr.HTML('<p style="font-size:10px;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:1.2px;margin:0 0 12px;">'
                        'Situation bancaire</p>')
                checking    = gr.Dropdown(opts(CAT_RAW["checking_status"]),     value=t("0<=X<200"),      label="Statut du compte courant")
                savings     = gr.Dropdown(opts(CAT_RAW["savings_status"]),      value=t("<100"),           label="Épargne disponible")
                cred_hist   = gr.Dropdown(opts(CAT_RAW["credit_history"]),      value=t("existing paid"),  label="Historique de crédit")
                other_plans = gr.Dropdown(opts(CAT_RAW["other_payment_plans"]), value=t("none"),           label="Autres crédits en cours")
                existing    = gr.Slider(1, 4, value=1, step=1,                                             label="Nombre de crédits existants")

            with gr.Group():
                gr.HTML('<p style="font-size:10px;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:1.2px;margin:8px 0 12px;">'
                        'Demande de crédit</p>')
                purpose     = gr.Dropdown(opts(CAT_RAW["purpose"]), value=t("new car"), label="Objet du crédit")
                amount      = gr.Slider(250, 18000, value=3000, step=100,               label="Montant demandé (Deutschmarks)")
                duration    = gr.Slider(4, 72, value=24, step=1,                        label="Durée de remboursement (mois)")
                installment = gr.Slider(1, 4, value=3, step=1,                          label="Niveau d'endettement — 1 (faible) à 4 (élevé)")

            with gr.Group():
                gr.HTML('<p style="font-size:10px;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:1.2px;margin:8px 0 12px;">'
                        'Profil personnel</p>')
                age         = gr.Slider(18, 80, value=35, step=1,                        label="Age")
                personal    = gr.Dropdown(opts(CAT_RAW["personal_status"]),  value=t("male single"), label="Situation personnelle")
                employment  = gr.Dropdown(opts(CAT_RAW["employment"]),       value=t("1<=X<4"),      label="Ancienneté dans l'emploi actuel")
                job         = gr.Dropdown(opts(CAT_RAW["job"]),              value=t("skilled"),     label="Catégorie professionnelle")
                dependents  = gr.Slider(1, 2, value=1, step=1,                                       label="Personnes à charge")

            with gr.Group():
                gr.HTML('<p style="font-size:10px;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:1.2px;margin:8px 0 12px;">'
                        'Patrimoine et logement</p>')
                housing       = gr.Dropdown(opts(CAT_RAW["housing"]),          value=t("rent"),  label="Statut de logement")
                property_mag  = gr.Dropdown(opts(CAT_RAW["property_magnitude"]),value=t("car"),  label="Bien principal possédé")
                residence     = gr.Slider(1, 4, value=2, step=1,                                  label="Ancienneté à l'adresse actuelle (années)")
                other_parties = gr.Dropdown(opts(CAT_RAW["other_parties"]),    value=t("none"),  label="Co-emprunteur ou garant")
                telephone     = gr.Dropdown(opts(CAT_RAW["own_telephone"]),    value=t("yes"),   label="Téléphone fixe enregistré")
                foreign       = gr.Dropdown(opts(CAT_RAW["foreign_worker"]),   value=t("yes"),   label="Travailleur étranger")

            btn = gr.Button("Analyser la demande", variant="primary", size="lg")

        # ── Résultats ─────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=400):

            result_html = gr.HTML("""
            <div style="padding:20px;background:#f8fafc;border:1px solid #e2e8f0;
                        border-radius:10px;color:#64748b;font-size:13px;">
              Remplissez le formulaire et cliquez sur <strong style="color:#1e293b;">
              Analyser la demande</strong> pour obtenir la décision et les explications.
            </div>
            """)

            with gr.Tabs():
                with gr.Tab("Explication SHAP"):
                    shap_plot = gr.Plot(show_label=False)
                with gr.Tab("Explication LIME"):
                    lime_plot = gr.Plot(show_label=False)

            gr.HTML("""
            <div style="margin-top:16px;padding:14px 18px;background:#f8fafc;
                        border:1px solid #e2e8f0;border-radius:8px;line-height:1.7;">
              <p style="margin:0 0 8px;font-size:10px;font-weight:600;color:#9ca3af;
                        text-transform:uppercase;letter-spacing:1px;">
                Méthodes d'explicabilité
              </p>
              <p style="margin:0 0 6px;font-size:12px;color:#374151;">
                <strong style="color:#1e293b;">SHAP</strong> — mesure la contribution de chaque
                variable à la décision, fondé sur la théorie des jeux coopératifs (Shapley, 1953).
              </p>
              <p style="margin:0;font-size:12px;color:#374151;">
                <strong style="color:#1e293b;">LIME</strong> — construit une approximation
                linéaire locale autour du profil analysé pour identifier les facteurs clés.
              </p>
            </div>
            """)

    inputs = [
        checking, duration, cred_hist, purpose, amount, savings,
        employment, installment, personal, other_parties, residence,
        property_mag, age, other_plans, housing, existing, job,
        dependents, telephone, foreign,
    ]
    btn.click(fn=predict_and_explain, inputs=inputs,
              outputs=[result_html, shap_plot, lime_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
