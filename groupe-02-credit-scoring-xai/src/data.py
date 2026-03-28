"""
data.py — Chargement et préparation du German Credit Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Charge le German Credit Dataset depuis OpenML.

    Returns:
        X : DataFrame des features (1000 lignes, 20 colonnes)
        y : Series binaire — 1 = bon client, 0 = mauvais client
    """
    print("Chargement du German Credit Dataset (OpenML)...")
    dataset = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    df["target"] = (df["class"] == "good").astype(int)
    df = df.drop(columns=["class"])

    X = df.drop(columns=["target"])
    y = df["target"]

    print(f"  {len(X)} demandes | {y.mean():.0%} bons clients | {X.shape[1]} features")
    return X, y


def get_column_types(X: pd.DataFrame) -> tuple[list, list]:
    """Identifie les colonnes numériques et catégorielles."""
    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical, categorical


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Construit le pipeline de prétraitement scikit-learn."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="passthrough",
    )


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> dict:
    """Split train/test + fit du préprocesseur.

    Returns:
        dict avec X_train, X_test, y_train, y_test (DataFrames originaux),
        X_train_proc, X_test_proc (arrays prétraités),
        preprocessor, feature_names, numerical_cols, categorical_cols
    """
    numerical_cols, categorical_cols = get_column_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    cat_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    feature_names = numerical_cols + list(cat_names)

    print(f"  Train : {len(X_train)} | Test : {len(X_test)} | Features encodées : {len(feature_names)}")

    return dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X_train_proc=X_train_proc, X_test_proc=X_test_proc,
        preprocessor=preprocessor, feature_names=feature_names,
        numerical_cols=numerical_cols, categorical_cols=categorical_cols,
    )
