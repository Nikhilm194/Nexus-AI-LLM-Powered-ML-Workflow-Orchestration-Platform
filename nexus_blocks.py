"""
nexus_blocks.py
===============
Isolated, reusable ML pipeline blocks for the Astrikos framework.

Each function is a self-contained "block" that can be composed into
a pipeline via drag-and-drop or JSON configuration.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# ── Block 1: Data Ingestion ──────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return it as a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    print(f"[BLOCK] load_csv  ->  Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"[BLOCK] load_csv  ->  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"[BLOCK] load_csv  ->  Columns: {list(df.columns)}")
    print(f"[BLOCK] load_csv  ->  Missing values per column:\n{df.isna().sum().to_string()}")
    return df


# ── Block 2: Data Preprocessing (Imputation) ────────────────────────

def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
) -> pd.DataFrame:
    """Fill NaN values in all numeric columns using the given strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain NaNs).
    strategy : str, optional
        Imputation strategy — one of 'mean', 'median', 'most_frequent',
        or 'constant'.  Defaults to 'mean'.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing numeric values imputed.
    """
    print(f"[BLOCK] impute_missing_values  ->  Strategy: '{strategy}'")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    nan_before = df[numeric_cols].isna().sum().sum()
    print(f"[BLOCK] impute_missing_values  ->  NaNs before imputation: {nan_before}")

    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    nan_after = df[numeric_cols].isna().sum().sum()
    print(f"[BLOCK] impute_missing_values  ->  NaNs after imputation:  {nan_after}")
    return df


# ── Block 3: Model Building ─────────────────────────────────────────

def train_random_forest(
    df: pd.DataFrame,
    target_col: str,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the provided DataFrame.

    All numeric columns except *target_col* are used as features.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame (no NaNs expected in numeric columns).
    target_col : str
        Name of the column to predict.

    Returns
    -------
    RandomForestRegressor
        The fitted model.
    """
    print(f"[BLOCK] train_random_forest  ->  Target column: '{target_col}'")

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    X = df[feature_cols]
    y = df[target_col]

    print(f"[BLOCK] train_random_forest  ->  Features: {feature_cols}")
    print(f"[BLOCK] train_random_forest  ->  Training samples: {len(X)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    score = model.score(X, y)
    print(f"[BLOCK] train_random_forest  ->  R^2 score (train): {score:.4f}")
    print(f"[BLOCK] train_random_forest  ->  Model trained successfully")
    return model
