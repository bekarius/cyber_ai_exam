from pathlib import Path
import pandas as pd

def load_dataset(path: Path):
    df = pd.read_csv(path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    target = _detect_target(df)
    y = _normalize_target(df[target])
    # numeric-only features (keep only needed later)
    X = df.select_dtypes(include=["number"])
    return X, y, df

def _detect_target(df: pd.DataFrame) -> str:
    cols = [c.strip().lower() for c in df.columns]
    if "is_spam" in cols:
        return df.columns[cols.index("is_spam")]
    if "label" in cols:
        return df.columns[cols.index("label")]
    raise ValueError("Expected target column 'is_spam' or 'Label' in dataset.")

def _normalize_target(series: pd.Series):
    # Support 'spam'/'legitimate' or 0/1
    if series.dtype == "O":
        return series.astype(str).str.lower().map({"spam": 1, "legitimate": 0}).values
    return series.values

def select_feature_columns(df: pd.DataFrame):
    # Expected feature names used for raw email extraction
    preferred = ["words", "links", "capital_words", "spam_word_count"]
    available = [c for c in preferred if c in df.columns]
    if len(available) == 0:
        # Fallback: all numeric except target
        target = _detect_target(df)
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if target in numeric:
            numeric.remove(target)
        return numeric
    return available
