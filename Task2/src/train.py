from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

def train_logreg(df, feature_cols, target_col="is_spam"):
    X = df[feature_cols].values
    y = _to_binary(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Holdout evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    info = {
        "coef_": clf.coef_[0],
        "intercept_": clf.intercept_,
        "holdout_accuracy": acc,
        "holdout_confusion": cm,
        "dump": lambda path: _save_model(clf, path),
    }
    return clf, info

def _save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)

def _to_binary(series):
    if series.dtype == "O":
        return series.astype(str).str.lower().map({"spam": 1, "legitimate": 0}).values
    return series.values
