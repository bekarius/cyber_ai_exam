#!/usr/bin/env python3
import argparse
import os
from joblib import dump, load
import numpy as np

from src.train import train_logreg
from src.evaluate import evaluate_model, save_confusion_matrix_plot
from src.features import extract_features_from_email_text
from src.visualize import class_balance_plot, top2_scatter_by_corr
from src.utils import load_dataset, select_feature_columns


def main():
    parser = argparse.ArgumentParser(description="Spam vs Legit Email - Logistic Regression")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train logistic regression on 70% split")
    p_train.add_argument("--data", required=True, help="Path to CSV")
    p_train.add_argument("--model", default="models/logreg.joblib", help="Output model path")

    # Evaluate
    p_eval = sub.add_parser("eval", help="Evaluate trained model on holdout")
    p_eval.add_argument("--model", required=True, help="Path to saved model")
    p_eval.add_argument("--data", required=True, help="Path to CSV")

    # Predict
    p_pred = sub.add_parser("predict", help="Predict spam probability for raw email text")
    p_pred.add_argument("--model", required=True, help="Path to saved model")
    p_pred.add_argument("--data", required=True, help="CSV (for schema)")
    p_pred.add_argument("--email", required=True, help="Path to a .txt email")

    # Visualizations
    p_viz = sub.add_parser("viz", help="Generate visualizations")
    p_viz.add_argument("--data", required=True, help="Path to CSV")
    p_viz.add_argument("--outdir", default="figs", help="Output directory for figures")

    args = parser.parse_args()

    if args.cmd == "train":
        X, y, df = load_dataset(args.data)
        feature_cols = select_feature_columns(df)
        target = info_get_target(df)

        model, info = train_logreg(df, feature_cols, target_col=target)

        # Ensure model directory exists
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        dump(model, args.model)

        print("\n=== Training complete ===")
        print(f"Used features: {feature_cols}")
        print(f"Target column: {target}")
        if "coef_" in info:
            print(f"Coefficients: {getattr(info['coef_'], 'tolist', lambda: info['coef_'])()}")
        if "intercept_" in info:
            print(f"Intercept: {getattr(info['intercept_'], 'tolist', lambda: info['intercept_'])()}")
        print(f"Model saved to: {args.model}")
        if "holdout_accuracy" in info:
            print(f"Holdout Accuracy: {info['holdout_accuracy']:.4f}")
        if "holdout_confusion" in info:
            print("Confusion Matrix:")
            print(info["holdout_confusion"])

    elif args.cmd == "eval":
        X, y, df = load_dataset(args.data)
        feature_cols = select_feature_columns(df)
        acc, cm = evaluate_model(args.model, df[feature_cols], y)
        print("\n=== Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        os.makedirs("figs", exist_ok=True)
        save_confusion_matrix_plot(cm, ["legitimate", "spam"], "figs/confusion_matrix.png")
        print("Saved: figs/confusion_matrix.png")

    elif args.cmd == "predict":
        _, _, df = load_dataset(args.data)
        feature_cols = select_feature_columns(df)
        with open(args.email, "r", encoding="utf-8", errors="ignore") as f:
            email_text = f.read()
        feats = extract_features_from_email_text(email_text)
        x = [feats.get(c, 0.0) for c in feature_cols]
        model = load(args.model)
        prob = model.predict_proba(np.array([x]))[0, 1]
        label = "spam" if prob >= 0.5 else "legitimate"
        print("=== Email prediction ===")
        print(f"Probability(spam) = {prob:.4f} -> class = {label}")

    elif args.cmd == "viz":
        X, y, df = load_dataset(args.data)
        os.makedirs(args.outdir, exist_ok=True)
        target = info_get_target(df)
        class_balance_plot(df, target_col=target, outpath=os.path.join(args.outdir, "class_balance.png"))
        top2_scatter_by_corr(df, target_col=target, outpath=os.path.join(args.outdir, "top2_scatter.png"))
        print(f"Saved visualizations in: {args.outdir}")


def info_get_target(df):
    cols = [c.strip() for c in df.columns]
    if "is_spam" in cols:
        return "is_spam"
    if "Label" in cols:
        return "Label"
    for c in df.columns:
        if c.strip().lower() in ("is_spam", "label"):
            return c
    raise ValueError("Target column not found. Expected 'is_spam' or 'Label'.")


if __name__ == "__main__":
    main()
