from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def class_balance_plot(df: pd.DataFrame, target_col: str, outpath: Path):
    counts = df[target_col].astype(str).str.lower().replace({
        "1":"spam","0":"legitimate"
    }).value_counts()
    plt.figure(figsize=(4.8,4))
    counts.plot(kind="bar")
    plt.title("Class Balance")
    plt.ylabel("Count")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()

def top2_scatter_by_corr(df: pd.DataFrame, target_col: str, outpath: Path):
    # numeric only
    num = df.select_dtypes(include=["number"]).copy()
    if target_col in num.columns:
        y = num[target_col].values
        num = num.drop(columns=[target_col])
    else:
        y = df[target_col].astype(str).str.lower().map({"spam":1,"legitimate":0}).values

    if num.shape[1] < 2:
        return

    # correlate with target
    corr = []
    for c in num.columns:
        x = num[c].fillna(0).values
        if np.std(x) == 0:
            continue
        r = np.corrcoef(x, y)[0,1]
        corr.append((c, abs(r)))
    corr.sort(key=lambda t: t[1], reverse=True)
    if len(corr) < 2:
        return
    c1, c2 = corr[0][0], corr[1][0]

    plt.figure(figsize=(5,4))
    plt.scatter(num[c1], num[c2], s=8, alpha=0.6)
    plt.xlabel(c1); plt.ylabel(c2)
    plt.title("Top-2 Correlated Feature Scatter")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
