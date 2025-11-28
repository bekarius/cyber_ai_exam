#!/usr/bin/env python3
"""
Compute Pearson correlation for the given points and save a scatter plot.

Usage:
  python3 compute_correlation.py --out figs/correlation_scatter.png
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = math.sqrt(((x - xm)**2).sum() * ((y - ym)**2).sum())
    return num / den

def fisher_p_two_tailed(r: float, n: int) -> float:
    """Approximate two-tailed p-value via t distribution."""
    from math import sqrt
    import mpmath as mp
    df = n - 2
    t = r * sqrt(df / (1 - r*r))
    # two-tailed p from Student's t
    p = 2 * (1 - mp.gammainc((df+1)/2, 0, (df/(df+t*t))) /
             (mp.sqrt(mp.pi*df) * mp.gamma(df/2)))
    # The above is a compact expression; if mpmath isn’t available, omit p.
    return float(p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figs/correlation_scatter.png",
                        help="Path to save scatter plot (PNG)")
    args = parser.parse_args()

    # Data
    pts = [(-5,2), (-1,1), (-3,-1), (-4,4), (1,-2), (3,1), (5,-3), (7,-2)]
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    n = x.size

    # Pearson r (manual)
    r = pearson_r(x, y)

    # Trendline (least squares)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min()-1, x.max()+1, 100)
    y_line = slope * x_line + intercept

    # Plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, label="Data points")
    plt.plot(x_line, y_line, label="Least-squares line")
    plt.title(f"Scatter (r = {r:.3f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Pearson r = {r:.4f}")
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
