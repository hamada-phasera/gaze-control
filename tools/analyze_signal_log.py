#!/usr/bin/env python3
"""視線信号ログ(CSV)の診断スクリプト。

`--log-signals out.csv` で記録したCSVを読み、各信号の範囲・分散や、
生の視線比率(iris)と最終出力(x/y)の相関を出す。これで例えば

  - iris_y の分散がほぼ0 → 縦信号が動いていない（センサ/幾何の問題）
  - iris_y は動くが y との相関が低い → キャリブのマッピングが潰れている

のように「どこで壊れているか」を切り分けられる。numpy だけで動く。

使い方:
    python tools/analyze_signal_log.py out.csv
"""

from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np


def load_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows: List[List[str]] = [ln.strip().split(",") for ln in f if ln.strip()]
    cols: Dict[str, np.ndarray] = {}
    for i, name in enumerate(header):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[i]) if r[i] != "" else np.nan)
            except (ValueError, IndexError):
                vals.append(np.nan)
        cols[name] = np.array(vals, dtype=float)
    return cols


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: python tools/analyze_signal_log.py <log.csv>")
        sys.exit(1)

    cols = load_csv(sys.argv[1])
    n = len(next(iter(cols.values()))) if cols else 0
    dur = float(np.nanmax(cols["t"])) if "t" in cols and n else 0.0
    print(f"=== {sys.argv[1]} : {n} フレーム / {dur:.1f} 秒 ===\n")

    print(f"{'列':<12}{'min':>10}{'max':>10}{'range':>10}{'std':>10}")
    for name in ("iris_x", "iris_y", "x", "y", "head_yaw", "head_pitch", "distance_cm"):
        if name not in cols:
            continue
        v = cols[name]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rng = float(np.max(v) - np.min(v))
        print(f"{name:<12}{np.min(v):>10.3f}{np.max(v):>10.3f}{rng:>10.3f}{np.std(v):>10.4f}")

    print("\n--- 診断 ---")
    # 縦信号が動いているか
    if "iris_y" in cols:
        sy = float(np.nanstd(cols["iris_y"]))
        if sy < 0.01:
            print(f"⚠ iris_y の分散が非常に小さい (std={sy:.4f}) — 縦の視線信号がほぼ動いていない")
        else:
            print(f"✓ iris_y は動いている (std={sy:.4f})")
    # 生信号と出力の相関（マッピング健全性）
    if "iris_x" in cols and "x" in cols:
        print(f"  corr(iris_x, x) = {_corr(cols['iris_x'], cols['x']):.3f}  (高いほど横マッピング良)")
    if "iris_y" in cols and "y" in cols:
        cy = _corr(cols["iris_y"], cols["y"])
        print(f"  corr(iris_y, y) = {cy:.3f}  (高いほど縦マッピング良)")
        if np.isfinite(cy) and abs(cy) < 0.3:
            print("⚠ 縦の相関が低い — 縦キャリブのマッピングが弱い/潰れている可能性")
    # 頭の動きとの相関（ドリフトの兆候）
    if "iris_x" in cols and "head_yaw" in cols:
        print(f"  corr(x, head_yaw) = {_corr(cols['x'], cols['head_yaw']):.3f}  (高いと頭ヨーに釣られている)")
    if "y" in cols and "head_pitch" in cols:
        print(f"  corr(y, head_pitch) = {_corr(cols['y'], cols['head_pitch']):.3f}  (高いと頭ピッチに釣られている)")


if __name__ == "__main__":
    main()
