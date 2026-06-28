"""信号フィルタ — One Euro Filter

`gaze_estimator` と `head_pose_estimator` の両方から使われるため、独立モジュールに
切り出して循環インポートを避ける（両者がこのモジュールを参照するだけにする）。
"""

from __future__ import annotations

import math
import time
from typing import Optional


class OneEuroFilter:
    """One Euro Filter — ジッター除去と遅延のバランスを自動調整するフィルタ

    静止時はカットオフ周波数を下げてノイズを強く除去し、
    素早い動きにはカットオフを上げて遅延なく追従する。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff  # 最小カットオフ周波数（低い→安定、高い→反応早い）
        self._beta = beta              # 速度による感度（大きい→動きに素早く反応）
        self._d_cutoff = d_cutoff      # 微分のカットオフ
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()

        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1.0 / 30.0  # フォールバック: 30fps想定

        # 微分値のフィルタリング
        dx = (x - self._x_prev) / dt
        a_d = self._smoothing_factor(dt, self._d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # 適応的カットオフ: 速度が大きいほどカットオフを上げる
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)

        # 値のフィルタリング
        a = self._smoothing_factor(dt, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None
