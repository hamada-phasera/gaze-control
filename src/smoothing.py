"""動き安定化 — 視線座標を間引き・注視保持して「ぬるっと」した追従を作る

リアルタイムに座標を流すと、推定の微小なラグ・ノイズでカーソルが落ち着かない。
本モジュールは次の3段で安定化する:

  1. 注視デッドゾーン: 目標近傍の微小な揺れは無視して注視点を保持する
  2. サッケード即追従: 大きな視線移動はそのまま追従する（素早い移動を妨げない）
  3. 間引き確定: 中間的な移動は一定間隔でのみ目標を更新し、フレームを「削る」

確定した目標へは呼び出し側（オーバーレイの `exp_smooth` 等）が滑らかに寄せるため、
全体として遅れて滑る「ぬるっと」した質感になる。

Qt・カメラに依存しない純粋ロジックで、ヘッドレスにユニットテストできる。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from . import config


class MotionStabilizer:
    """視線座標を安定化する状態機械。`update(x, y, t)` で安定化座標を返す。"""

    def __init__(
        self,
        deadzone: float = config.STAB_DEADZONE,
        saccade: float = config.STAB_SACCADE,
        commit_interval: float = config.STAB_COMMIT_INTERVAL,
        commit_ratio: float = config.STAB_COMMIT_RATIO,
    ) -> None:
        self._deadzone = deadzone
        self._saccade = saccade
        self._commit_interval = commit_interval
        self._commit_ratio = commit_ratio

        self._tx: Optional[float] = None
        self._ty: Optional[float] = None
        self._last_commit: float = 0.0

    def update(self, x: float, y: float, t: float) -> Tuple[float, float]:
        """生の視線座標 (x, y) と時刻 t から、安定化した目標座標を返す。"""
        if self._tx is None:
            self._tx, self._ty = x, y
            self._last_commit = t
            return (self._tx, self._ty)

        dx = x - self._tx
        dy = y - self._ty
        dist = math.hypot(dx, dy)

        if dist >= self._saccade:
            # 大きく動いた → 即追従（素早い視線移動を妨げない）
            self._tx, self._ty = x, y
            self._last_commit = t
        elif dist <= self._deadzone:
            # 微小な揺れ → 注視点を保持（ブレを削る）
            pass
        elif (t - self._last_commit) >= self._commit_interval:
            # 中間移動 → 間隔を空けて部分的にのみ確定（データを間引く）
            self._tx += self._commit_ratio * dx
            self._ty += self._commit_ratio * dy
            self._last_commit = t

        return (self._tx, self._ty)

    @property
    def target(self) -> Optional[Tuple[float, float]]:
        if self._tx is None:
            return None
        return (self._tx, self._ty)

    def reset(self) -> None:
        self._tx = None
        self._ty = None
        self._last_commit = 0.0
