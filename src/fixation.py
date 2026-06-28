"""注視（フィキセーション）ベースのターゲット確定 — ロック＆エスケープ方式

「常に視線を追う」のではなく、視線が落ち着いた場所に**ロック**し、その近傍に
視線がいる限りカーソルを**完全に静止**させる。明確に遠く（release_radius の外）を
dwell_time 見続けて初めて、その場所へ乗り換える。

これにより:
  * ロック中は頭の微ドリフトや視線ノイズを完全に無視 → カーソルがピタッと定まる
  * 重心の少しの揺れで再確定し続ける「じわじわ移動」が起きない
  * 履歴の重心で乗り換え先を決めるので、毎回ピタリ見続けなくてよい

mediapipe等に依存しない純粋ロジックで、ヘッドレスにユニットテストできる。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

from . import config


class FixationTracker:
    """視線をロックし、明確な視線移動があったときだけ乗り換える状態機械。"""

    def __init__(
        self,
        dwell_time: float = config.FIXATION_DWELL_TIME,
        radius: float = config.FIXATION_RADIUS,
        release_radius: float = config.FIXATION_RELEASE_RADIUS,
    ) -> None:
        self._dwell_time = max(0.05, dwell_time)
        self._radius = radius
        self._release_radius = release_radius
        # 履歴は dwell_time ぶんだけ保持（乗り換え判定の窓）
        self._buf: Deque[Tuple[float, float, float]] = deque()
        self._locked: Optional[Tuple[float, float]] = None

    @property
    def target(self) -> Optional[Tuple[float, float]]:
        return self._locked

    def update(self, x: float, y: float, t: float) -> Tuple[float, float, bool]:
        """生の視線座標と時刻から、ロック中のターゲット (tx, ty, just_committed) を返す。"""
        self._buf.append((x, y, t))
        while self._buf and (t - self._buf[0][2]) > self._dwell_time:
            self._buf.popleft()

        # 初回はその場にロック
        if self._locked is None:
            self._locked = (x, y)
            return (x, y, True)

        lx, ly = self._locked
        # ロック近傍にいる限り完全静止（ブレ・ドリフトを無視）
        if math.hypot(x - lx, y - ly) <= self._release_radius:
            return (lx, ly, False)

        # ロック圏外: 遠方で落ち着いた注視ができたら乗り換え
        span = t - self._buf[0][2]
        if len(self._buf) >= 3 and span >= self._dwell_time * 0.8:
            cx = sum(p[0] for p in self._buf) / len(self._buf)
            cy = sum(p[1] for p in self._buf) / len(self._buf)
            spread = max(math.hypot(px - cx, py - cy) for (px, py, _) in self._buf)
            if spread <= self._radius and math.hypot(cx - lx, cy - ly) > self._release_radius:
                self._locked = (cx, cy)
                return (cx, cy, True)

        # 乗り換え条件を満たさない間はロックを保持
        return (lx, ly, False)

    def reset(self) -> None:
        self._buf.clear()
        self._locked = None
