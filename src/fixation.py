"""注視（フィキセーション）ベースのターゲット確定 — 履歴参照 + ドウェル

「常に視線を追う」のではなく、視線がある場所に dwell_time 秒とどまったら、
その滞留中の**履歴の重心**を確定ターゲットとして返す。視線移動中（サッケード）や
ノイズの間はターゲットを更新せず保持するため、

  * カーソルは目的地に着いたらそこに留まる（途中の距離情報を追わない＝情報を削る）
  * 重心を使うので毎回ピタリと同じ点を見続けなくてよい（履歴参照で楽）
  * 新しい場所を dwell_time 見続けて初めて移る（自然なディレイ 0.2〜0.5秒）

mediapipe等に依存しない純粋ロジックで、ヘッドレスにユニットテストできる。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

from . import config


class FixationTracker:
    """視線の滞留を検出して確定ターゲットを返す状態機械。"""

    def __init__(
        self,
        dwell_time: float = config.FIXATION_DWELL_TIME,
        radius: float = config.FIXATION_RADIUS,
        min_move: float = config.FIXATION_MIN_MOVE,
    ) -> None:
        self._dwell_time = max(0.05, dwell_time)
        self._radius = radius
        self._min_move = min_move
        # 履歴は dwell_time ぶんだけ保持（この窓＝注視判定窓）
        self._buf: Deque[Tuple[float, float, float]] = deque()
        self._target: Optional[Tuple[float, float]] = None

    @property
    def target(self) -> Optional[Tuple[float, float]]:
        return self._target

    def update(self, x: float, y: float, t: float) -> Tuple[float, float, bool]:
        """生の視線座標と時刻から、確定ターゲット (tx, ty, just_committed) を返す。

        just_committed は今回のフレームで新しい注視が確定したとき True。
        """
        self._buf.append((x, y, t))
        # dwell_time より古いサンプルを捨てる（情報を削る）
        while self._buf and (t - self._buf[0][2]) > self._dwell_time:
            self._buf.popleft()

        # 初回はその場をターゲットに（カーソルに初期位置を与える）
        if self._target is None:
            self._target = (x, y)
            return (x, y, True)

        committed = False
        span = t - self._buf[0][2]
        # 窓が dwell_time をほぼ満たし、十分なサンプルがあるときのみ判定
        if len(self._buf) >= 3 and span >= self._dwell_time * 0.8:
            cx = sum(p[0] for p in self._buf) / len(self._buf)
            cy = sum(p[1] for p in self._buf) / len(self._buf)
            # ばらつき（重心からの最大距離）
            spread = max(math.hypot(px - cx, py - cy) for (px, py, _) in self._buf)
            if spread <= self._radius:
                # 視線が落ち着いている＝注視。重心へ確定（微小な再確定は無視）
                if math.hypot(cx - self._target[0], cy - self._target[1]) >= self._min_move:
                    self._target = (cx, cy)
                    committed = True

        return (self._target[0], self._target[1], committed)

    def reset(self) -> None:
        self._buf.clear()
        self._target = None
