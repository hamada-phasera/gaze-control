"""ジェスチャ検出 — 閉眼判定と「長い閉眼でリセット」トリガー

mediapipe 等に依存しない純粋ロジックでユニットテストできる。
"""

from __future__ import annotations

from typing import Optional

from . import config


def is_eye_closed(
    left_ear: float, right_ear: float, blink_score: Optional[float] = None
) -> bool:
    """閉眼かどうか。blendshape の瞬きスコアがあれば最優先、無ければ EAR。"""
    if blink_score is not None and config.USE_BLENDSHAPE_BLINK:
        return blink_score >= config.BLINK_BLENDSHAPE_THRESHOLD
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear < config.BLINK_EAR_THRESHOLD


class EyesClosedTrigger:
    """目を duration 秒だけ閉じ続けたら一度だけ True を返すトリガー。

    目を開けるとリセットされ、再び閉じれば次回また発火できる。
    """

    def __init__(self, duration: float = config.EYES_CLOSED_RESET_SEC) -> None:
        self._duration = duration
        self._closed_since: Optional[float] = None
        self._fired = False

    def update(self, is_closed: bool, t: float) -> bool:
        """閉眼状態と時刻から、発火タイミングで True を返す。"""
        if is_closed:
            if self._closed_since is None:
                self._closed_since = t
                self._fired = False
            elif not self._fired and (t - self._closed_since) >= self._duration:
                self._fired = True
                return True
        else:
            self._closed_since = None
            self._fired = False
        return False

    def progress(self, t: float) -> float:
        """現在の閉眼継続の進捗 (0〜1)。閉じていなければ 0。"""
        if self._closed_since is None:
            return 0.0
        return min(1.0, (t - self._closed_since) / self._duration)

    def reset(self) -> None:
        self._closed_since = None
        self._fired = False
