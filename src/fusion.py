"""視線+頭部姿勢の加重融合モジュール"""

import math
from typing import Tuple

from . import config


class GazeFusion:
    """視線位置と頭部姿勢位置の加重融合

    サッケード（素早い視線移動）時は視線を優先し、
    静止時は頭部の微調整を優先する速度適応型の融合を行う。
    """

    def __init__(
        self,
        w_gaze_min: float = config.FUSION_W_GAZE_MIN,
        w_gaze_max: float = config.FUSION_W_GAZE_MAX,
        saccade_threshold: float = config.FUSION_SACCADE_THRESHOLD,
    ) -> None:
        self._w_gaze_min = w_gaze_min
        self._w_gaze_max = w_gaze_max
        self._saccade_threshold = saccade_threshold

        # 前回の視線位置（速度計算用）
        self._prev_gaze_x: float | None = None
        self._prev_gaze_y: float | None = None
        self._prev_time: float | None = None

    def compute(
        self,
        gaze_x: float,
        gaze_y: float,
        head_x: float,
        head_y: float,
        t: float,
    ) -> Tuple[float, float, float, float]:
        """視線と頭部の加重融合を計算する

        Args:
            gaze_x, gaze_y: 視線ベースの画面座標
            head_x, head_y: 頭部姿勢ベースの画面座標
            t: タイムスタンプ (秒)

        Returns:
            (fused_x, fused_y, w_gaze, w_head) 融合後の座標と重み
        """
        gaze_velocity = self._compute_gaze_velocity(gaze_x, gaze_y, t)

        # 速度に基づいて重みを適応的に計算
        w_gaze = self._compute_gaze_weight(gaze_velocity)
        w_head = 1.0 - w_gaze

        fused_x = w_gaze * gaze_x + w_head * head_x
        fused_y = w_gaze * gaze_y + w_head * head_y

        return (fused_x, fused_y, w_gaze, w_head)

    def compute_precision(
        self,
        anchor_x: float,
        anchor_y: float,
        head_delta_x: float,
        head_delta_y: float,
    ) -> Tuple[float, float]:
        """精密モード: アンカーポイント + 頭部オフセット

        Args:
            anchor_x, anchor_y: 精密モード開始時の固定位置
            head_delta_x, head_delta_y: 頭部姿勢のオフセット (px)

        Returns:
            (x, y) 精密モードの座標
        """
        return (anchor_x + head_delta_x, anchor_y + head_delta_y)

    def reset(self) -> None:
        """内部状態をリセットする"""
        self._prev_gaze_x = None
        self._prev_gaze_y = None
        self._prev_time = None

    def _compute_gaze_velocity(self, gaze_x: float, gaze_y: float, t: float) -> float:
        """視線の速度を計算する (px/s)"""
        if self._prev_gaze_x is None or self._prev_time is None:
            self._prev_gaze_x = gaze_x
            self._prev_gaze_y = gaze_y
            self._prev_time = t
            return 0.0

        dt = t - self._prev_time
        if dt <= 0:
            dt = 1.0 / 30.0

        dx = gaze_x - self._prev_gaze_x
        dy = gaze_y - self._prev_gaze_y
        velocity = math.sqrt(dx * dx + dy * dy) / dt

        self._prev_gaze_x = gaze_x
        self._prev_gaze_y = gaze_y
        self._prev_time = t

        return velocity

    def _compute_gaze_weight(self, gaze_velocity: float) -> float:
        """速度に基づいて視線の重みを計算する"""
        if gaze_velocity >= self._saccade_threshold:
            return self._w_gaze_max

        t = gaze_velocity / self._saccade_threshold
        return self._w_gaze_min + t * (self._w_gaze_max - self._w_gaze_min)
