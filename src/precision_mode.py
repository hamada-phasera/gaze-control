"""精密モード検出モジュール — 眉上げジェスチャーによるトリガー"""

import time
from typing import Optional

from . import config


class PrecisionModeDetector:
    """眉上げジェスチャーで精密モードを切り替える

    両眉と目上端の距離比を監視し、閾値を超えた状態が
    一定時間持続したら精密モードに遷移する。
    ヒステリシスにより誤検出を防止する。
    """

    def __init__(
        self,
        brow_threshold: float = config.PRECISION_MODE_BROW_THRESHOLD,
        enter_duration: float = config.PRECISION_MODE_ENTER_DURATION,
        exit_duration: float = config.PRECISION_MODE_EXIT_DURATION,
    ) -> None:
        self._brow_threshold = brow_threshold
        self._enter_duration = enter_duration
        self._exit_duration = exit_duration

        self._is_active = False
        self._brow_raised_since: Optional[float] = None
        self._brow_lowered_since: Optional[float] = None

        # ベースライン（通常の眉-目間距離）
        self._baseline_ratio: Optional[float] = None
        self._baseline_samples: list[float] = []

    @property
    def is_active(self) -> bool:
        return self._is_active

    def update(
        self,
        landmarks: object,
        brow_score: Optional[float] = None,
        t: Optional[float] = None,
    ) -> bool:
        """ランドマーク／blendshapeから精密モード状態を更新する

        Args:
            landmarks: MediaPipe のランドマーク（幾何フォールバック時に使用）
            brow_score: blendshape の眉上げスコア(0〜1)。あれば最優先で使う
            t: タイムスタンプ(秒)。None なら time.time()

        Returns:
            True if precision mode is active
        """
        now = t if t is not None else time.time()

        if brow_score is not None and config.USE_BLENDSHAPE_BROW:
            # blendshape優先: 正規化済みスコアなのでベースライン不要
            is_raised = brow_score >= config.PRECISION_MODE_BLENDSHAPE_THRESHOLD
        else:
            # 幾何フォールバック: 眉-目間距離をベースラインと比較
            ratio = self._compute_brow_raise_ratio(landmarks)

            # ベースライン収集（最初の30フレーム分）
            if self._baseline_ratio is None:
                self._baseline_samples.append(ratio)
                if len(self._baseline_samples) >= 30:
                    self._baseline_ratio = sum(self._baseline_samples) / len(self._baseline_samples)
                    self._baseline_samples.clear()
                return False

            # 眉が上がっているかの判定（ベースラインからの差分）
            is_raised = (ratio - self._baseline_ratio) > self._brow_threshold

        if not self._is_active:
            # 非アクティブ → アクティブへの遷移
            if is_raised:
                if self._brow_raised_since is None:
                    self._brow_raised_since = now
                elif now - self._brow_raised_since >= self._enter_duration:
                    self._is_active = True
                    self._brow_raised_since = None
                    self._brow_lowered_since = None
            else:
                self._brow_raised_since = None
        else:
            # アクティブ → 非アクティブへの遷移
            if not is_raised:
                if self._brow_lowered_since is None:
                    self._brow_lowered_since = now
                elif now - self._brow_lowered_since >= self._exit_duration:
                    self._is_active = False
                    self._brow_lowered_since = None
                    self._brow_raised_since = None
            else:
                self._brow_lowered_since = None

        return self._is_active

    def reset(self) -> None:
        """状態をリセットする"""
        self._is_active = False
        self._brow_raised_since = None
        self._brow_lowered_since = None
        self._baseline_ratio = None
        self._baseline_samples.clear()

    def _compute_brow_raise_ratio(self, landmarks: object) -> float:
        """眉と目上端の距離比を計算する

        眉ランドマークのY座標平均と目上端のY座標の差を、
        目の幅で正規化して返す。値が大きいほど眉が上がっている。
        """
        lm = landmarks.landmark

        # 左眉のY座標平均
        left_brow_y = sum(lm[i].y for i in config.LEFT_EYEBROW_INDICES) / len(
            config.LEFT_EYEBROW_INDICES
        )
        # 右眉のY座標平均
        right_brow_y = sum(lm[i].y for i in config.RIGHT_EYEBROW_INDICES) / len(
            config.RIGHT_EYEBROW_INDICES
        )

        # 目上端のY座標
        left_eye_top_y = lm[config.LEFT_EYE_TOP].y
        right_eye_top_y = lm[config.RIGHT_EYE_TOP].y

        # 目の幅（正規化用）
        left_eye_width = abs(lm[config.LEFT_EYE_INNER].x - lm[config.LEFT_EYE_OUTER].x)
        right_eye_width = abs(lm[config.RIGHT_EYE_INNER].x - lm[config.RIGHT_EYE_OUTER].x)
        avg_eye_width = max(0.001, (left_eye_width + right_eye_width) / 2.0)

        # 眉-目間距離（MediaPipeのYは上が0なので、目-眉で正の値が眉が上）
        left_dist = left_eye_top_y - left_brow_y
        right_dist = right_eye_top_y - right_brow_y
        avg_dist = (left_dist + right_dist) / 2.0

        # 目の幅で正規化
        return avg_dist / avg_eye_width
