"""頭部姿勢推定モジュール — solvePnPベースのyaw/pitch/roll算出"""

import math
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

from . import config
from .filters import OneEuroFilter


class HeadPoseResult(NamedTuple):
    """頭部姿勢推定結果"""
    yaw: float      # 横回転 (度) 左が正
    pitch: float    # 縦回転 (度) 上が正
    roll: float     # 傾き (度)


class HeadPoseEstimator:
    """solvePnPを用いた頭部姿勢推定

    6つの顔ランドマーク（鼻先、顎、両目外角、両口角）から
    頭部の3軸回転を推定し、One Euro Filterで平滑化する。
    ベースライン（正面向き）からの差分を画面座標の差分に変換する。
    """

    # solvePnPに使用するランドマークインデックス
    LANDMARK_INDICES = [
        config.NOSE_TIP,
        config.CHIN,
        config.LEFT_EYE_LEFT_CORNER,
        config.RIGHT_EYE_RIGHT_CORNER,
        config.LEFT_MOUTH_CORNER,
        config.RIGHT_MOUTH_CORNER,
    ]

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height

        # ベースライン（キャリブレーション時に設定）
        self._baseline_yaw: Optional[float] = None
        self._baseline_pitch: Optional[float] = None

        # One Euro Filter で姿勢角を平滑化
        self._filter_yaw = OneEuroFilter(
            min_cutoff=config.ONE_EURO_HEAD_MIN_CUTOFF,
            beta=config.ONE_EURO_HEAD_BETA,
        )
        self._filter_pitch = OneEuroFilter(
            min_cutoff=config.ONE_EURO_HEAD_MIN_CUTOFF,
            beta=config.ONE_EURO_HEAD_BETA,
        )

    def estimate(
        self, landmarks: object, frame_w: int, frame_h: int, t: Optional[float] = None
    ) -> Optional[HeadPoseResult]:
        """ランドマークから頭部姿勢を推定する

        Args:
            landmarks: MediaPipe Face Meshのランドマーク
            frame_w: フレーム幅 (px)
            frame_h: フレーム高さ (px)
            t: タイムスタンプ (秒, Noneの場合自動)

        Returns:
            HeadPoseResult or None (推定失敗時)
        """
        lm = landmarks.landmark

        # 2D画像座標を抽出
        image_points = np.array([
            [lm[idx].x * frame_w, lm[idx].y * frame_h]
            for idx in self.LANDMARK_INDICES
        ], dtype=np.float64)

        # カメラ内部行列の近似
        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # solvePnP
        success, rotation_vec, _ = cv2.solvePnP(
            config.FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # 回転ベクトル → 回転行列 → オイラー角
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_mat)

        # One Euro Filter で平滑化
        yaw = self._filter_yaw(yaw, t)
        pitch = self._filter_pitch(pitch, t)

        return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)

    def set_baseline(self, result: HeadPoseResult) -> None:
        """現在の姿勢をベースライン（正面向き）として記録する"""
        self._baseline_yaw = result.yaw
        self._baseline_pitch = result.pitch

    def reset_baseline(self) -> None:
        """ベースラインをリセットする"""
        self._baseline_yaw = None
        self._baseline_pitch = None

    @property
    def has_baseline(self) -> bool:
        return self._baseline_yaw is not None

    def get_screen_offset(
        self, result: HeadPoseResult, sensitivity_mult: float = 1.0
    ) -> Tuple[float, float]:
        """頭部姿勢から画面座標のオフセットを算出する

        Args:
            result: 頭部姿勢推定結果
            sensitivity_mult: 感度倍率 (精密モードで使用)

        Returns:
            (delta_x, delta_y) 画面中心からのオフセット (px)
        """
        if self._baseline_yaw is None:
            return (0.0, 0.0)

        delta_yaw = result.yaw - self._baseline_yaw
        delta_pitch = result.pitch - self._baseline_pitch

        delta_x = delta_yaw * config.HEAD_SENSITIVITY_YAW * sensitivity_mult
        delta_y = -delta_pitch * config.HEAD_SENSITIVITY_PITCH * sensitivity_mult

        return (delta_x, delta_y)

    def yaw_offset(self, result: HeadPoseResult, gain: float) -> float:
        """頭のヨー（左右）が基準からズレた分 × gain を返す（横ドリフト補正用）。

        基準未設定 or gain=0 なら 0。
        """
        if self._baseline_yaw is None or gain == 0.0:
            return 0.0
        return (result.yaw - self._baseline_yaw) * gain

    def vertical_assist(self, result: HeadPoseResult, gain: float) -> float:
        """頭のピッチ（うなずき）から縦方向のオフセット (px) を返す。

        基準より下を向く（pitch小）と正（下へ）、上を向くと負（上へ）。横は使わない。
        基準未設定なら 0。
        """
        if self._baseline_pitch is None or gain == 0.0:
            return 0.0
        delta_pitch = result.pitch - self._baseline_pitch
        return -delta_pitch * gain

    def get_head_screen_position(
        self, result: HeadPoseResult, sensitivity_mult: float = 1.0
    ) -> Tuple[float, float]:
        """頭部姿勢から画面座標（絶対値）を算出する

        Returns:
            (screen_x, screen_y) 画面座標
        """
        delta_x, delta_y = self.get_screen_offset(result, sensitivity_mult)
        screen_x = self._screen_width / 2.0 + delta_x
        screen_y = self._screen_height / 2.0 + delta_y
        return (screen_x, screen_y)

    def reset(self) -> None:
        """フィルタとベースラインを全リセットする"""
        self._filter_yaw.reset()
        self._filter_pitch.reset()
        self._baseline_yaw = None
        self._baseline_pitch = None

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角 (yaw, pitch, roll) を度数で返す"""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy > 1e-6:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(-R[1, 2], R[1, 1])
            roll = 0.0

        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )
