"""視線推定モジュール — One Euro Filter + 頭部姿勢融合 + 多項式キャリブレーション"""

import math
import time
from typing import List, NamedTuple, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .fusion import GazeFusion
from .head_pose_estimator import HeadPoseEstimator


class GazeResult(NamedTuple):
    """視線推定結果"""
    x: float
    y: float
    left_ear: float
    right_ear: float
    confidence: float
    landmarks: Optional[object] = None
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    precision_mode: bool = False
    fusion_w_gaze: float = 1.0


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


class GazeEstimator:
    """MediaPipe Face Meshを用いた視線推定クラス

    虹彩の目内相対位置にOne Euro Filterを適用してノイズを除去し、
    顔位置補正・多項式キャリブレーションで画面座標に変換する。
    """

    def __init__(self, screen_width: int, screen_height: int, skip_model: bool = False) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height

        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        if not skip_model:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=config.MAX_NUM_FACES,
                    refine_landmarks=config.REFINE_LANDMARKS,
                    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                )
            except Exception as e:
                print(f"警告: MediaPipe Face Meshの初期化に失敗しました — {e}")

        # キャリブレーション
        self._calib_coeff_x: Optional[np.ndarray] = None
        self._calib_coeff_y: Optional[np.ndarray] = None
        self._calibration_matrix: Optional[np.ndarray] = None
        self._calibration_offset: Optional[np.ndarray] = None

        self._sensitivity = config.DEFAULT_SENSITIVITY

        # 頭部姿勢推定 + 融合
        self._head_pose = HeadPoseEstimator(screen_width, screen_height)
        self._fusion = GazeFusion()

        # 精密モード状態
        self._precision_mode = False
        self._precision_anchor_x: float = 0.0
        self._precision_anchor_y: float = 0.0

        # One Euro Filter（虹彩比率用 — ここでノイズを元から断つ）
        self._filter_iris_x = OneEuroFilter(
            min_cutoff=config.ONE_EURO_MIN_CUTOFF,
            beta=config.ONE_EURO_BETA,
            d_cutoff=config.ONE_EURO_D_CUTOFF,
        )
        self._filter_iris_y = OneEuroFilter(
            min_cutoff=config.ONE_EURO_MIN_CUTOFF,
            beta=config.ONE_EURO_BETA,
            d_cutoff=config.ONE_EURO_D_CUTOFF,
        )

        # One Euro Filter（画面座標用 — 最終出力の安定化）
        self._filter_screen_x = OneEuroFilter(
            min_cutoff=config.ONE_EURO_SCREEN_MIN_CUTOFF,
            beta=config.ONE_EURO_SCREEN_BETA,
            d_cutoff=1.0,
        )
        self._filter_screen_y = OneEuroFilter(
            min_cutoff=config.ONE_EURO_SCREEN_MIN_CUTOFF,
            beta=config.ONE_EURO_SCREEN_BETA,
            d_cutoff=1.0,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calib_coeff_x is not None or self._calibration_matrix is not None

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = max(1.0, min(10.0, value))

    @property
    def precision_mode(self) -> bool:
        return self._precision_mode

    @precision_mode.setter
    def precision_mode(self, value: bool) -> None:
        self._precision_mode = value

    @property
    def head_pose_estimator(self) -> HeadPoseEstimator:
        return self._head_pose

    def process_frame(self, frame: np.ndarray) -> Optional[GazeResult]:
        if self._face_mesh is None:
            return None

        orig_h, orig_w = frame.shape[:2]

        if orig_w > config.PROCESS_WIDTH:
            process_frame = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        else:
            process_frame = frame

        proc_h, proc_w = process_frame.shape[:2]

        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        now = time.time()

        # 虹彩比率を算出（顔オフセット減算なし — 頭部姿勢は融合で活用）
        iris_x, iris_y = self._compute_iris_ratio(landmarks)

        # One Euro Filter で虹彩比率を平滑化
        iris_x = self._filter_iris_x(iris_x, now)
        iris_y = self._filter_iris_y(iris_y, now)

        # EAR
        left_ear = self._compute_ear(landmarks, "left")
        right_ear = self._compute_ear(landmarks, "right")

        avg_ear = (left_ear + right_ear) / 2.0
        confidence = min(1.0, avg_ear / config.BLINK_EAR_THRESHOLD) if avg_ear < config.BLINK_EAR_THRESHOLD else 1.0

        # --- 視線ベースの画面座標 ---
        if self._calib_coeff_x is not None:
            features = self._poly_features(iris_x, iris_y)
            gaze_screen_x = float(features @ self._calib_coeff_x)
            gaze_screen_y = float(features @ self._calib_coeff_y)
        elif self._calibration_matrix is not None:
            gaze_vec = np.array([iris_x, iris_y])
            screen_pos = self._calibration_matrix @ gaze_vec + self._calibration_offset
            gaze_screen_x = float(screen_pos[0])
            gaze_screen_y = float(screen_pos[1])
        else:
            gaze_screen_x = iris_x * self._screen_width
            gaze_screen_y = iris_y * self._screen_height

        # --- 頭部姿勢ベースの画面座標 ---
        head_pose_result = self._head_pose.estimate(landmarks, proc_w, proc_h, now)

        head_yaw = 0.0
        head_pitch = 0.0
        w_gaze = 1.0

        if head_pose_result is not None:
            head_yaw = head_pose_result.yaw
            head_pitch = head_pose_result.pitch

            # ベースラインがまだない場合は自動設定
            if not self._head_pose.has_baseline:
                self._head_pose.set_baseline(head_pose_result)

            if self._precision_mode:
                # 精密モード: 頭部のみでアンカー周辺を微調整
                sensitivity_mult = config.PRECISION_MODE_SENSITIVITY_MULT
                head_delta_x, head_delta_y = self._head_pose.get_screen_offset(
                    head_pose_result, sensitivity_mult
                )
                screen_x, screen_y = self._fusion.compute_precision(
                    self._precision_anchor_x, self._precision_anchor_y,
                    head_delta_x, head_delta_y,
                )
                w_gaze = 0.0
            else:
                # 通常モード: 視線 + 頭部の加重融合
                head_screen_x, head_screen_y = self._head_pose.get_head_screen_position(
                    head_pose_result
                )
                screen_x, screen_y, w_gaze, _ = self._fusion.compute(
                    gaze_screen_x, gaze_screen_y,
                    head_screen_x, head_screen_y,
                    now,
                )
        else:
            # 頭部姿勢推定失敗時は視線のみ
            screen_x = gaze_screen_x
            screen_y = gaze_screen_y

        # One Euro Filter で最終出力を平滑化
        screen_x = self._filter_screen_x(screen_x, now)
        screen_y = self._filter_screen_y(screen_y, now)

        screen_x = max(0.0, min(float(self._screen_width - 1), screen_x))
        screen_y = max(0.0, min(float(self._screen_height - 1), screen_y))

        return GazeResult(
            x=screen_x, y=screen_y,
            left_ear=left_ear, right_ear=right_ear,
            confidence=confidence, landmarks=landmarks,
            head_yaw=head_yaw, head_pitch=head_pitch,
            precision_mode=self._precision_mode,
            fusion_w_gaze=w_gaze,
        )

    def set_precision_anchor(self, x: float, y: float) -> None:
        """精密モードのアンカーポイントを設定する"""
        self._precision_anchor_x = x
        self._precision_anchor_y = y

    def reset_face_baseline(self) -> None:
        self._head_pose.reset_baseline()

    def set_calibration(self, matrix: np.ndarray, offset: np.ndarray) -> None:
        self._calibration_matrix = matrix.copy()
        self._calibration_offset = offset.copy()

    def compute_calibration(
        self, gaze_points: List[Tuple[float, float]], screen_points: List[Tuple[float, float]]
    ) -> bool:
        if len(gaze_points) < config.CALIBRATION_MIN_POINTS:
            return False

        gaze_arr = np.array(gaze_points)
        screen_arr = np.array(screen_points)
        n = len(gaze_points)

        A = np.column_stack([
            gaze_arr[:, 0], gaze_arr[:, 1],
            gaze_arr[:, 0] ** 2, gaze_arr[:, 1] ** 2,
            gaze_arr[:, 0] * gaze_arr[:, 1],
            np.ones(n),
        ])

        result_x, _, _, _ = np.linalg.lstsq(A, screen_arr[:, 0], rcond=None)
        result_y, _, _, _ = np.linalg.lstsq(A, screen_arr[:, 1], rcond=None)

        self._calib_coeff_x = result_x
        self._calib_coeff_y = result_y
        self._calibration_matrix = None
        self._calibration_offset = None

        # フィルタ・頭部姿勢・融合もリセット
        self._filter_iris_x.reset()
        self._filter_iris_y.reset()
        self._filter_screen_x.reset()
        self._filter_screen_y.reset()
        self._head_pose.reset()
        self._fusion.reset()

        return True

    @staticmethod
    def _poly_features(gx: float, gy: float) -> np.ndarray:
        return np.array([gx, gy, gx ** 2, gy ** 2, gx * gy, 1.0])

    def _compute_iris_ratio(self, landmarks: object) -> Tuple[float, float]:
        lm = landmarks.landmark

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x, lm[idx].y])

        # 左目
        left_iris = pt(config.LEFT_IRIS_CENTER)
        left_inner = pt(config.LEFT_EYE_INNER)
        left_outer = pt(config.LEFT_EYE_OUTER)

        left_eye_center = (left_inner + left_outer) / 2.0
        left_eye_width = max(0.001, np.linalg.norm(left_outer - left_inner))

        left_dir = left_outer - left_inner
        left_dir_norm = left_dir / np.linalg.norm(left_dir)
        left_perp = np.array([-left_dir_norm[1], left_dir_norm[0]])

        left_diff = left_iris - left_eye_center
        left_ratio_x = np.dot(left_diff, left_dir_norm) / left_eye_width
        left_ratio_y = np.dot(left_diff, left_perp) / max(0.001, np.linalg.norm(
            pt(config.LEFT_EYE_BOTTOM) - pt(config.LEFT_EYE_TOP)))

        # 右目
        right_iris = pt(config.RIGHT_IRIS_CENTER)
        right_inner = pt(config.RIGHT_EYE_INNER)
        right_outer = pt(config.RIGHT_EYE_OUTER)

        right_eye_center = (right_inner + right_outer) / 2.0
        right_eye_width = max(0.001, np.linalg.norm(right_outer - right_inner))

        right_dir = right_outer - right_inner
        right_dir_norm = right_dir / np.linalg.norm(right_dir)
        right_perp = np.array([-right_dir_norm[1], right_dir_norm[0]])

        right_diff = right_iris - right_eye_center
        right_ratio_x = np.dot(right_diff, right_dir_norm) / right_eye_width
        right_ratio_y = np.dot(right_diff, right_perp) / max(0.001, np.linalg.norm(
            pt(config.RIGHT_EYE_BOTTOM) - pt(config.RIGHT_EYE_TOP)))

        avg_x = (left_ratio_x + right_ratio_x) / 2.0
        avg_y = (left_ratio_y + right_ratio_y) / 2.0

        mapped_x = 0.5 + avg_x * self._sensitivity
        mapped_y = 0.5 + avg_y * self._sensitivity

        return (mapped_x, mapped_y)

    def _compute_ear(self, landmarks: object, eye: str) -> float:
        lm = landmarks.landmark

        if eye == "left":
            top_idx, bottom_idx = config.LEFT_EYE_TOP, config.LEFT_EYE_BOTTOM
            inner_idx, outer_idx = config.LEFT_EYE_INNER, config.LEFT_EYE_OUTER
        else:
            top_idx, bottom_idx = config.RIGHT_EYE_TOP, config.RIGHT_EYE_BOTTOM
            inner_idx, outer_idx = config.RIGHT_EYE_INNER, config.RIGHT_EYE_OUTER

        top = np.array([lm[top_idx].x, lm[top_idx].y])
        bottom = np.array([lm[bottom_idx].x, lm[bottom_idx].y])
        inner = np.array([lm[inner_idx].x, lm[inner_idx].y])
        outer = np.array([lm[outer_idx].x, lm[outer_idx].y])

        vertical = np.linalg.norm(bottom - top)
        horizontal = np.linalg.norm(outer - inner)

        if horizontal < 0.001:
            return 0.0
        return float(vertical / horizontal)

    def release(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
