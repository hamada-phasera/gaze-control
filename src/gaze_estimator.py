"""視線推定モジュール — One Euro Filter + 頭部姿勢融合 + 多項式キャリブレーション"""

import time
from typing import List, NamedTuple, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from . import blendshapes, config
from .filters import OneEuroFilter  # 循環回避のため filters から（re-export）
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
    blink_score: Optional[float] = None   # blendshape瞬きスコア(0〜1)。無ければNone
    brow_score: Optional[float] = None    # blendshape眉上げスコア(0〜1)。無ければNone
    distance_cm: float = 0.0              # カメラからの推定距離 (cm, 概算)


class _LandmarksAdapter:
    """Tasks API の出力（NormalizedLandmark のリスト）を、旧 FaceMesh と同じ
    `landmarks.landmark[idx].x` インターフェースで参照できるようにする薄いラッパー。

    これにより head_pose_estimator / precision_mode など index ベースの下流コードを
    一切変更せずに新APIへ移行できる。
    """

    __slots__ = ("landmark",)

    def __init__(self, landmark_list: object) -> None:
        self.landmark = landmark_list


class GazeEstimator:
    """MediaPipe による視線推定クラス（Tasks FaceLandmarker 優先 / 旧FaceMeshフォールバック）

    虹彩の目内相対位置にOne Euro Filterを適用してノイズを除去し、
    顔位置補正・多項式キャリブレーションで画面座標に変換する。
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        skip_model: bool = False,
        enable_precision: bool = False,
    ) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height

        self._face_mesh = None          # 旧 mp.solutions.face_mesh（フォールバック）
        self._landmarker = None         # 新 Tasks FaceLandmarker（優先）
        self._last_ts_ms: Optional[int] = None  # VIDEOモード用の単調増加タイムスタンプ
        self._last_blendshapes = None   # 直近フレームの表情係数（あれば）

        if not skip_model:
            if config.USE_FACE_LANDMARKER_TASKS:
                self._landmarker = self._create_face_landmarker()
            if self._landmarker is None:
                self._face_mesh = self._create_legacy_face_mesh()

        # キャリブレーション
        self._calib_coeff_x: Optional[np.ndarray] = None
        self._calib_coeff_y: Optional[np.ndarray] = None
        self._calibration_matrix: Optional[np.ndarray] = None
        self._calibration_offset: Optional[np.ndarray] = None

        self._sensitivity = config.DEFAULT_SENSITIVITY

        # 可動域・距離適応
        self._range_mult = config.GAZE_RANGE_MULT          # 可動域（ゲイン）倍率
        self._distance_adapt = config.DISTANCE_ADAPT       # 距離適応の強さ
        self._distance_ref = config.DISTANCE_REF_INTER_EYE  # 基準の目外角間距離（キャリブ時に更新）
        self._last_inter_eye = config.DISTANCE_REF_INTER_EYE
        self._last_distance_cm = config.DISTANCE_REF_CM
        self._effective_gain = config.DEFAULT_SENSITIVITY  # 距離適応後の実効ゲイン（毎フレーム算出）

        # 縦方向の可動域調整（下方向の届きにくさ対策）
        self._v_gain = config.VERTICAL_GAIN
        self._down_boost = config.VERTICAL_DOWN_BOOST
        self._down_smooth = config.VERTICAL_DOWN_SMOOTH  # 下を見るほど強く平滑化

        # 頭部姿勢推定 + 融合
        self._head_pose = HeadPoseEstimator(screen_width, screen_height)
        self._fusion = GazeFusion()

        # 精密モード状態
        self._precision_mode = False
        self._precision_anchor_x: float = 0.0
        self._precision_anchor_y: float = 0.0

        # 眉上げ（blendshape優先）で精密モードを自動切替するディテクタ（opt-in）
        self._precision_detector = None
        if enable_precision:
            from .precision_mode import PrecisionModeDetector
            self._precision_detector = PrecisionModeDetector()

        # 直近の出力座標（精密モード開始時のアンカー用）
        self._last_output_x: float = screen_width / 2.0
        self._last_output_y: float = screen_height / 2.0

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

    @staticmethod
    def _create_face_landmarker():
        """Tasks API の FaceLandmarker を生成する。失敗時は None（フォールバック）。"""
        import os

        model_path = config.FACE_LANDMARKER_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"警告: FaceLandmarkerモデルが見つかりません ({model_path}) — 旧FaceMeshにフォールバックします")
            return None
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=config.MAX_NUM_FACES,
                min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=config.MIN_FACE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                output_face_blendshapes=config.OUTPUT_BLENDSHAPES,
                output_facial_transformation_matrixes=False,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
            print("FaceLandmarker (Tasks API) を初期化しました")
            return landmarker
        except Exception as e:  # noqa: BLE001 — フォールバックするため握り潰す
            print(f"警告: FaceLandmarker(Tasks API)の初期化に失敗しました — {e}")
            return None

    @staticmethod
    def _create_legacy_face_mesh():
        """旧 mp.solutions.face_mesh.FaceMesh を生成する。失敗時は None。"""
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=config.MAX_NUM_FACES,
                refine_landmarks=config.REFINE_LANDMARKS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            print("MediaPipe FaceMesh (legacy) を初期化しました")
            return face_mesh
        except Exception as e:  # noqa: BLE001
            print(f"警告: MediaPipe Face Meshの初期化に失敗しました — {e}")
            return None

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
    def range_mult(self) -> float:
        return self._range_mult

    @range_mult.setter
    def range_mult(self, value: float) -> None:
        self._range_mult = max(0.1, float(value))

    @property
    def distance_adapt(self) -> float:
        return self._distance_adapt

    @distance_adapt.setter
    def distance_adapt(self, value: float) -> None:
        self._distance_adapt = max(0.0, min(1.0, float(value)))

    @property
    def v_gain(self) -> float:
        return self._v_gain

    @v_gain.setter
    def v_gain(self, value: float) -> None:
        self._v_gain = max(0.1, float(value))

    @property
    def down_boost(self) -> float:
        return self._down_boost

    @down_boost.setter
    def down_boost(self, value: float) -> None:
        self._down_boost = max(0.0, float(value))

    @property
    def down_smooth(self) -> float:
        return self._down_smooth

    @down_smooth.setter
    def down_smooth(self, value: float) -> None:
        self._down_smooth = max(0.0, min(0.95, float(value)))

    @staticmethod
    def _down_smooth_scale(screen_y: float, screen_h: float, down_smooth: float) -> float:
        """下にいるほど小さい（=より強く平滑化）カットオフ倍率を返す。

        画面上半分/中央は 1.0、最下端で 1 - down_smooth。
        """
        if down_smooth <= 0.0 or screen_h <= 0.0:
            return 1.0
        frac_down = max(0.0, min(1.0, (screen_y / screen_h - 0.5) * 2.0))
        return max(0.05, 1.0 - down_smooth * frac_down)

    @staticmethod
    def _vertical_offset(
        avg_y: float, base_gain: float, v_gain: float,
        down_boost: float, sign: float, ref: float,
    ) -> float:
        """縦方向のオフセット (0.5 からのズレ)。下を見るほどゲインを増やす。"""
        gy = base_gain * v_gain
        down = avg_y * sign  # >0 で「下向き」
        if down_boost > 0.0 and down > 0.0 and ref > 0.0:
            gy *= 1.0 + down_boost * min(1.0, down / ref)
        return avg_y * gy

    @property
    def last_distance_cm(self) -> float:
        return self._last_distance_cm

    @property
    def precision_mode(self) -> bool:
        return self._precision_mode

    @precision_mode.setter
    def precision_mode(self, value: bool) -> None:
        self._precision_mode = value

    @property
    def head_pose_estimator(self) -> HeadPoseEstimator:
        return self._head_pose

    def _next_timestamp_ms(self, now: float) -> int:
        """VIDEOモードに渡す単調増加タイムスタンプ (ms) を返す。"""
        ts = int(now * 1000)
        if self._last_ts_ms is not None and ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _detect(self, rgb_frame: np.ndarray, now: float) -> Optional[object]:
        """有効なバックエンドでランドマークを検出し、共通アダプタ形式で返す。"""
        if self._landmarker is not None:
            ts_ms = self._next_timestamp_ms(now)
            try:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )
                result = self._landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:  # noqa: BLE001 — 1フレーム失敗は無視
                return None
            if not result.face_landmarks:
                return None
            self._last_blendshapes = (
                result.face_blendshapes[0]
                if getattr(result, "face_blendshapes", None)
                else None
            )
            return _LandmarksAdapter(result.face_landmarks[0])

        # 旧FaceMeshフォールバック
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def process_frame(self, frame: np.ndarray) -> Optional[GazeResult]:
        if self._landmarker is None and self._face_mesh is None:
            return None

        orig_h, orig_w = frame.shape[:2]

        if orig_w > config.PROCESS_WIDTH:
            process_frame = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        else:
            process_frame = frame

        proc_h, proc_w = process_frame.shape[:2]

        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        now = time.time()

        landmarks = self._detect(rgb_frame, now)
        if landmarks is None:
            return None

        # 距離推定（目外角間の正規化距離）と距離適応ゲイン
        self._last_inter_eye = self._inter_eye_distance(landmarks)
        self._last_distance_cm = self._inter_eye_to_cm(self._last_inter_eye)
        df = self.distance_factor(self._last_inter_eye, self._distance_ref, self._distance_adapt)
        self._effective_gain = self._sensitivity * self._range_mult * df

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

        # --- Blendshape（表情係数）スコア（Tasks API利用時のみ）---
        bs = blendshapes.score_map(self._last_blendshapes)
        blink_score = blendshapes.blink_score(bs)
        brow_score = blendshapes.brow_raise_score(bs)

        # --- 眉上げで精密モードを自動切替（opt-in: enable_precision）---
        if self._precision_detector is not None:
            was_active = self._precision_mode
            active = self._precision_detector.update(landmarks, brow_score)
            if active and not was_active:
                # 精密モード開始: 直前の出力位置をアンカーに固定
                self._precision_anchor_x = self._last_output_x
                self._precision_anchor_y = self._last_output_y
            self._precision_mode = active

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

        # One Euro Filter で最終出力を平滑化（縦は下にいるほど強く平滑化して分散を抑える）
        cutoff_scale_y = self._down_smooth_scale(screen_y, self._screen_height, self._down_smooth)
        screen_x = self._filter_screen_x(screen_x, now)
        screen_y = self._filter_screen_y(screen_y, now, cutoff_scale=cutoff_scale_y)

        screen_x = max(0.0, min(float(self._screen_width - 1), screen_x))
        screen_y = max(0.0, min(float(self._screen_height - 1), screen_y))

        # 精密モード開始時のアンカー用に出力を保持
        self._last_output_x = screen_x
        self._last_output_y = screen_y

        return GazeResult(
            x=screen_x, y=screen_y,
            left_ear=left_ear, right_ear=right_ear,
            confidence=confidence, landmarks=landmarks,
            head_yaw=head_yaw, head_pitch=head_pitch,
            precision_mode=self._precision_mode,
            fusion_w_gaze=w_gaze,
            blink_score=blink_score,
            brow_score=brow_score,
            distance_cm=self._last_distance_cm,
        )

    def set_precision_anchor(self, x: float, y: float) -> None:
        """精密モードのアンカーポイントを設定する"""
        self._precision_anchor_x = x
        self._precision_anchor_y = y

    @staticmethod
    def _inter_eye_distance(landmarks: object) -> float:
        """両目の外角どうしの正規化距離。近いほど大きい（距離の逆数の代理）。"""
        lm = landmarks.landmark
        ax, ay = lm[config.LEFT_EYE_OUTER].x, lm[config.LEFT_EYE_OUTER].y
        bx, by = lm[config.RIGHT_EYE_OUTER].x, lm[config.RIGHT_EYE_OUTER].y
        return float(np.hypot(ax - bx, ay - by))

    @staticmethod
    def _inter_eye_to_cm(inter_eye: float) -> float:
        """目外角間距離からカメラ距離 (cm) を概算する。"""
        if inter_eye <= 1e-6:
            return config.DISTANCE_REF_CM
        return float(config.DISTANCE_REF_CM * (config.DISTANCE_REF_INTER_EYE / inter_eye))

    @staticmethod
    def distance_factor(inter_eye: float, ref: float, adapt: float) -> float:
        """距離適応ゲイン係数。近い(inter_eye大)ほど下げ、遠いほど上げる。

        adapt=0 で 1.0（無効）。極端を避けるため [0.4, 2.5] にクランプ。
        """
        if inter_eye <= 1e-6 or adapt <= 0.0 or ref <= 0.0:
            return 1.0
        raw = ref / inter_eye  # 近い→<1, 遠い→>1
        factor = 1.0 + adapt * (raw - 1.0)
        return float(min(2.5, max(0.4, factor)))

    def save_calibration(self, path: str) -> bool:
        """多項式キャリブ係数と距離基準を JSON に保存する。"""
        if self._calib_coeff_x is None or self._calib_coeff_y is None:
            return False
        import json
        data = {
            "type": "poly",
            "coeff_x": [float(v) for v in self._calib_coeff_x],
            "coeff_y": [float(v) for v in self._calib_coeff_y],
            "distance_ref": float(self._distance_ref),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f)
            return True
        except Exception:  # noqa: BLE001
            return False

    def load_calibration(self, path: str) -> bool:
        """保存済みキャリブを読み込む。成功で True。"""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            cx = np.array(data["coeff_x"], dtype=float)
            cy = np.array(data["coeff_y"], dtype=float)
            if cx.shape[0] != 6 or cy.shape[0] != 6:
                return False
            self._calib_coeff_x = cx
            self._calib_coeff_y = cy
            self._calibration_matrix = None
            self._calibration_offset = None
            self._distance_ref = float(data.get("distance_ref", config.DISTANCE_REF_INTER_EYE))
            return True
        except Exception:  # noqa: BLE001
            return False

    def reset_face_baseline(self) -> None:
        self._head_pose.reset_baseline()

    def reset_runtime(self) -> None:
        """実行中リセット: フィルタ・頭部基準・融合・精密モードを初期化して中央へ戻す。

        カーソルが端に張り付いた／ドリフトした時に、現在の頭の向きを新しい基準
        （正面）として取り直し、出力を画面中央付近へ復帰させる。
        """
        self._filter_iris_x.reset()
        self._filter_iris_y.reset()
        self._filter_screen_x.reset()
        self._filter_screen_y.reset()
        self._head_pose.reset()   # 基準クリア → 次フレームで現在の頭部姿勢を基準に再設定
        self._fusion.reset()
        self._precision_mode = False
        self._last_ts_ms = None
        self._last_output_x = self._screen_width / 2.0
        self._last_output_y = self._screen_height / 2.0

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

        # 距離適応の基準を「キャリブした距離」に設定
        self._distance_ref = self._last_inter_eye

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

        mapped_x = 0.5 + avg_x * self._effective_gain
        mapped_y = 0.5 + self._vertical_offset(
            avg_y, self._effective_gain, self._v_gain, self._down_boost,
            config.VERTICAL_DOWN_SIGN, config.VERTICAL_DOWN_REF,
        )

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

    @property
    def last_blendshapes(self) -> Optional[object]:
        """直近フレームの表情係数（Tasks API利用時のみ。無ければ None）。"""
        return self._last_blendshapes

    def release(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
        if self._face_mesh is not None:
            self._face_mesh.close()
