"""キャリブレーションUI モジュール — OpenCVフルスクリーン + IQR外れ値除去 + ランダム順"""

import random
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import config


class CalibrationOverlay:
    """OpenCVフルスクリーンウィンドウによるキャリブレーション

    画面上にランダム順でターゲット点を表示し、ユーザーが各点を見ている間の
    視線データを収集する。IQR外れ値除去を適用した上で
    多項式キャリブレーション用のデータを生成する。
    """

    WINDOW_NAME = "GazeControl Calibration"

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        gaze_callback: Callable[[], Optional[Tuple[float, float]]],
        head_baseline_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._gaze_callback = gaze_callback
        self._head_baseline_callback = head_baseline_callback

        # キャリブレーション結果
        self._gaze_points: List[Tuple[float, float]] = []
        self._screen_points: List[Tuple[float, float]] = []
        self._completed = False
        self._success = False

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def is_success(self) -> bool:
        return self._success

    @property
    def gaze_points(self) -> List[Tuple[float, float]]:
        return self._gaze_points.copy()

    @property
    def screen_points(self) -> List[Tuple[float, float]]:
        return self._screen_points.copy()

    def run(self) -> bool:
        """キャリブレーションを実行する（ブロッキング）"""
        self._gaze_points.clear()
        self._screen_points.clear()
        self._completed = False
        self._success = False

        # ターゲット座標を生成
        targets = self._generate_targets()

        # ランダム順に並び替え
        if config.CALIBRATION_RANDOMIZE:
            random.shuffle(targets)

        # OpenCVフルスクリーンウィンドウ作成
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 説明画面
        frame = np.zeros((self._screen_height, self._screen_width, 3), dtype=np.uint8)
        self._put_text_centered(
            frame,
            f"Calibration: {len(targets)} points",
            (self._screen_width // 2, self._screen_height // 2 - 40),
            scale=1.5,
        )
        self._put_text_centered(
            frame,
            "Look at each green dot. Press ESC to cancel.",
            (self._screen_width // 2, self._screen_height // 2 + 40),
            scale=0.9,
        )
        cv2.imshow(self.WINDOW_NAME, frame)
        if self._wait_key(2000):  # ESC pressed
            cv2.destroyWindow(self.WINDOW_NAME)
            return False

        # 各ターゲットでデータ収集
        for i, (tx, ty) in enumerate(targets):
            # ターゲット描画
            frame = np.zeros((self._screen_height, self._screen_width, 3), dtype=np.uint8)

            itx, ity = int(tx), int(ty)
            r = 15

            # 十字
            cv2.line(frame, (itx - r * 2, ity), (itx + r * 2, ity), (0, 255, 0), 1)
            cv2.line(frame, (itx, ity - r * 2), (itx, ity + r * 2), (0, 255, 0), 1)
            # 円
            cv2.circle(frame, (itx, ity), r, (0, 255, 0), 2)
            # 中心ドット
            cv2.circle(frame, (itx, ity), 3, (0, 255, 0), -1)

            # 進捗テキスト
            self._put_text_centered(
                frame,
                f"Point {i + 1}/{len(targets)} - Look at the green dot",
                (self._screen_width // 2, 40),
                scale=0.8,
            )

            # 進捗バー
            bar_x1, bar_y1 = 50, 60
            bar_x2 = self._screen_width - 50
            bar_w = int((bar_x2 - bar_x1) * (i / len(targets)))
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 10), (0, 170, 0), -1)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + 10), (80, 80, 80), 1)

            cv2.imshow(self.WINDOW_NAME, frame)
            if self._wait_key(500):  # 安定化待機 + ESCチェック
                break

            # データ収集
            gaze_samples: List[Tuple[float, float]] = []
            start_time = time.time()

            while time.time() - start_time < config.CALIBRATION_DURATION:
                gaze = self._gaze_callback()
                if gaze is not None:
                    gaze_samples.append(gaze)

                # 収集中もターゲットを表示し続ける（タイマー表示更新）
                elapsed = time.time() - start_time
                remaining = config.CALIBRATION_DURATION - elapsed
                display = frame.copy()
                self._put_text_centered(
                    display,
                    f"{remaining:.1f}s",
                    (itx, ity + r + 30),
                    scale=0.6,
                    color=(0, 200, 200),
                )
                cv2.imshow(self.WINDOW_NAME, display)

                key = cv2.waitKey(33) & 0xFF  # ~30fps
                if key == 27:  # ESC
                    cv2.destroyWindow(self.WINDOW_NAME)
                    self._completed = True
                    self._success = False
                    return False

            # IQR外れ値除去後の平均を使用
            filtered = self._filter_outliers(gaze_samples)
            if filtered is not None:
                self._gaze_points.append(filtered)
                self._screen_points.append((tx, ty))

        # --- 頭部ベースライン取得フェーズ ---
        if self._head_baseline_callback is not None:
            frame = np.zeros((self._screen_height, self._screen_width, 3), dtype=np.uint8)

            # 中央にターゲット描画
            cx, cy = self._screen_width // 2, self._screen_height // 2
            cv2.circle(frame, (cx, cy), 20, (0, 200, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)

            self._put_text_centered(
                frame,
                "Look at the center dot and face forward",
                (cx, cy - 60),
                scale=0.9,
                color=(0, 200, 255),
            )
            self._put_text_centered(
                frame,
                "Recording head baseline...",
                (cx, cy + 60),
                scale=0.7,
                color=(150, 150, 150),
            )

            cv2.imshow(self.WINDOW_NAME, frame)
            cv2.waitKey(500)

            # 2秒間データ収集してベースライン設定
            start_time = time.time()
            while time.time() - start_time < 2.0:
                self._head_baseline_callback()

                elapsed = time.time() - start_time
                progress = min(1.0, elapsed / 2.0)
                display = frame.copy()

                bar_x1, bar_y1 = cx - 100, cy + 90
                bar_w = int(200 * progress)
                cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 10), (0, 200, 255), -1)
                cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + 200, bar_y1 + 10), (80, 80, 80), 1)

                cv2.imshow(self.WINDOW_NAME, display)
                key = cv2.waitKey(33) & 0xFF
                if key == 27:
                    break

        cv2.destroyWindow(self.WINDOW_NAME)

        self._completed = True
        self._success = len(self._gaze_points) >= config.CALIBRATION_MIN_POINTS

        return self._success

    def _wait_key(self, ms: int) -> bool:
        """指定ミリ秒待機。ESCが押されたらTrueを返す。"""
        key = cv2.waitKey(ms) & 0xFF
        return key == 27

    @staticmethod
    def _put_text_centered(
        frame: np.ndarray,
        text: str,
        center: Tuple[int, int],
        scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> None:
        """テキストを中央揃えで描画"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = center[0] - tw // 2
        y = center[1] + th // 2
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)

    @staticmethod
    def _filter_outliers(
        samples: List[Tuple[float, float]], min_samples: int = 5
    ) -> Optional[Tuple[float, float]]:
        """IQR（四分位範囲）による外れ値除去後の平均を返す"""
        if len(samples) < min_samples:
            return None

        arr = np.array(samples)  # (N, 2)

        mask = np.ones(len(arr), dtype=bool)
        for axis in range(2):
            q1, q3 = np.percentile(arr[:, axis], [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask &= (arr[:, axis] >= lower) & (arr[:, axis] <= upper)

        filtered = arr[mask]

        if len(filtered) >= 3:
            return (float(np.mean(filtered[:, 0])), float(np.mean(filtered[:, 1])))
        else:
            return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

    def _generate_targets(self) -> List[Tuple[float, float]]:
        """グリッドのターゲット座標を生成"""
        margin_x = self._screen_width * config.CALIBRATION_MARGIN
        margin_y = self._screen_height * config.CALIBRATION_MARGIN

        usable_w = self._screen_width - 2 * margin_x
        usable_h = self._screen_height - 2 * margin_y

        targets = []
        grid = config.CALIBRATION_GRID_SIZE
        for row in range(grid):
            for col in range(grid):
                x = margin_x + usable_w * col / (grid - 1)
                y = margin_y + usable_h * row / (grid - 1)
                targets.append((float(x), float(y)))

        return targets
