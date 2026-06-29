"""カーソル制御モジュール — 慣性スムージング + クリック判定 + UIスナップ"""

import math
import time
from typing import Optional

import pyautogui

from . import config
from .accessibility_snap import AccessibilitySnap

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class CursorController:
    """OSカーソル移動とクリック判定

    GazeEstimator側でOne Euro Filterによるノイズ除去済みの座標を受け取り、
    慣性（バネ+摩擦）モデルでスムーズにカーソルを移動させる。
    クリック時にはAccessibility APIで最寄りのUI要素にスナップする。
    """

    def __init__(self, blink_click: bool = False, snap_enabled: bool = True) -> None:
        self._blink_click = blink_click

        # Accessibility スナップ
        self._snap: Optional[AccessibilitySnap] = None
        if snap_enabled and config.SNAP_ENABLED:
            self._snap = AccessibilitySnap()
            if not AccessibilitySnap.is_available():
                print("注意: pyobjcが未インストールのためUI要素スナップは無効です")
                self._snap = None

        # 慣性カーソル状態
        self._cursor_x: Optional[float] = None
        self._cursor_y: Optional[float] = None
        self._cursor_vx: float = 0.0
        self._cursor_vy: float = 0.0

        # Dwell Click 状態
        self._dwell_start_time: Optional[float] = None
        self._dwell_center_x: float = 0.0
        self._dwell_center_y: float = 0.0
        self._dwell_cooldown_until: float = 0.0

        # 瞬きクリック状態
        self._blink_counter: int = 0
        self._was_blinking: bool = False

    def update(
        self,
        target_x: float,
        target_y: float,
        left_ear: float,
        right_ear: float,
        confidence: float,
        blink_score: Optional[float] = None,
    ) -> bool:
        if confidence < 0.1:
            return False

        # 初期化
        if self._cursor_x is None:
            self._cursor_x = target_x
            self._cursor_y = target_y

        # バネ力: ターゲットとの差分に比例
        dx = target_x - self._cursor_x
        dy = target_y - self._cursor_y

        self._cursor_vx += dx * config.INERTIA_ACCELERATION
        self._cursor_vy += dy * config.INERTIA_ACCELERATION

        # 摩擦
        self._cursor_vx *= config.INERTIA_FRICTION
        self._cursor_vy *= config.INERTIA_FRICTION

        # 速度制限
        speed = math.sqrt(self._cursor_vx ** 2 + self._cursor_vy ** 2)
        if speed > config.INERTIA_MAX_SPEED:
            scale = config.INERTIA_MAX_SPEED / speed
            self._cursor_vx *= scale
            self._cursor_vy *= scale
            speed = config.INERTIA_MAX_SPEED

        # 微小速度 + ターゲット近傍で停止
        dist = math.sqrt(dx * dx + dy * dy)
        if speed < config.INERTIA_MIN_SPEED and dist < config.ADAPTIVE_DEAD_ZONE_MAX:
            self._cursor_vx = 0.0
            self._cursor_vy = 0.0
        else:
            self._cursor_x += self._cursor_vx
            self._cursor_y += self._cursor_vy

        # 画面範囲にクランプ
        screen_w, screen_h = pyautogui.size()
        self._cursor_x = max(0.0, min(float(screen_w - 1), self._cursor_x))
        self._cursor_y = max(0.0, min(float(screen_h - 1), self._cursor_y))

        # OSカーソルを移動
        try:
            pyautogui.moveTo(int(self._cursor_x), int(self._cursor_y), _pause=False)
        except pyautogui.FailSafeException:
            return False

        # クリック判定
        if self._blink_click:
            is_blinking = self._is_eye_closed(left_ear, right_ear, blink_score)
            return self._check_blink_click(is_blinking)
        else:
            return self._check_dwell_click(self._cursor_x, self._cursor_y)

    @staticmethod
    def _is_eye_closed(
        left_ear: float, right_ear: float, blink_score: Optional[float]
    ) -> bool:
        """閉眼かどうかを判定する。blendshape瞬きスコアがあれば最優先。"""
        if blink_score is not None and config.USE_BLENDSHAPE_BLINK:
            return blink_score >= config.BLINK_BLENDSHAPE_THRESHOLD
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < config.BLINK_EAR_THRESHOLD

    def _snap_and_click(self) -> bool:
        """スナップしてクリックする"""
        if self._snap is not None and self._cursor_x is not None:
            snap_x, snap_y = self._snap.snap_to_nearest(self._cursor_x, self._cursor_y)
            try:
                pyautogui.moveTo(int(snap_x), int(snap_y), _pause=False)
                self._cursor_x = snap_x
                self._cursor_y = snap_y
            except pyautogui.FailSafeException:
                pass

        try:
            pyautogui.click(_pause=False)
        except pyautogui.FailSafeException:
            return False
        return True

    def _check_dwell_click(self, x: float, y: float) -> bool:
        now = time.time()
        if now < self._dwell_cooldown_until:
            return False

        dx = x - self._dwell_center_x
        dy = y - self._dwell_center_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > config.DWELL_RADIUS:
            self._dwell_center_x = x
            self._dwell_center_y = y
            self._dwell_start_time = now
            return False

        if self._dwell_start_time is None:
            self._dwell_start_time = now
            self._dwell_center_x = x
            self._dwell_center_y = y
            return False

        if now - self._dwell_start_time >= config.DWELL_TIME:
            if not self._snap_and_click():
                return False
            self._dwell_start_time = None
            self._dwell_cooldown_until = now + config.DWELL_COOLDOWN
            return True

        return False

    def _check_blink_click(self, is_blinking: bool) -> bool:
        if is_blinking:
            self._blink_counter += 1
            self._was_blinking = True
            return False

        if self._was_blinking:
            self._was_blinking = False
            frames = self._blink_counter
            self._blink_counter = 0
            if config.BLINK_FRAMES_MIN <= frames <= config.BLINK_FRAMES_MAX:
                return self._snap_and_click()

        return False

    @property
    def dwell_progress(self) -> float:
        if self._blink_click:
            return 0.0
        if self._dwell_start_time is None:
            return 0.0
        elapsed = time.time() - self._dwell_start_time
        return min(1.0, elapsed / config.DWELL_TIME)
