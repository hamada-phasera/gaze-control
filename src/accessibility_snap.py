"""macOS Accessibility API スナップモジュール

クリック時にカーソル位置近傍の最寄りUI要素にスナップし、
小さなターゲットも確実にクリック可能にする。

pyobjcが利用できない場合はグレースフルに無効化される。
"""

import math
from typing import Optional, Tuple

from . import config

# pyobjcのインポート（オプショナル）
_AX_AVAILABLE = False
try:
    from ApplicationServices import (  # type: ignore[import-not-found]
        AXUIElementCreateSystemWide,
        AXUIElementCopyElementAtPosition,
    )
    from CoreFoundation import CFRelease  # type: ignore[import-not-found]
    import Quartz  # type: ignore[import-not-found]
    _AX_AVAILABLE = True
except ImportError:
    pass

# インタラクション可能なAXRole一覧
_INTERACTABLE_ROLES = {
    "AXButton",
    "AXTextField",
    "AXTextArea",
    "AXLink",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXComboBox",
    "AXSlider",
    "AXMenuItem",
    "AXMenuButton",
    "AXTabGroup",
    "AXTab",
    "AXCell",
    "AXIncrementor",
    "AXSearchField",
    "AXSecureTextField",
}


class AccessibilitySnap:
    """macOS Accessibility APIを使用したUI要素スナップ

    カーソル位置の近傍を検索し、最も近いインタラクション可能な
    UI要素の中心座標を返す。毎フレームではなくクリック時のみ使用。
    """

    def __init__(
        self,
        snap_radius: float = config.SNAP_RADIUS,
        search_step: int = config.SNAP_SEARCH_STEP,
    ) -> None:
        self._snap_radius = snap_radius
        self._search_step = search_step
        self._system_wide = None

        if _AX_AVAILABLE:
            self._system_wide = AXUIElementCreateSystemWide()

    @staticmethod
    def is_available() -> bool:
        """Accessibility APIが利用可能かどうか"""
        return _AX_AVAILABLE

    def snap_to_nearest(self, x: float, y: float) -> Tuple[float, float]:
        """カーソル位置の最寄りUI要素にスナップする

        Args:
            x, y: 現在のカーソル位置 (px)

        Returns:
            (snap_x, snap_y): スナップ先の座標。
            スナップ対象がない場合は入力座標をそのまま返す。
        """
        if not _AX_AVAILABLE or self._system_wide is None:
            return (x, y)

        best_dist = float("inf")
        best_pos: Optional[Tuple[float, float]] = None

        radius = int(self._snap_radius)
        step = self._search_step

        # カーソル位置自体をまず検索
        result = self._get_interactable_center(x, y)
        if result is not None:
            return result

        # グリッド検索で近傍の要素を探す
        seen_positions: set[Tuple[int, int]] = set()

        for dx in range(-radius, radius + 1, step):
            for dy in range(-radius, radius + 1, step):
                if dx * dx + dy * dy > radius * radius:
                    continue

                probe_x = x + dx
                probe_y = y + dy

                center = self._get_interactable_center(probe_x, probe_y)
                if center is None:
                    continue

                # 同じ要素の重複を避ける（中心座標が同じ）
                center_key = (int(center[0]), int(center[1]))
                if center_key in seen_positions:
                    continue
                seen_positions.add(center_key)

                dist = math.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
                if dist < best_dist and dist <= self._snap_radius:
                    best_dist = dist
                    best_pos = center

        return best_pos if best_pos is not None else (x, y)

    def _get_interactable_center(
        self, x: float, y: float
    ) -> Optional[Tuple[float, float]]:
        """指定座標のUI要素がインタラクション可能なら中心座標を返す"""
        if self._system_wide is None:
            return None

        try:
            err, element = AXUIElementCopyElementAtPosition(
                self._system_wide, float(x), float(y)
            )
        except Exception:
            return None

        if err != 0 or element is None:
            return None

        try:
            role = self._get_attribute(element, "AXRole")
            if role not in _INTERACTABLE_ROLES:
                return None

            position = self._get_attribute(element, "AXPosition")
            size = self._get_attribute(element, "AXSize")

            if position is None or size is None:
                return None

            # CGPoint / CGSize から値を取得
            pos_x = Quartz.CGPointGetX(position) if hasattr(Quartz, "CGPointGetX") else position.x
            pos_y = Quartz.CGPointGetY(position) if hasattr(Quartz, "CGPointGetY") else position.y
            width = Quartz.CGSizeGetWidth(size) if hasattr(Quartz, "CGSizeGetWidth") else size.width
            height = Quartz.CGSizeGetHeight(size) if hasattr(Quartz, "CGSizeGetHeight") else size.height

            center_x = pos_x + width / 2.0
            center_y = pos_y + height / 2.0

            return (center_x, center_y)
        except Exception:
            return None

    @staticmethod
    def _get_attribute(element: object, attribute: str) -> Optional[object]:
        """AXUIElementの属性値を取得する"""
        try:
            from ApplicationServices import AXUIElementCopyAttributeValue  # type: ignore[import-not-found]
            err, value = AXUIElementCopyAttributeValue(element, attribute, None)
            if err == 0:
                return value
        except Exception:
            pass
        return None
