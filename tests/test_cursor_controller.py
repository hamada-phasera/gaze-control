"""CursorController のテスト — PyAutoGUIをモックしてテスト"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.cursor_controller import CursorController


class TestCursorController:
    """CursorController の基本テスト"""

    @patch("src.cursor_controller.pyautogui")
    def test_init_dwell_mode(self, mock_pyautogui: MagicMock) -> None:
        controller = CursorController(blink_click=False)
        assert controller.dwell_progress == 0.0

    @patch("src.cursor_controller.pyautogui")
    def test_init_blink_mode(self, mock_pyautogui: MagicMock) -> None:
        controller = CursorController(blink_click=True)
        assert controller.dwell_progress == 0.0

    @patch("src.cursor_controller.pyautogui")
    def test_update_moves_cursor(self, mock_pyautogui: MagicMock) -> None:
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.FAILSAFE = True

        controller = CursorController(blink_click=False)
        controller.update(500.0, 300.0, 0.3, 0.3, 1.0)
        mock_pyautogui.moveTo.assert_called()

    @patch("src.cursor_controller.pyautogui")
    def test_low_confidence_no_update(self, mock_pyautogui: MagicMock) -> None:
        controller = CursorController(blink_click=False)
        result = controller.update(500.0, 300.0, 0.3, 0.3, 0.05)
        assert result is False
        mock_pyautogui.moveTo.assert_not_called()

    @patch("src.cursor_controller.pyautogui")
    def test_blink_click_detection(self, mock_pyautogui: MagicMock) -> None:
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=True)
        for _ in range(5):
            controller.update(500.0, 300.0, 0.1, 0.1, 1.0)
        clicked = controller.update(500.0, 300.0, 0.3, 0.3, 1.0)
        assert clicked is True

    @patch("src.cursor_controller.pyautogui")
    def test_blink_too_short_no_click(self, mock_pyautogui: MagicMock) -> None:
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=True)
        controller.update(500.0, 300.0, 0.1, 0.1, 1.0)
        clicked = controller.update(500.0, 300.0, 0.3, 0.3, 1.0)
        assert clicked is False

    @patch("src.cursor_controller.pyautogui")
    def test_blink_too_long_no_click(self, mock_pyautogui: MagicMock) -> None:
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=True)
        for _ in range(15):
            controller.update(500.0, 300.0, 0.1, 0.1, 1.0)
        clicked = controller.update(500.0, 300.0, 0.3, 0.3, 1.0)
        assert clicked is False

    @patch("src.cursor_controller.pyautogui")
    def test_blink_click_via_blendshape(self, mock_pyautogui: MagicMock) -> None:
        """blendshape瞬きスコアでクリックが成立する"""
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=True)
        # 閉眼（高スコア）を数フレーム → 開眼で確定
        for _ in range(5):
            controller.update(500.0, 300.0, 0.3, 0.3, 1.0, blink_score=0.9)
        clicked = controller.update(500.0, 300.0, 0.3, 0.3, 1.0, blink_score=0.0)
        assert clicked is True

    @patch("src.cursor_controller.pyautogui")
    def test_blendshape_overrides_ear(self, mock_pyautogui: MagicMock) -> None:
        """EARは開眼(0.3)でも、blendshape瞬きスコアが高ければ閉眼として扱う"""
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=True)
        for _ in range(5):
            controller.update(500.0, 300.0, 0.3, 0.3, 1.0, blink_score=0.9)
        clicked = controller.update(500.0, 300.0, 0.3, 0.3, 1.0, blink_score=0.0)
        assert clicked is True

    @patch("src.cursor_controller.pyautogui")
    def test_inertia_smooth_movement(self, mock_pyautogui: MagicMock) -> None:
        """慣性: カーソルが一度に目標へジャンプせず、徐々に近づく"""
        mock_pyautogui.size.return_value = (1920, 1080)

        controller = CursorController(blink_click=False)

        # 初回: (100,100)に設定
        controller.update(100.0, 100.0, 0.3, 0.3, 1.0)

        # 遠い目標(800, 600)
        controller.update(800.0, 600.0, 0.3, 0.3, 1.0)

        # 慣性により到達していないが前進はしている
        assert controller._cursor_x < 800.0
        assert controller._cursor_y < 600.0
        assert controller._cursor_x > 100.0
        assert controller._cursor_y > 100.0
