"""ジェスチャのテスト — 閉眼判定 + 3秒閉眼リセットトリガー"""

from src import config
from src.gestures import EyesClosedTrigger, is_eye_closed


class TestIsEyeClosed:
    def test_blendshape_priority(self) -> None:
        # 高スコア=閉眼（EARは開でも）
        assert is_eye_closed(0.3, 0.3, blink_score=0.9) is True
        assert is_eye_closed(0.05, 0.05, blink_score=0.0) is False

    def test_ear_fallback(self) -> None:
        # blink_score なし → EAR
        assert is_eye_closed(0.05, 0.05, blink_score=None) is True
        assert is_eye_closed(0.4, 0.4, blink_score=None) is False


class TestEyesClosedTrigger:
    def test_fires_after_duration(self) -> None:
        trig = EyesClosedTrigger(duration=3.0)
        assert trig.update(True, 0.0) is False
        assert trig.update(True, 1.5) is False
        assert trig.update(True, 2.99) is False
        assert trig.update(True, 3.0) is True   # 3秒到達で発火

    def test_fires_only_once(self) -> None:
        trig = EyesClosedTrigger(duration=3.0)
        trig.update(True, 0.0)
        assert trig.update(True, 3.1) is True
        assert trig.update(True, 3.2) is False  # 連続発火しない
        assert trig.update(True, 5.0) is False

    def test_opening_eyes_resets(self) -> None:
        trig = EyesClosedTrigger(duration=3.0)
        trig.update(True, 0.0)
        trig.update(False, 1.0)                  # 目を開けた
        assert trig.update(True, 1.1) is False   # 計時やり直し
        assert trig.update(True, 4.0) is False   # 1.1から3秒未満
        assert trig.update(True, 4.2) is True    # 1.1+3.0=4.1 を超えた

    def test_progress(self) -> None:
        trig = EyesClosedTrigger(duration=2.0)
        assert trig.progress(0.0) == 0.0
        trig.update(True, 0.0)
        assert trig.progress(1.0) == 0.5
        assert trig.progress(3.0) == 1.0

    def test_reset(self) -> None:
        trig = EyesClosedTrigger(duration=3.0)
        trig.update(True, 0.0)
        trig.reset()
        assert trig.progress(1.0) == 0.0
