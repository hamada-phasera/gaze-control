"""PrecisionModeDetector のテスト — blendshape経路の遷移（時刻注入で決定的に）"""

from src import config
from src.precision_mode import PrecisionModeDetector


def _det() -> PrecisionModeDetector:
    return PrecisionModeDetector(enter_duration=0.3, exit_duration=0.5)


# blendshape閾値より十分上/下のスコア
HIGH = config.PRECISION_MODE_BLENDSHAPE_THRESHOLD + 0.3
LOW = config.PRECISION_MODE_BLENDSHAPE_THRESHOLD - 0.3


class TestPrecisionModeBlendshape:
    def test_enter_requires_sustained_raise(self) -> None:
        d = _det()
        assert d.update(None, brow_score=HIGH, t=0.0) is False   # 開始計時
        assert d.update(None, brow_score=HIGH, t=0.2) is False   # 0.2 < 0.3
        assert d.update(None, brow_score=HIGH, t=0.31) is True    # 0.31 >= 0.3 → 有効
        assert d.is_active is True

    def test_exit_requires_sustained_lower(self) -> None:
        d = _det()
        d.update(None, brow_score=HIGH, t=0.0)
        d.update(None, brow_score=HIGH, t=0.31)
        assert d.is_active is True
        # 下げ始め
        assert d.update(None, brow_score=LOW, t=0.4) is True      # まだ有効
        assert d.update(None, brow_score=LOW, t=0.95) is False     # 0.55 >= 0.5 → 解除

    def test_brief_raise_does_not_activate(self) -> None:
        d = _det()
        d.update(None, brow_score=HIGH, t=0.0)
        # 0.3秒未満で下げる → 有効化されない
        assert d.update(None, brow_score=LOW, t=0.1) is False
        assert d.is_active is False

    def test_low_score_never_activates(self) -> None:
        d = _det()
        assert d.update(None, brow_score=LOW, t=0.0) is False
        assert d.update(None, brow_score=LOW, t=1.0) is False

    def test_reset(self) -> None:
        d = _det()
        d.update(None, brow_score=HIGH, t=0.0)
        d.update(None, brow_score=HIGH, t=0.31)
        assert d.is_active is True
        d.reset()
        assert d.is_active is False
