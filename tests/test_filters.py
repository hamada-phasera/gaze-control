"""OneEuroFilter のテスト + 循環インポート回帰ガード（mediapipe非依存）"""

from src.filters import OneEuroFilter


class TestOneEuroFilter:
    def test_first_value_passthrough(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        assert f(5.0, t=0.0) == 5.0

    def test_smoothing_converges(self) -> None:
        f = OneEuroFilter(min_cutoff=0.3, beta=0.0)
        out = 0.0
        for i in range(30):
            out = f(0.5 + (0.1 if i % 2 == 0 else -0.1), t=i / 30.0)
        assert abs(out - 0.5) < 0.05

    def test_fast_movement_tracking(self) -> None:
        f = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        f(0.0, t=0.0)
        f(0.0, t=0.1)
        assert f(1.0, t=0.2) > 0.2

    def test_reset(self) -> None:
        f = OneEuroFilter(1.0, 0.0)
        f(5.0, t=0.0)
        f(6.0, t=0.1)
        f.reset()
        assert f(10.0, t=1.0) == 10.0


def test_no_circular_import() -> None:
    """head_pose_estimator は filters から OneEuroFilter を取り、gaze_estimator を
    import しない（循環回避）。mediapipe 無しで import できることを保証する。"""
    import src.filters as flt
    import src.head_pose_estimator as hpe

    assert hpe.OneEuroFilter is flt.OneEuroFilter
