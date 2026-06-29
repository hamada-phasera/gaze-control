"""再センタリング（中央を見て追加オフセットを自動補正）のテスト。

robust_mean_xy（外れ値に強い平均）と、増分オフセットの計算ロジックを検証。
GazeEstimator は import mediapipe するためヘッドレスではスタブを注入。
"""

import sys
import types

import pytest

try:  # 実機に mediapipe があれば本物を優先
    import mediapipe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - 環境依存
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from src.gaze_estimator import GazeEstimator  # noqa: E402

_mean = GazeEstimator.robust_mean_xy


class TestRobustMeanXY:
    def test_none_when_empty(self):
        assert _mean([]) is None

    def test_median_for_few_samples(self):
        # 5未満は中央値
        out = _mean([(10.0, 20.0), (12.0, 22.0), (11.0, 21.0)])
        assert out == (11.0, 21.0)

    def test_trims_outliers(self):
        # 大多数が (100,100) 付近で、1つだけ極端な外れ値 → 外れ値は捨てられる
        samples = [(100.0, 100.0)] * 9 + [(2000.0, -3000.0)]
        x, y = _mean(samples)
        assert x == pytest.approx(100.0, abs=1e-6)
        assert y == pytest.approx(100.0, abs=1e-6)

    def test_plain_mean_without_outliers(self):
        samples = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0), (50.0, 0.0)]
        x, y = _mean(samples)
        # trim=0.2 → 5点中4点採用。中央値30から近い順: 30,20,40,10 の平均=25
        assert x == pytest.approx(25.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)


class TestIncrementalOffset:
    """「中央を見たときの観測平均が中央に来る」増分オフセットの算術。

    new_offset = old_offset + (center - mean_observed)
    """

    def test_cancels_uniform_bias(self):
        center = (960.0, 540.0)
        # 中央を見ているのに観測は左下(=右上にバイアス)へずれている
        observed = (900.0, 600.0)
        old = (0.0, 0.0)
        dx = center[0] - observed[0]
        dy = center[1] - observed[1]
        new = (old[0] + dx, old[1] + dy)
        assert new == (60.0, -60.0)
        # 補正後、同じ観測に new を足すと中央に一致
        assert (observed[0] + new[0], observed[1] + new[1]) == center

    def test_accumulates_on_existing_offset(self):
        center = (960.0, 540.0)
        observed = (980.0, 530.0)   # 既存オフセット込みでまだ少しズレ
        old = (100.0, -50.0)
        new = (old[0] + (center[0] - observed[0]), old[1] + (center[1] - observed[1]))
        assert new == (80.0, -40.0)
