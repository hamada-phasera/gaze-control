"""頭部正規化（虹彩比率のヨー/ピッチ補正）のテスト。

GazeEstimator は `import mediapipe` するため、ヘッドレス環境では
ダミーモジュールを注入してからインポートする（モデルは生成しない）。
head_normalize_ratio は純粋関数なのでインスタンス化は不要。
"""

import sys
import types

try:  # 実機に mediapipe があれば本物を優先（スタブで隠さない）
    import mediapipe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - 環境依存
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from src.gaze_estimator import GazeEstimator  # noqa: E402

_norm = GazeEstimator.head_normalize_ratio


class TestHeadNormalizeRatio:
    def test_strength_zero_is_noop(self) -> None:
        x, y = _norm(0.6, 0.4, 10.0, -5.0, 0.0, 0.0067, 0.0067)
        assert x == 0.6
        assert y == 0.4

    def test_positive_yaw_reduces_x(self) -> None:
        # dyaw>0（基準より右を向く）→ x を戻す方向に減らす
        x, _ = _norm(0.6, 0.5, 10.0, 0.0, 1.0, 0.01, 0.01)
        assert x == 0.6 - 0.1  # 1.0 * 0.01 * 10

    def test_positive_pitch_reduces_y(self) -> None:
        _, y = _norm(0.5, 0.6, 0.0, 10.0, 1.0, 0.01, 0.01)
        assert y == 0.6 - 0.1

    def test_strength_scales_linearly(self) -> None:
        x, _ = _norm(0.5, 0.5, 10.0, 0.0, 0.5, 0.01, 0.01)
        assert x == 0.5 - 0.05

    def test_negative_strength_flips_sign(self) -> None:
        x, _ = _norm(0.5, 0.5, 10.0, 0.0, -1.0, 0.01, 0.01)
        assert x == 0.5 + 0.1

    def test_axes_are_independent(self) -> None:
        # ヨーだけ動いても y は不変、ピッチだけ動いても x は不変
        x, y = _norm(0.5, 0.5, 10.0, 0.0, 1.0, 0.01, 0.02)
        assert y == 0.5
        x2, y2 = _norm(0.5, 0.5, 0.0, 10.0, 1.0, 0.01, 0.02)
        assert x2 == 0.5
        assert y2 == 0.5 - 0.2
