"""VirtualCursor のテスト — スプライト生成と追従スムージング（Qt非依存）"""

import numpy as np
import pytest

from src import config
from src.virtual_cursor import exp_smooth, make_cursor_sprite


class TestMakeCursorSprite:
    """広範囲ブラー円形スプライトの生成テスト"""

    def test_shape_and_dtype(self) -> None:
        size = 120
        sprite = make_cursor_sprite(size, (0, 200, 255))
        assert sprite.shape == (size, size, 4)
        assert sprite.dtype == np.uint8

    def test_rgb_channels_match_color(self) -> None:
        """RGBチャンネルは指定色（R,G,B順）で塗られている"""
        sprite = make_cursor_sprite(80, (10, 120, 240))
        assert np.all(sprite[..., 0] == 10)
        assert np.all(sprite[..., 1] == 120)
        assert np.all(sprite[..., 2] == 240)

    def test_center_opaque_corners_transparent(self) -> None:
        """中心は不透明・四隅はほぼ透明（円形＋有限の広がり）"""
        size = 160
        sprite = make_cursor_sprite(size, (0, 200, 255), max_alpha=0.85)
        alpha = sprite[..., 3]
        c = size // 2
        assert alpha[c, c] >= 180  # 中心は濃い
        for corner in [alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]]:
            assert corner <= 10  # 四隅はほぼ透明

    def test_radial_falloff(self) -> None:
        """中心から外側へアルファが単調に減衰する（広範囲ブラー）"""
        size = 160
        sprite = make_cursor_sprite(size, (0, 200, 255))
        alpha = sprite[..., 3].astype(int)
        c = size // 2
        center = alpha[c, c]
        mid = alpha[c, c + int(size * 0.2)]
        outer = alpha[c, c + int(size * 0.42)]
        assert center > mid > outer

    def test_horizontal_symmetry(self) -> None:
        """左右対称（円形）"""
        size = 160
        sprite = make_cursor_sprite(size, (0, 200, 255))
        alpha = sprite[..., 3].astype(int)
        c = size // 2
        d = int(size * 0.2)
        assert abs(alpha[c, c - d] - alpha[c, c + d]) <= 3

    def test_max_alpha_respected(self) -> None:
        """max_alpha が中心アルファの上限になる"""
        sprite = make_cursor_sprite(100, (255, 255, 255), max_alpha=0.5)
        assert sprite[..., 3].max() <= int(0.5 * 255) + 1

    def test_tiny_diameter_is_safe(self) -> None:
        """極端に小さい直径でもクラッシュしない"""
        sprite = make_cursor_sprite(2, (0, 0, 0))
        assert sprite.shape[2] == 4
        assert sprite.dtype == np.uint8

    def test_uses_config_defaults(self) -> None:
        """設定のデフォルト直径で生成できる"""
        sprite = make_cursor_sprite(
            config.VIRTUAL_CURSOR_DIAMETER, config.VIRTUAL_CURSOR_COLOR
        )
        assert sprite.shape == (
            config.VIRTUAL_CURSOR_DIAMETER,
            config.VIRTUAL_CURSOR_DIAMETER,
            4,
        )


class TestExpSmooth:
    """フレームレート非依存指数スムージングのテスト"""

    def test_half_close_at_reference_dt(self) -> None:
        """responsiveness=0.5 で基準dtのとき残り距離の50%を詰める"""
        out = exp_smooth(0.0, 100.0, 0.5, 1.0 / 60.0, ref_dt=1.0 / 60.0)
        assert out == pytest.approx(50.0, abs=1e-6)

    def test_zero_dt_returns_current(self) -> None:
        assert exp_smooth(7.0, 100.0, 0.5, 0.0) == 7.0

    def test_negative_dt_returns_current(self) -> None:
        assert exp_smooth(7.0, 100.0, 0.5, -0.5) == 7.0

    def test_full_responsiveness_jumps_to_target(self) -> None:
        assert exp_smooth(0.0, 100.0, 1.0, 1.0 / 60.0) == 100.0

    def test_zero_responsiveness_stays(self) -> None:
        assert exp_smooth(20.0, 100.0, 0.0, 1.0 / 60.0) == 20.0

    def test_monotonic_no_overshoot(self) -> None:
        """繰り返し適用で単調に近づき、目標を超えない"""
        x = 0.0
        prev = x
        for _ in range(200):
            x = exp_smooth(x, 100.0, 0.3, 1.0 / 60.0)
            assert x >= prev          # 単調増加
            assert x <= 100.0 + 1e-9  # オーバーシュートしない
            prev = x
        assert x == pytest.approx(100.0, abs=1.0)

    def test_frame_rate_independence(self) -> None:
        """1ステップdtと2分割dtで到達点がほぼ一致する"""
        one_step = exp_smooth(0.0, 100.0, 0.5, 1.0 / 30.0)
        half = exp_smooth(0.0, 100.0, 0.5, 1.0 / 60.0)
        two_step = exp_smooth(half, 100.0, 0.5, 1.0 / 60.0)
        assert one_step == pytest.approx(two_step, abs=1e-6)
