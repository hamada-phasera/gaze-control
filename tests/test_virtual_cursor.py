"""VirtualCursor のテスト — スプライト生成と追従スムージング（Qt非依存）"""

import numpy as np
import pytest

from src import config
from src.virtual_cursor import (
    advance_cursor,
    exp_smooth,
    make_cursor_sprite,
    transit_scale,
)


class TestTransitScale:
    """サイズ・イージング（移動中は縮み、静止で膨らむ）のテスト"""

    def test_still_is_full_size(self) -> None:
        assert transit_scale(0.0, 7.5, 0.45) == 1.0

    def test_max_speed_shrinks_by_dip(self) -> None:
        assert transit_scale(7.5, 7.5, 0.45) == pytest.approx(0.55)

    def test_beyond_ref_clamped(self) -> None:
        assert transit_scale(100.0, 7.5, 0.45) == pytest.approx(0.55)

    def test_zero_dip_is_full_size(self) -> None:
        assert transit_scale(7.5, 7.5, 0.0) == 1.0

    def test_zero_ref_is_full_size(self) -> None:
        assert transit_scale(5.0, 0.0, 0.45) == 1.0

    def test_monotonic_decreasing(self) -> None:
        prev = 1.0
        for step in [0.0, 1.0, 2.0, 4.0, 7.5]:
            s = transit_scale(step, 7.5, 0.45)
            assert s <= prev + 1e-9
            prev = s


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


class TestAdvanceCursor:
    """速度キャップ＋画面内クランプのテスト"""

    def test_speed_cap_limits_step(self) -> None:
        """遠い目標でも1ティックの移動量は max_step 以下（開始点は画面内）"""
        max_step = 7.5
        sx, sy = 100.0, 100.0
        nx, ny = advance_cursor(
            sx, sy, 3000.0, 100.0,
            responsiveness=1.0, dt=1.0 / 120.0, max_step=max_step,
            screen_w=10000.0, screen_h=10000.0, half_w=90.0, half_h=90.0,
        )
        dist = ((nx - sx) ** 2 + (ny - sy) ** 2) ** 0.5
        assert dist <= max_step + 1e-9

    def test_no_cap_when_zero(self) -> None:
        """max_step=0 ならキャップ無し（exp_smoothそのまま, クランプ無効化）"""
        nx, _ = advance_cursor(
            0.0, 0.0, 100.0, 0.0,
            responsiveness=0.5, dt=1.0 / 60.0, max_step=0.0,
            screen_w=10000.0, screen_h=10000.0, half_w=0.0, half_h=0.0,
        )
        assert nx == pytest.approx(exp_smooth(0.0, 100.0, 0.5, 1.0 / 60.0))

    def test_clamp_keeps_sprite_on_screen(self) -> None:
        """中心がスプライト半径ぶん内側にクランプされ、全体が画面内に残る"""
        nx, ny = advance_cursor(
            5.0, 5.0, 0.0, 0.0,
            responsiveness=1.0, dt=1.0 / 60.0, max_step=0.0,
            screen_w=1000.0, screen_h=800.0, half_w=90.0, half_h=90.0,
        )
        assert nx >= 90.0 and ny >= 90.0

    def test_clamp_upper_bound(self) -> None:
        nx, ny = advance_cursor(
            0.0, 0.0, 5000.0, 5000.0,
            responsiveness=1.0, dt=1.0 / 60.0, max_step=0.0,
            screen_w=1000.0, screen_h=800.0, half_w=90.0, half_h=90.0,
        )
        assert nx <= 1000.0 - 90.0 and ny <= 800.0 - 90.0

    def test_converges_to_target_over_time(self) -> None:
        """十分なティック数で目標へ収束する（キャップ有りでも）"""
        x, y = 0.0, 0.0
        for _ in range(2000):
            x, y = advance_cursor(
                x, y, 1500.0, 400.0,
                responsiveness=0.2, dt=1.0 / 120.0, max_step=7.5,
                screen_w=3024.0, screen_h=1964.0, half_w=90.0, half_h=90.0,
            )
        assert x == pytest.approx(1500.0, abs=1.0)
        assert y == pytest.approx(400.0, abs=1.0)
