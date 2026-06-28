"""3D眼内ベクトル（④-lite）の正規化テスト。

- 頭部回転（ヨー/ピッチ）で虹彩比率が不変であること
- 左右の目（目頭→目尻が逆向き）で同じ視線が同符号になること（相殺しない）
- z=0（正面）では非反転側の目が2Dと一致すること

GazeEstimator は `import mediapipe` するため、ヘッドレスではスタブを注入。
iris_ratio_3d は純粋関数なのでインスタンス化は不要。
"""

import sys
import types

import numpy as np
import pytest

try:  # 実機に mediapipe があれば本物を優先
    import mediapipe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - 環境依存
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from src.gaze_estimator import GazeEstimator  # noqa: E402

_r3 = GazeEstimator.iris_ratio_3d


def _eye(side: str, gaze=(0.0, 0.0)):
    """正面顔の片目を合成する。

    side="right_img": 画像右側の目（目尻が +x）
    side="left_img" : 画像左側の目（目尻が -x、2Dだと符号が逆になる側）
    gaze=(gx, gy): 虹彩を目中心から (gx, gy) ずらす（+x=画像右, +y=画像下）。
    返り値は (iris, inner, outer, top, bottom) の3D点。
    """
    sx = 1.0 if side == "right_img" else -1.0
    inner = np.array([0.0, 0.0, 0.0])
    outer = np.array([sx * 0.04, 0.0, 0.0])
    center = (inner + outer) / 2.0
    top = center + np.array([0.0, -0.01, 0.0])
    bottom = center + np.array([0.0, 0.01, 0.0])
    iris = center + np.array([gaze[0], gaze[1], 0.0])
    return iris, inner, outer, top, bottom


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _apply(R, pts):
    return tuple(R @ p for p in pts)


class TestSignConsistency:
    def test_look_right_both_eyes_positive_x(self):
        rx_r, _ = _r3(*_eye("right_img", gaze=(0.008, 0.0)))
        rx_l, _ = _r3(*_eye("left_img", gaze=(0.008, 0.0)))
        # 左右で目頭→目尻が逆向きでも、同じ「右を見る」で同符号（相殺しない）
        assert rx_r > 0.0
        assert rx_l > 0.0
        assert rx_r == pytest.approx(rx_l, abs=1e-9)

    def test_look_down_both_eyes_positive_y(self):
        _, ry_r = _r3(*_eye("right_img", gaze=(0.0, 0.005)))
        _, ry_l = _r3(*_eye("left_img", gaze=(0.0, 0.005)))
        assert ry_r > 0.0
        assert ry_l > 0.0
        assert ry_r == pytest.approx(ry_l, abs=1e-9)

    def test_centered_gaze_is_zero(self):
        rx, ry = _r3(*_eye("right_img", gaze=(0.0, 0.0)))
        assert rx == pytest.approx(0.0, abs=1e-9)
        assert ry == pytest.approx(0.0, abs=1e-9)


class TestRotationInvariance:
    @pytest.mark.parametrize("deg", [-40, -20, 0, 20, 40])
    def test_yaw_invariant(self, deg):
        pts = _eye("right_img", gaze=(0.008, 0.003))
        base = _r3(*pts)
        rotated = _r3(*_apply(_rot_y(np.radians(deg)), pts))
        assert rotated[0] == pytest.approx(base[0], abs=1e-9)
        assert rotated[1] == pytest.approx(base[1], abs=1e-9)

    @pytest.mark.parametrize("deg", [-30, -15, 15, 30])
    def test_pitch_invariant(self, deg):
        pts = _eye("left_img", gaze=(0.006, 0.004))
        base = _r3(*pts)
        rotated = _r3(*_apply(_rot_x(np.radians(deg)), pts))
        assert rotated[0] == pytest.approx(base[0], abs=1e-9)
        assert rotated[1] == pytest.approx(base[1], abs=1e-9)

    def test_combined_yaw_pitch_invariant(self):
        pts = _eye("right_img", gaze=(0.007, -0.004))
        base = _r3(*pts)
        R = _rot_y(np.radians(25)) @ _rot_x(np.radians(-18))
        rotated = _r3(*_apply(R, pts))
        # 横は3D投影なので複合回転でも厳密に不変
        assert rotated[0] == pytest.approx(base[0], abs=1e-9)
        # 縦は画像平面(2D)。複合の面外回転では近似不変（zノイズ回避とのトレードオフ）
        assert rotated[1] == pytest.approx(base[1], abs=0.02)


class TestBlinkClamp:
    def test_tiny_eye_height_is_clamped(self):
        from src import config
        # 目をほぼ閉じた状態（top≈bottom）で虹彩がずれている → 比率が発散しかける
        iris = np.array([0.02, 0.005, 0.0])
        inner = np.array([0.0, 0.0, 0.0])
        outer = np.array([0.04, 0.0, 0.0])
        top = np.array([0.02, -0.0001, 0.0])
        bottom = np.array([0.02, 0.0001, 0.0])   # eye_h ~ 0.0002
        rx, ry = _r3(iris, inner, outer, top, bottom)
        assert abs(ry) <= config.GAZE_RATIO_CLAMP + 1e-9
        assert abs(rx) <= config.GAZE_RATIO_CLAMP + 1e-9


class TestMatches2DOnFrontalNonFlippedEye:
    def test_matches_2d_helper_at_z0(self):
        # 非反転側（画像右の目）は z=0 で 2D の比率と一致するはず
        iris, inner, outer, top, bottom = _eye("right_img", gaze=(0.009, 0.004))

        # 2D 参照計算（_eye_ratio_2d と同じ式）
        center = (inner[:2] + outer[:2]) / 2.0
        width = np.linalg.norm(outer[:2] - inner[:2])
        d = outer[:2] - inner[:2]
        d_hat = d / np.linalg.norm(d)
        perp = np.array([-d_hat[1], d_hat[0]])
        diff = iris[:2] - center
        rx2d = float(np.dot(diff, d_hat)) / width
        ry2d = float(np.dot(diff, perp)) / np.linalg.norm(bottom[:2] - top[:2])

        rx3d, ry3d = _r3(iris, inner, outer, top, bottom)
        assert rx3d == pytest.approx(rx2d, abs=1e-9)
        assert ry3d == pytest.approx(ry2d, abs=1e-9)
