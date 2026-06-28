"""一周なぞりキャリブのテスト — ウェイポイント生成（mediapipe非依存）"""

from src import config
from src.calibration import CalibrationOverlay


def _ov() -> CalibrationOverlay:
    return CalibrationOverlay(1920, 1080, gaze_callback=lambda: None)


def test_waypoints_enough_for_polynomial() -> None:
    pts = _ov()._generate_perimeter_waypoints()
    assert len(pts) >= config.CALIBRATION_MIN_POINTS


def test_waypoints_within_screen() -> None:
    for x, y in _ov()._generate_perimeter_waypoints():
        assert 0.0 <= x <= 1920.0
        assert 0.0 <= y <= 1080.0


def test_waypoints_cover_corners_and_center() -> None:
    pts = _ov()._generate_perimeter_waypoints()
    m = config.CALIBRATION_MARGIN
    x0, y0 = 1920 * m, 1080 * m
    x1, y1 = 1920 * (1 - m), 1080 * (1 - m)
    cx, cy = 960.0, 540.0

    def has(px, py):
        return any(abs(x - px) < 1.0 and abs(y - py) < 1.0 for x, y in pts)

    assert has(x0, y0)   # 左上
    assert has(x1, y1)   # 右下
    assert has(cx, cy)   # 中央


def test_draw_dot_returns_full_frame() -> None:
    frame = _ov()._draw_dot(500.0, 300.0, "1/13")
    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype.name == "uint8"
