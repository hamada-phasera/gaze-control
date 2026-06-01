"""GazeEstimator のテスト"""

import numpy as np
import pytest

from src import config
from src.gaze_estimator import GazeEstimator, OneEuroFilter


class TestOneEuroFilter:
    """One Euro Filter のテスト"""

    def test_first_value_passthrough(self) -> None:
        """初回の値はそのまま返る"""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        assert f(5.0, t=0.0) == 5.0

    def test_smoothing_effect(self) -> None:
        """ノイジーな入力が平滑化される"""
        f = OneEuroFilter(min_cutoff=0.3, beta=0.0)
        results = []
        for i in range(30):
            # 0.5を中心にノイズ±0.1
            noisy = 0.5 + (0.1 if i % 2 == 0 else -0.1)
            results.append(f(noisy, t=i / 30.0))

        # 後半の値は0.5付近に収束しているはず
        last_5 = results[-5:]
        for v in last_5:
            assert abs(v - 0.5) < 0.05

    def test_fast_movement_tracking(self) -> None:
        """素早い動きにはbetaにより追従する"""
        f = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        f(0.0, t=0.0)
        f(0.0, t=0.1)
        # 急に大きく動く
        result = f(1.0, t=0.2)
        # beta > 0 なので、0にとどまらず追従を開始
        assert result > 0.2

    def test_reset(self) -> None:
        """リセット後は初期状態に戻る"""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f(5.0, t=0.0)
        f(6.0, t=0.1)
        f.reset()
        # リセット後の初回はそのまま返る
        assert f(10.0, t=1.0) == 10.0


class TestGazeEstimator:
    """GazeEstimator の基本テスト"""

    def test_init(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)
        assert not estimator.is_calibrated
        assert estimator.sensitivity == config.DEFAULT_SENSITIVITY
        estimator.release()

    def test_sensitivity_clamp(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)
        estimator.sensitivity = 0.5
        assert estimator.sensitivity == 1.0
        estimator.sensitivity = 15.0
        assert estimator.sensitivity == 10.0
        estimator.sensitivity = 7.0
        assert estimator.sensitivity == 7.0
        estimator.release()

    def test_process_frame_no_face(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.process_frame(frame)
        assert result is None
        estimator.release()

    def test_compute_calibration_insufficient_points(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)
        result = estimator.compute_calibration(
            [(0.3, 0.3), (0.5, 0.5)],
            [(500.0, 300.0), (960.0, 540.0)],
        )
        assert result is False
        assert not estimator.is_calibrated
        estimator.release()

    def test_compute_calibration_success(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        ]
        screen_pts = [
            (192.0, 108.0), (960.0, 108.0), (1728.0, 108.0),
            (192.0, 540.0), (960.0, 540.0), (1728.0, 540.0),
            (192.0, 972.0), (960.0, 972.0), (1728.0, 972.0),
        ]

        result = estimator.compute_calibration(gaze_pts, screen_pts)
        assert result is True
        assert estimator.is_calibrated
        assert estimator._calib_coeff_x is not None
        assert len(estimator._calib_coeff_x) == 6
        estimator.release()

    def test_set_calibration(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)
        matrix = np.array([[1920.0, 0.0], [0.0, 1080.0]])
        offset = np.array([0.0, 0.0])
        estimator.set_calibration(matrix, offset)
        assert estimator.is_calibrated
        estimator.release()

    def test_poly_features(self) -> None:
        features = GazeEstimator._poly_features(0.3, 0.7)
        expected = np.array([0.3, 0.7, 0.09, 0.49, 0.21, 1.0])
        np.testing.assert_allclose(features, expected, atol=1e-10)

    def test_compute_calibration_polynomial_accuracy(self) -> None:
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = []
        screen_pts = []
        for gx in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for gy in [0.1, 0.3, 0.5, 0.7, 0.9]:
                gaze_pts.append((gx, gy))
                sx = 1920.0 * gx + 100.0 * gx * gx
                sy = 1080.0 * gy + 80.0 * gy * gy
                screen_pts.append((sx, sy))

        result = estimator.compute_calibration(gaze_pts, screen_pts)
        assert result is True

        test_gx, test_gy = 0.4, 0.6
        features = GazeEstimator._poly_features(test_gx, test_gy)
        est_x = float(features @ estimator._calib_coeff_x)
        est_y = float(features @ estimator._calib_coeff_y)

        expected_x = 1920.0 * test_gx + 100.0 * test_gx * test_gx
        expected_y = 1080.0 * test_gy + 80.0 * test_gy * test_gy

        np.testing.assert_allclose(est_x, expected_x, atol=5.0)
        np.testing.assert_allclose(est_y, expected_y, atol=5.0)
        estimator.release()
