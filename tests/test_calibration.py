"""キャリブレーション数値テスト — 多項式キャリブレーション + IQR外れ値除去"""

import numpy as np
import pytest

from src.calibration import CalibrationOverlay
from src.gaze_estimator import GazeEstimator


class TestCalibration:
    """キャリブレーション計算の数値テスト"""

    def test_linear_calibration(self) -> None:
        """線形なデータでも多項式キャリブレーションが成功する"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75),
        ]
        screen_pts = [
            (0.0, 0.0), (1920.0, 0.0), (0.0, 1080.0), (1920.0, 1080.0),
            (960.0, 540.0), (480.0, 270.0), (1440.0, 810.0), (480.0, 810.0),
        ]

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is True
        assert estimator.is_calibrated

        # 多項式係数で推定
        features = GazeEstimator._poly_features(0.5, 0.5)
        est_x = float(features @ estimator._calib_coeff_x)
        est_y = float(features @ estimator._calib_coeff_y)

        np.testing.assert_allclose(est_x, 960.0, atol=10.0)
        np.testing.assert_allclose(est_y, 540.0, atol=10.0)
        estimator.release()

    def test_offset_calibration(self) -> None:
        """オフセット付き変換"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75),
        ]
        # 全体に100pxオフセット
        screen_pts = [
            (100.0, 100.0), (2020.0, 100.0), (100.0, 1180.0), (2020.0, 1180.0),
            (1060.0, 640.0), (580.0, 370.0), (1540.0, 910.0), (580.0, 910.0),
        ]

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is True

        # 原点でのオフセットを検証
        features_origin = GazeEstimator._poly_features(0.0, 0.0)
        est_x = float(features_origin @ estimator._calib_coeff_x)
        est_y = float(features_origin @ estimator._calib_coeff_y)
        np.testing.assert_allclose(est_x, 100.0, atol=10.0)
        np.testing.assert_allclose(est_y, 100.0, atol=10.0)
        estimator.release()

    def test_scaled_calibration(self) -> None:
        """スケール変換のテスト"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
            (0.5, 0.5), (0.35, 0.35), (0.65, 0.65), (0.35, 0.65),
        ]
        screen_pts = [
            (0.0, 0.0), (1920.0, 0.0), (0.0, 1080.0), (1920.0, 1080.0),
            (960.0, 540.0), (480.0, 270.0), (1440.0, 810.0), (480.0, 810.0),
        ]

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is True
        assert estimator.is_calibrated
        estimator.release()

    def test_min_points_boundary(self) -> None:
        """最小点数(8点)ちょうどで成功する"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75),
        ]
        screen_pts = [
            (0.0, 0.0), (1920.0, 0.0), (0.0, 1080.0), (1920.0, 1080.0),
            (960.0, 540.0), (480.0, 270.0), (1440.0, 810.0), (480.0, 810.0),
        ]

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is True
        estimator.release()

    def test_below_min_points_fails(self) -> None:
        """最小点数未満（7点）では失敗する"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.5), (0.25, 0.25), (0.75, 0.75),
        ]
        screen_pts = [
            (0.0, 0.0), (1920.0, 0.0), (0.0, 1080.0), (1920.0, 1080.0),
            (960.0, 540.0), (480.0, 270.0), (1440.0, 810.0),
        ]

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is False
        assert not estimator.is_calibrated
        estimator.release()

    def test_nonlinear_mapping(self) -> None:
        """非線形マッピングの多項式フィットテスト"""
        estimator = GazeEstimator(1920, 1080, skip_model=True)

        gaze_pts = []
        screen_pts = []
        for gx in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for gy in [0.1, 0.3, 0.5, 0.7, 0.9]:
                gaze_pts.append((gx, gy))
                sx = 1920.0 * gx + 200.0 * gx * gx
                sy = 1080.0 * gy + 150.0 * gy * gy
                screen_pts.append((sx, sy))

        success = estimator.compute_calibration(gaze_pts, screen_pts)
        assert success is True

        # 補間テスト
        test_gx, test_gy = 0.6, 0.4
        features = GazeEstimator._poly_features(test_gx, test_gy)
        est_x = float(features @ estimator._calib_coeff_x)
        est_y = float(features @ estimator._calib_coeff_y)

        expected_x = 1920.0 * test_gx + 200.0 * test_gx * test_gx
        expected_y = 1080.0 * test_gy + 150.0 * test_gy * test_gy

        np.testing.assert_allclose(est_x, expected_x, atol=5.0)
        np.testing.assert_allclose(est_y, expected_y, atol=5.0)
        estimator.release()


class TestOutlierFiltering:
    """CalibrationOverlay.filter_outliers のテスト"""

    def test_no_outliers(self) -> None:
        """外れ値なしデータではそのまま平均が返る"""
        samples = [(0.5, 0.5), (0.51, 0.49), (0.49, 0.51), (0.50, 0.50), (0.52, 0.48)]
        result = CalibrationOverlay._filter_outliers(samples)
        assert result is not None
        np.testing.assert_allclose(result[0], 0.504, atol=0.01)
        np.testing.assert_allclose(result[1], 0.496, atol=0.01)

    def test_with_outliers(self) -> None:
        """外れ値があればIQRで除去される"""
        samples = [
            (0.5, 0.5), (0.51, 0.49), (0.49, 0.51), (0.50, 0.50), (0.52, 0.48),
            (0.5, 0.5), (0.51, 0.49),
            (10.0, 10.0),  # 外れ値
        ]
        result = CalibrationOverlay._filter_outliers(samples)
        assert result is not None
        # 外れ値が除去されているので、0.5付近の値になるはず
        assert abs(result[0] - 0.5) < 0.05
        assert abs(result[1] - 0.5) < 0.05

    def test_insufficient_samples(self) -> None:
        """サンプル不足ではNone"""
        samples = [(0.5, 0.5), (0.6, 0.6)]
        result = CalibrationOverlay._filter_outliers(samples)
        assert result is None

    def test_all_outliers_fallback_to_median(self) -> None:
        """全データが分散している場合でもフォールバックで値を返す"""
        samples = [
            (0.1, 0.1), (0.9, 0.9), (0.2, 0.8), (0.8, 0.2), (0.5, 0.5),
        ]
        result = CalibrationOverlay._filter_outliers(samples)
        assert result is not None
