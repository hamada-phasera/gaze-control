"""HeadPoseEstimator の縦アシストのテスト（mediapipe非依存）"""

from src.head_pose_estimator import HeadPoseEstimator, HeadPoseResult


def _hp() -> HeadPoseEstimator:
    return HeadPoseEstimator(1920, 1080)


class TestVerticalAssist:
    def test_zero_without_baseline(self) -> None:
        hp = _hp()
        assert hp.vertical_assist(HeadPoseResult(0.0, 0.0, 0.0), 45.0) == 0.0

    def test_looking_down_is_positive(self) -> None:
        hp = _hp()
        hp.set_baseline(HeadPoseResult(yaw=0.0, pitch=10.0, roll=0.0))
        # 下を向く（pitchが基準より小）→ 正（下へ）
        assert hp.vertical_assist(HeadPoseResult(0.0, 5.0, 0.0), 45.0) == 225.0

    def test_looking_up_is_negative(self) -> None:
        hp = _hp()
        hp.set_baseline(HeadPoseResult(yaw=0.0, pitch=10.0, roll=0.0))
        assert hp.vertical_assist(HeadPoseResult(0.0, 15.0, 0.0), 45.0) == -225.0

    def test_zero_gain(self) -> None:
        hp = _hp()
        hp.set_baseline(HeadPoseResult(yaw=0.0, pitch=10.0, roll=0.0))
        assert hp.vertical_assist(HeadPoseResult(0.0, 5.0, 0.0), 0.0) == 0.0

    def test_horizontal_yaw_ignored(self) -> None:
        """ヨー（左右）は縦アシストに影響しない"""
        hp = _hp()
        hp.set_baseline(HeadPoseResult(yaw=0.0, pitch=10.0, roll=0.0))
        a = hp.vertical_assist(HeadPoseResult(yaw=30.0, pitch=5.0, roll=0.0), 45.0)
        assert a == 225.0  # yaw=30 でも変わらない
