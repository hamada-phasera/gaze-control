"""FixationTracker のテスト — ロック＆エスケープ（近傍は完全静止、遠方注視で乗り換え）"""

from src.fixation import FixationTracker


def _ft() -> FixationTracker:
    return FixationTracker(dwell_time=0.3, radius=80.0, release_radius=110.0)


class TestFixationTracker:
    def test_first_update_locks_input(self) -> None:
        f = _ft()
        tx, ty, committed = f.update(500.0, 500.0, 0.0)
        assert (tx, ty) == (500.0, 500.0)
        assert committed is True

    def test_holds_perfectly_still_within_release_radius(self) -> None:
        """ロック近傍（release_radius内）の揺れ・ドリフトでは一切動かない"""
        f = _ft()
        f.update(1000.0, 1000.0, 0.0)
        tx = ty = None
        for i in range(1, 80):
            t = i * 0.02
            # ±45px のジッタ（< release 110）
            x = 1000.0 + (i % 5 - 2) * 22.0
            y = 1000.0 + (i % 3 - 1) * 22.0
            tx, ty, committed = f.update(x, y, t)
            assert committed is False
        assert (tx, ty) == (1000.0, 1000.0)  # 完全に固定

    def test_switches_to_far_steady_gaze(self) -> None:
        """release_radius の外を dwell とどまると、その重心へ乗り換える"""
        f = _ft()
        f.update(0.0, 0.0, 0.0)  # lock at origin
        committed_target = None
        for i in range(1, 60):
            t = i * 0.02
            x = 600.0 + (i % 3 - 1) * 8.0
            y = 400.0 + (i % 2) * 8.0
            tx, ty, committed = f.update(x, y, t)
            if committed:
                committed_target = (tx, ty)
                break
        assert committed_target is not None
        assert abs(committed_target[0] - 600.0) < 60.0
        assert abs(committed_target[1] - 400.0) < 60.0

    def test_scanning_does_not_switch(self) -> None:
        """遠くを見ても、動き続けている間は乗り換えない（ロック保持）"""
        f = _ft()
        f.update(0.0, 0.0, 0.0)
        any_commit = False
        last = None
        for i in range(1, 40):
            t = i * 0.02
            tx, ty, committed = f.update(i * 100.0, 0.0, t)  # 動き続け（spread大）
            any_commit = any_commit or committed
            last = (tx, ty)
        assert any_commit is False
        assert last == (0.0, 0.0)

    def test_reset(self) -> None:
        f = _ft()
        f.update(300.0, 300.0, 0.0)
        f.reset()
        assert f.target is None
        tx, ty, committed = f.update(700.0, 200.0, 1.0)
        assert (tx, ty) == (700.0, 200.0)
        assert committed is True
