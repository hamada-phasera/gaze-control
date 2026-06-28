"""FixationTracker のテスト — 履歴重心 + ドウェルで確定、移動中は保持"""

from src.fixation import FixationTracker


def _ft() -> FixationTracker:
    return FixationTracker(dwell_time=0.3, radius=110.0, min_move=55.0)


class TestFixationTracker:
    def test_first_update_commits_input(self) -> None:
        f = _ft()
        tx, ty, committed = f.update(500.0, 500.0, 0.0)
        assert (tx, ty) == (500.0, 500.0)
        assert committed is True

    def test_commits_after_dwell_on_steady_gaze(self) -> None:
        """新しい場所を dwell_time とどまると、その重心へ確定する"""
        f = _ft()
        f.update(500.0, 500.0, 0.0)  # 初期ターゲット

        committed_target = None
        for i in range(1, 60):
            t = i * 0.02
            # (1000,800) 付近に小さなジッタ（半径~10px）で滞留
            x = 1000.0 + (i % 3 - 1) * 6.0
            y = 800.0 + (i % 2) * 6.0
            tx, ty, committed = f.update(x, y, t)
            if committed:
                committed_target = (tx, ty)
                break

        assert committed_target is not None
        assert abs(committed_target[0] - 1000.0) < 60.0
        assert abs(committed_target[1] - 800.0) < 60.0

    def test_holds_target_while_moving(self) -> None:
        """視線が動き続ける間は確定せず、ターゲットを保持する"""
        f = _ft()
        f.update(0.0, 0.0, 0.0)

        any_commit = False
        last = None
        for i in range(1, 40):
            t = i * 0.02
            tx, ty, committed = f.update(i * 120.0, 0.0, t)  # 大きく動き続ける
            any_commit = any_commit or committed
            last = (tx, ty)

        assert any_commit is False
        assert last == (0.0, 0.0)  # 初期ターゲットを保持

    def test_no_recommit_for_tiny_move(self) -> None:
        """min_move 未満の重心移動では再確定しない"""
        f = _ft()
        f.update(1000.0, 1000.0, 0.0)

        tx = ty = None
        for i in range(1, 60):
            t = i * 0.02
            # 目標(1000,1000)からごく近い場所（~15px）に滞留
            tx, ty, _ = f.update(1010.0 + (i % 2) * 4.0, 1005.0, t)

        assert (tx, ty) == (1000.0, 1000.0)

    def test_reset(self) -> None:
        f = _ft()
        f.update(300.0, 300.0, 0.0)
        f.reset()
        assert f.target is None
        tx, ty, committed = f.update(700.0, 200.0, 1.0)
        assert (tx, ty) == (700.0, 200.0)
        assert committed is True
