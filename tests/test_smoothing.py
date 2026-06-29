"""MotionStabilizer のテスト — 注視デッドゾーン・サッケード即追従・間引き確定"""

from src.smoothing import MotionStabilizer


def _stab() -> MotionStabilizer:
    # 決定的な値で検証
    return MotionStabilizer(deadzone=10.0, saccade=100.0, commit_interval=0.1, commit_ratio=0.5)


class TestMotionStabilizer:
    def test_first_update_returns_input(self) -> None:
        s = _stab()
        assert s.update(100.0, 100.0, 0.0) == (100.0, 100.0)
        assert s.target == (100.0, 100.0)

    def test_deadzone_holds_target(self) -> None:
        """デッドゾーン内の微小な揺れは無視され注視点を保持する"""
        s = _stab()
        s.update(100.0, 100.0, 0.0)
        # 5px の揺れ（< deadzone 10）→ 変化なし
        assert s.update(105.0, 100.0, 1.0) == (100.0, 100.0)
        assert s.update(100.0, 105.0, 2.0) == (100.0, 100.0)

    def test_saccade_snaps_immediately(self) -> None:
        """大きな移動は間隔に関係なく即追従する"""
        s = _stab()
        s.update(100.0, 100.0, 0.0)
        # 200px の移動（>= saccade 100）→ 即スナップ（interval未経過でも）
        assert s.update(300.0, 100.0, 0.001) == (300.0, 100.0)

    def test_medium_move_is_decimated(self) -> None:
        """中間移動は commit_interval ごとに commit_ratio 分だけ確定する"""
        s = _stab()
        s.update(0.0, 0.0, 0.0)
        # interval 未経過 → データを間引いて据え置き
        assert s.update(50.0, 0.0, 0.05) == (0.0, 0.0)
        # interval 経過 → 50 の 0.5 = 25 だけ寄る
        assert s.update(50.0, 0.0, 0.15) == (25.0, 0.0)
        # さらに経過 → 残り25 の 0.5 = 12.5 寄って 37.5
        assert s.update(50.0, 0.0, 0.30) == (37.5, 0.0)

    def test_reset(self) -> None:
        s = _stab()
        s.update(10.0, 10.0, 0.0)
        s.reset()
        assert s.target is None
        # reset後の初回は入力をそのまま返す
        assert s.update(7.0, 7.0, 1.0) == (7.0, 7.0)

    def test_default_construction(self) -> None:
        """設定デフォルトでも構築でき、初回は入力を返す"""
        s = MotionStabilizer()
        assert s.update(500.0, 300.0, 0.0) == (500.0, 300.0)
