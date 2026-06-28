"""HotkeyController のテスト — キー判定ロジック（pynput非依存）"""

from src.hotkeys import HotkeyController


class TestHotkeyController:
    def test_toggle_sets_pending_once(self) -> None:
        h = HotkeyController(toggle_char="p")
        assert h.poll_toggle() is False
        h._handle_char("p")
        assert h.poll_toggle() is True
        # 一度消費したら False
        assert h.poll_toggle() is False

    def test_quit_char_sets_should_quit(self) -> None:
        h = HotkeyController(quit_chars=("q",))
        assert h.should_quit is False
        h._handle_char("q")
        assert h.should_quit is True

    def test_esc_sets_should_quit(self) -> None:
        h = HotkeyController()
        h._handle_char(None, is_esc=True)
        assert h.should_quit is True

    def test_unrelated_char_noop(self) -> None:
        h = HotkeyController(toggle_char="p", quit_chars=("q",))
        h._handle_char("z")
        assert h.should_quit is False
        assert h.poll_toggle() is False

    def test_case_insensitive(self) -> None:
        h = HotkeyController(toggle_char="p", quit_chars=("q",))
        h._handle_char("P")
        assert h.poll_toggle() is True
        h._handle_char("Q")
        assert h.should_quit is True

    def test_none_char_noop(self) -> None:
        h = HotkeyController()
        h._handle_char(None)
        assert h.should_quit is False
        assert h.poll_toggle() is False

    def test_reset_key(self) -> None:
        h = HotkeyController(reset_char="r")
        assert h.poll_reset() is False
        h._handle_char("r")
        assert h.poll_reset() is True
        # 一度消費したら False
        assert h.poll_reset() is False

    def test_custom_keys(self) -> None:
        h = HotkeyController(toggle_char="v", quit_chars=("x", "e"))
        h._handle_char("v")
        assert h.poll_toggle() is True
        h._handle_char("e")
        assert h.should_quit is True
