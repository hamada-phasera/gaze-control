"""グローバルホットキー — ウィンドウ非表示でも終了/表示切替を受け付ける

プレビューウィンドウを出すと、人はつい画面内の自分の目を見てしまい、視線が
そちらに引っ張られてカーソル制御が乱れる。そこで既定ではウィンドウを出さず、
必要な時だけキーでプレビューをトグルできるようにする。ウィンドウが無いと
`cv2.waitKey` ではキーを拾えないため、`pynput` のグローバルリスナーを使う。

`pynput` が無い環境では `start()` が False を返し、呼び出し側は cv2 の
ウィンドウ + waitKey にフォールバックする。キー判定ロジック (`_handle_char`) は
純粋関数でユニットテスト可能。
"""

from __future__ import annotations

from typing import Iterable, Optional


class HotkeyController:
    """終了・プレビュー切替のグローバルホットキー。"""

    def __init__(
        self,
        toggle_char: str = "p",
        quit_chars: Iterable[str] = ("q",),
        reset_char: str = "r",
        recenter_char: str = "c",
    ) -> None:
        self._toggle_char = toggle_char.lower()
        self._quit_chars = {c.lower() for c in quit_chars}
        self._reset_char = reset_char.lower()
        self._recenter_char = recenter_char.lower()
        self._should_quit = False
        self._toggle_pending = False
        self._reset_pending = False
        self._recenter_pending = False
        self._listener = None

    @property
    def should_quit(self) -> bool:
        return self._should_quit

    def poll_toggle(self) -> bool:
        """前回のpoll以降にトグルが要求されていれば True を一度だけ返す。"""
        if self._toggle_pending:
            self._toggle_pending = False
            return True
        return False

    def poll_reset(self) -> bool:
        """前回のpoll以降にリセットが要求されていれば True を一度だけ返す。"""
        if self._reset_pending:
            self._reset_pending = False
            return True
        return False

    def poll_recenter(self) -> bool:
        """前回のpoll以降に再センタリングが要求されていれば True を一度だけ返す。"""
        if self._recenter_pending:
            self._recenter_pending = False
            return True
        return False

    def _handle_char(self, ch: Optional[str], is_esc: bool = False) -> None:
        """押下キーからフラグを更新する純粋ロジック（テスト対象）。"""
        if is_esc:
            self._should_quit = True
            return
        if ch is None:
            return
        ch = ch.lower()
        if ch in self._quit_chars:
            self._should_quit = True
        elif ch == self._toggle_char:
            self._toggle_pending = True
        elif ch == self._reset_char:
            self._reset_pending = True
        elif ch == self._recenter_char:
            self._recenter_pending = True

    def start(self) -> bool:
        """pynputリスナーを開始する。pynput不在なら False。"""
        try:
            from pynput import keyboard  # type: ignore
        except Exception:
            return False

        def on_press(key) -> None:
            try:
                if key == keyboard.Key.esc:
                    self._handle_char(None, is_esc=True)
                else:
                    self._handle_char(getattr(key, "char", None))
            except Exception:
                pass

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception:
            self._listener = None
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
