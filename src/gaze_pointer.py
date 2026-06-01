"""視線ポインター — Tkinter不使用・座標保持のみの軽量実装

デバッグウィンドウやOSカーソル自体が視覚フィードバックとなるため、
Tkinterオーバーレイは使用せず座標の保持とスムージングのみ行う。
"""

from . import config


class GazePointer:
    """視線ポインターの座標管理

    OSカーソルが視線位置に移動するため、別途オーバーレイは不要。
    座標のスムージングとDwell進捗を保持する。
    """

    def __init__(self) -> None:
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._dwell_progress: float = 0.0
        self._running = False

    def start(self) -> None:
        """ポインター追跡を開始"""
        self._running = True

    def stop(self) -> None:
        """ポインター追跡を停止"""
        self._running = False

    def update_position(self, x: float, y: float, dwell_progress: float = 0.0) -> None:
        """ポインター位置を更新する"""
        self._target_x = x
        self._target_y = y
        self._dwell_progress = dwell_progress

        # スムージング
        smooth = 0.08
        self._current_x += smooth * (self._target_x - self._current_x)
        self._current_y += smooth * (self._target_y - self._current_y)

    @property
    def current_x(self) -> float:
        return self._current_x

    @property
    def current_y(self) -> float:
        return self._current_y

    @property
    def dwell_progress(self) -> float:
        return self._dwell_progress
