"""スレッド化カメラキャプチャ — 取得と推論を分離してFPSを底上げする

`cv2.VideoCapture.read()` はブロッキングで、メインループ内で呼ぶと
「取得待ち」が推論時間に直列で積み上がる。本モジュールは取得を専用スレッドに
逃がし、常に**最新フレームだけ**を返す（古いフレームは破棄する）ことで
実効FPSと体感遅延を改善する。

`cv2.VideoCapture` を直接受け取る依存性注入設計のため、フェイクのキャプチャを
渡してカメラ無しでもユニットテストできる。
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np


class CameraStream:
    """`cv2.VideoCapture` をバックグラウンドスレッドで読み続けるラッパー。

    `read()` はロックを取って最新フレームのコピー参照を返すだけなので、
    メインループは取得待ちでブロックされない。
    """

    def __init__(self, capture: object, poll_sleep: float = 0.001) -> None:
        """
        Args:
            capture: `read() -> (bool, frame)` と `release()`, `isOpened()` を持つ
                オブジェクト（通常は開いた cv2.VideoCapture）。
            poll_sleep: 読み取り失敗時のバックオフ (秒)。
        """
        self._cap = capture
        self._poll_sleep = poll_sleep
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        """取得スレッドを開始する。"""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = bool(ret)
                self._frame = frame
            if not ret:
                time.sleep(self._poll_sleep)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """最新フレームを返す。まだ1枚も取得できていなければ (False, None)。"""
        with self._lock:
            if self._frame is None:
                return (False, None)
            return (self._ret, self._frame)

    def isOpened(self) -> bool:  # noqa: N802 — cv2.VideoCapture 互換
        try:
            return bool(self._cap.isOpened())
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        """取得スレッドを停止する（キャプチャは解放しない）。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def release(self) -> None:
        """スレッドを止め、基盤のキャプチャを解放する。"""
        self.stop()
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass
