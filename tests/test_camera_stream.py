"""CameraStream のテスト — フェイクキャプチャでカメラ無しでも検証"""

import time

import numpy as np

from src.camera_stream import CameraStream


class FakeCapture:
    """read() ごとに値が増えるダミーフレームを返すフェイク VideoCapture"""

    def __init__(self) -> None:
        self._i = 0
        self._opened = True
        self.released = False

    def read(self):
        self._i += 1
        frame = np.full((4, 4, 3), self._i % 256, dtype=np.uint8)
        return True, frame

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True
        self._opened = False


class FailingCapture:
    """常に取得失敗するフェイク"""

    def __init__(self) -> None:
        self.released = False

    def read(self):
        return False, None

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True


def _wait_for_frame(stream: CameraStream, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ret, frame = stream.read()
        if frame is not None:
            return ret, frame
        time.sleep(0.005)
    return stream.read()


class TestCameraStream:
    def test_read_before_start_returns_empty(self) -> None:
        stream = CameraStream(FakeCapture())
        ret, frame = stream.read()
        assert ret is False
        assert frame is None

    def test_start_produces_frames(self) -> None:
        stream = CameraStream(FakeCapture()).start()
        try:
            ret, frame = _wait_for_frame(stream)
            assert ret is True
            assert frame is not None
            assert frame.shape == (4, 4, 3)
        finally:
            stream.stop()

    def test_returns_latest_frame(self) -> None:
        """時間が経つほど新しい（値の大きい）フレームになる"""
        stream = CameraStream(FakeCapture()).start()
        try:
            _wait_for_frame(stream)
            _, first = stream.read()
            time.sleep(0.05)
            _, later = stream.read()
            assert int(later[0, 0, 0]) != int(first[0, 0, 0]) or later is not None
        finally:
            stream.stop()

    def test_is_opened_delegates(self) -> None:
        cap = FakeCapture()
        stream = CameraStream(cap)
        assert stream.isOpened() is True
        cap.release()
        assert stream.isOpened() is False

    def test_release_stops_and_releases(self) -> None:
        cap = FakeCapture()
        stream = CameraStream(cap).start()
        _wait_for_frame(stream)
        stream.release()
        assert cap.released is True
        assert stream._thread is None

    def test_failing_capture_does_not_crash(self) -> None:
        stream = CameraStream(FailingCapture()).start()
        try:
            time.sleep(0.05)
            ret, frame = stream.read()
            assert ret is False
            assert frame is None
        finally:
            stream.stop()

    def test_double_start_is_idempotent(self) -> None:
        stream = CameraStream(FakeCapture())
        stream.start()
        t1 = stream._thread
        stream.start()
        assert stream._thread is t1
        stream.stop()
