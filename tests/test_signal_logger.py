"""SignalLogger / format_row のテスト（依存なし）。"""

import os
import tempfile

from src import signal_logger
from src.signal_logger import SignalLogger, format_header, format_row


class FakeResult:
    """GazeResult 互換の最小フェイク。"""

    def __init__(self, **kw):
        self.iris_x = kw.get("iris_x", 0.5)
        self.iris_y = kw.get("iris_y", 0.5)
        self.x = kw.get("x", 0.0)
        self.y = kw.get("y", 0.0)
        self.head_yaw = kw.get("head_yaw", 0.0)
        self.head_pitch = kw.get("head_pitch", 0.0)
        self.distance_cm = kw.get("distance_cm", 0.0)
        self.left_ear = kw.get("left_ear", 0.3)
        self.right_ear = kw.get("right_ear", 0.3)
        self.blink_score = kw.get("blink_score", None)


class TestFormat:
    def test_header_matches_columns(self):
        assert format_header() == ",".join(signal_logger.COLUMNS)

    def test_row_field_count_matches_header(self):
        row = format_row(1.0, FakeResult())
        assert len(row.split(",")) == len(signal_logger.COLUMNS)

    def test_none_blink_is_empty_field(self):
        row = format_row(0.0, FakeResult(blink_score=None))
        # 最後の列(blink)が空
        assert row.split(",")[-1] == ""

    def test_values_present(self):
        row = format_row(2.5, FakeResult(iris_x=0.6, iris_y=0.4, x=960.0, y=540.0))
        parts = row.split(",")
        assert parts[0] == "2.500"
        assert parts[1].startswith("0.6")
        assert parts[2].startswith("0.4")
        assert parts[3] == "960.0"
        assert parts[4] == "540.0"


class TestSignalLogger:
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.csv")
            log = SignalLogger(path)
            assert log.start() is True
            log.log(100.0, FakeResult(iris_y=0.3, y=200.0))
            log.log(100.05, FakeResult(iris_y=0.7, y=800.0))
            log.close()

            with open(path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            assert lines[0] == format_header()
            assert len(lines) == 3  # header + 2 rows
            assert log.rows == 2

    def test_first_timestamp_is_zero_based(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.csv")
            log = SignalLogger(path)
            log.start()
            log.log(500.0, FakeResult())   # t0=500
            log.log(500.5, FakeResult())
            log.close()
            with open(path) as f:
                rows = [ln.strip() for ln in f if ln.strip()][1:]
            assert rows[0].split(",")[0] == "0.000"
            assert rows[1].split(",")[0] == "0.500"

    def test_none_result_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.csv")
            log = SignalLogger(path)
            log.start()
            log.log(1.0, None)
            log.close()
            assert log.rows == 0
