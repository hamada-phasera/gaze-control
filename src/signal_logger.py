"""視線信号のCSVロガー — 遠隔デバッグ用。

カメラ映像（顔の画像）は一切保存せず、推定済みの数値だけをCSVに残す。
これにより「Yが動かない」「ズレる」等の症状を、生信号の分散・相関から
オフラインで切り分けできる（tools/analyze_signal_log.py で解析）。

行フォーマットは純粋関数 `format_row` に分離してテスト可能にしてある。
"""

from __future__ import annotations

from typing import List, Optional

# CSVの列順（解析スクリプトと共有）
COLUMNS: List[str] = [
    "t",            # 記録開始からの経過秒
    "iris_x",       # キャリブ前の視線比率X（0.5中心）
    "iris_y",       # キャリブ前の視線比率Y（0.5中心）
    "x",            # 最終出力 画面X (px)
    "y",            # 最終出力 画面Y (px)
    "head_yaw",     # 頭ヨー(度)
    "head_pitch",   # 頭ピッチ(度)
    "distance_cm",  # 推定距離(cm)
    "left_ear",
    "right_ear",
    "blink",        # blendshape瞬きスコア（無ければ空）
]


def _fmt(v: Optional[float], nd: int = 5) -> str:
    """数値を固定小数で。None は空欄。"""
    if v is None:
        return ""
    return f"{float(v):.{nd}f}"


def format_header() -> str:
    return ",".join(COLUMNS)


def format_row(t: float, result: object) -> str:
    """GazeResult（または同等の属性を持つオブジェクト）から1行を組み立てる。"""
    g = lambda name, d=0.0: getattr(result, name, d)  # noqa: E731
    fields = [
        _fmt(t, 3),
        _fmt(g("iris_x", 0.5)),
        _fmt(g("iris_y", 0.5)),
        _fmt(g("x"), 1),
        _fmt(g("y"), 1),
        _fmt(g("head_yaw"), 2),
        _fmt(g("head_pitch"), 2),
        _fmt(g("distance_cm"), 1),
        _fmt(g("left_ear")),
        _fmt(g("right_ear")),
        _fmt(g("blink_score", None)),
    ]
    return ",".join(fields)


class SignalLogger:
    """視線信号を1フレーム1行でCSVに追記するロガー。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None
        self._t0: Optional[float] = None
        self._rows = 0

    def start(self) -> bool:
        try:
            self._file = open(self._path, "w", buffering=1)  # 行バッファ
            self._file.write(format_header() + "\n")
            return True
        except Exception:  # noqa: BLE001
            self._file = None
            return False

    def log(self, t: float, result: object) -> None:
        """1フレーム記録する。最初の呼び出しの t を 0 基準にする。"""
        if self._file is None or result is None:
            return
        if self._t0 is None:
            self._t0 = t
        try:
            self._file.write(format_row(t - self._t0, result) + "\n")
            self._rows += 1
        except Exception:  # noqa: BLE001
            pass

    @property
    def rows(self) -> int:
        return self._rows

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
            self._file = None
