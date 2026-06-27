"""仮想カーソル — gaze駆動・OSカーソルから独立した表示専用カーソル

OSの実マウス（ユーザーが手で動かすカーソル）には一切干渉せず、視線で動く
「広範囲ブラーの円形カーソル」を、透明・クリックスルー・常時最前面の
オーバーレイとして描画する。

設計方針:
  * スプライト生成 (`make_cursor_sprite`) と追従スムージング (`exp_smooth`) は
    Qtに依存しない純粋関数として実装し、ヘッドレスでユニットテスト可能にする。
  * GUI描画は PySide6 / PyQt5 を**子プロセス側で遅延importして**動かす。
    こうすることで OpenCV のメインループ（メインスレッドで cv2.imshow を回す）と
    Qt のイベントループが衝突しない。プロセス間は共有メモリ (multiprocessing.Array)
    で最新座標だけを受け渡す。
  * Qt が未インストール、もしくはディスプレイが無い環境でも、親プロセスを
    巻き込まずグレースフルに無効化される。
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# 純粋ロジック（Qt非依存・テスト対象）
# ---------------------------------------------------------------------------
def make_cursor_sprite(
    diameter: int,
    color_rgb: Tuple[int, int, int],
    core_ratio: float = config.VIRTUAL_CURSOR_CORE_RATIO,
    blur_ratio: float = config.VIRTUAL_CURSOR_BLUR_RATIO,
    max_alpha: float = config.VIRTUAL_CURSOR_MAX_OPACITY,
) -> np.ndarray:
    """広範囲ブラーの円形カーソルスプライトを RGBA で生成する。

    くっきりした中心円（core）に、広範囲へ滑らかに減衰するブラー halo を重ね、
    アルファチャンネルで柔らかいグローを表現する。

    Args:
        diameter: スプライトの一辺 (px)。正方形 RGBA を返す。
        color_rgb: カーソル色 (R, G, B) 各 0〜255。
        core_ratio: 中心円の半径比（スプライト半径に対する割合）。
        blur_ratio: ブラーの広がり（大きいほど halo が広範囲）。
        max_alpha: 中心の最大不透明度 (0.0〜1.0)。

    Returns:
        (diameter, diameter, 4) uint8 の RGBA 配列。チャンネル順は R, G, B, A。
    """
    import cv2  # 局所import: モジュール読み込み時の依存を最小化

    size = max(4, int(diameter))
    center = (size - 1) / 2.0
    radius = size / 2.0

    core_r = max(1, int(round(radius * max(0.01, core_ratio))))

    # 中心の塗りつぶし円マスク
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, (int(round(center)), int(round(center))), core_r, 1.0, -1, lineType=cv2.LINE_AA)

    # 広範囲ガウシアンブラーで halo を生成
    ksize = int(size * max(0.05, blur_ratio))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    sigma = max(1.0, size * blur_ratio / 3.0)
    halo = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    peak = float(halo.max())
    if peak > 0:
        halo = halo / peak

    # くっきりした中心（軽くアンチエイリアスした円）と halo を合成し、
    # 「明瞭な円 + 広範囲のグロー」を両立させる
    core = np.zeros((size, size), dtype=np.float32)
    cv2.circle(core, (int(round(center)), int(round(center))), core_r, 1.0, -1, lineType=cv2.LINE_AA)
    core = cv2.GaussianBlur(core, (5, 5), sigmaX=1.5, sigmaY=1.5)

    alpha = np.maximum(halo, core)
    alpha = np.clip(alpha * float(max_alpha), 0.0, 1.0)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    r, g, b = (int(c) for c in color_rgb)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
    return rgba


def exp_smooth(
    current: float,
    target: float,
    responsiveness: float,
    dt: float,
    ref_dt: float = 1.0 / 60.0,
) -> float:
    """フレームレート非依存の指数スムージングで current を target へ近づける。

    `responsiveness` は ref_dt 秒あたりに「残り距離のうち詰める割合」(0〜1)。
    実 dt が変動しても見かけの追従速度が一定になるよう補正する。
    オーバーシュートしない（単調に target へ近づく）。

    Args:
        current: 現在値。
        target: 目標値。
        responsiveness: 応答性 (0〜1]。1で ref_dt ごとに即到達。
        dt: 今回の経過時間 (秒)。0以下なら current をそのまま返す。
        ref_dt: 基準時間 (秒)。

    Returns:
        更新後の値。
    """
    if dt <= 0:
        return current
    r = min(1.0, max(0.0, responsiveness))
    if r >= 1.0:
        return target
    frac = 1.0 - (1.0 - r) ** (dt / ref_dt)
    return current + frac * (target - current)


# ---------------------------------------------------------------------------
# オーバーレイ（Qt — 子プロセス側で実行）
# ---------------------------------------------------------------------------
# 共有メモリ Array のインデックス
_IDX_X = 0          # 目標 x (px)
_IDX_Y = 1          # 目標 y (px)
_IDX_VISIBLE = 2    # 1.0=表示, 0.0=非表示, 負値=シャットダウン要求
_IDX_DWELL = 3      # dwell進捗 0.0〜1.0（将来用・現状は表示のみ）
_SHARED_LEN = 4

_SHUTDOWN = -1.0


def _import_qt():
    """PySide6 を優先し、無ければ PyQt5 を試す。両方無ければ ImportError。"""
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        return QtCore, QtGui, QtWidgets, "exec"
    except ImportError:
        pass
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    return QtCore, QtGui, QtWidgets, "exec_"


def _run_overlay_process(shared, params: dict) -> None:
    """子プロセスのエントリーポイント。オーバーレイは best-effort で、
    例外が起きても親を巻き込まずログだけ出して終了する。"""
    try:
        _qt_overlay_main(shared, params)
    except Exception as e:  # noqa: BLE001 — 子プロセスの保険
        print(f"[virtual_cursor] オーバーレイを起動できませんでした: {e}", file=sys.stderr)


def _qt_overlay_main(shared, params: dict) -> None:
    QtCore, QtGui, QtWidgets, exec_name = _import_qt()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    sprite = make_cursor_sprite(
        diameter=params["diameter"],
        color_rgb=tuple(params["color"]),
        core_ratio=params["core_ratio"],
        blur_ratio=params["blur_ratio"],
        max_alpha=params["max_opacity"],
    )
    h, w = sprite.shape[:2]
    buf = np.ascontiguousarray(sprite)  # QImage がバッファを参照するため保持し続ける
    qimg = QtGui.QImage(
        buf.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888
    ).copy()
    pixmap = QtGui.QPixmap.fromImage(qimg)

    class _Overlay(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.WindowStaysOnTopHint
                | QtCore.Qt.Tool
                | QtCore.Qt.WindowTransparentForInput
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            self.resize(w, h)
            self._pixmap = pixmap
            self._opacity = 0.0

        def set_opacity(self, value: float) -> None:
            self._opacity = value

        def paintEvent(self, _event) -> None:  # noqa: N802 — Qt命名規約
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.setOpacity(max(0.0, min(1.0, self._opacity)))
            painter.drawPixmap(0, 0, self._pixmap)
            painter.end()

    overlay = _Overlay()
    overlay.show()

    half_w = w / 2.0
    half_h = h / 2.0
    tick_dt = float(params["tick_dt"])
    responsiveness = float(params["smoothing"])
    max_opacity = float(params["max_opacity"])

    state = {"x": float(shared[_IDX_X]), "y": float(shared[_IDX_Y])}

    def tick() -> None:
        with shared.get_lock():
            tx = float(shared[_IDX_X])
            ty = float(shared[_IDX_Y])
            vis = float(shared[_IDX_VISIBLE])

        if vis == _SHUTDOWN:
            app.quit()
            return

        state["x"] = exp_smooth(state["x"], tx, responsiveness, tick_dt)
        state["y"] = exp_smooth(state["y"], ty, responsiveness, tick_dt)
        overlay.move(int(round(state["x"] - half_w)), int(round(state["y"] - half_h)))

        target_op = max_opacity if vis >= 0.5 else 0.0
        overlay.set_opacity(overlay._opacity + 0.25 * (target_op - overlay._opacity))
        overlay.update()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(max(1, int(tick_dt * 1000)))

    getattr(app, exec_name)()


class VirtualCursorOverlay:
    """gaze駆動の仮想カーソル。OSの実マウスには干渉しない。

    `GazePointer` と同じ `start()/update_position()/stop()` インターフェースを持つ
    ドロップイン。内部では Qt オーバーレイを別プロセスで動かし、最新座標を
    共有メモリ経由で渡す。Qt 不在・GUI不能環境では自動的に無効化される。
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        diameter: int = config.VIRTUAL_CURSOR_DIAMETER,
        color: Tuple[int, int, int] = config.VIRTUAL_CURSOR_COLOR,
        core_ratio: float = config.VIRTUAL_CURSOR_CORE_RATIO,
        blur_ratio: float = config.VIRTUAL_CURSOR_BLUR_RATIO,
        max_opacity: float = config.VIRTUAL_CURSOR_MAX_OPACITY,
        smoothing: float = config.VIRTUAL_CURSOR_SMOOTHING,
        tick_dt: float = config.VIRTUAL_CURSOR_TICK_DT,
    ) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._params = {
            "diameter": int(diameter),
            "color": tuple(int(c) for c in color),
            "core_ratio": float(core_ratio),
            "blur_ratio": float(blur_ratio),
            "max_opacity": float(max_opacity),
            "smoothing": float(smoothing),
            "tick_dt": float(tick_dt),
        }
        self._shared = None
        self._proc = None
        self._active = False

    @property
    def is_active(self) -> bool:
        """オーバーレイプロセスが起動しているか"""
        return self._active and self._proc is not None and self._proc.is_alive()

    def start(self) -> bool:
        """オーバーレイプロセスを起動する。成功で True。失敗してもraiseしない。"""
        import multiprocessing as mp

        try:
            self._shared = mp.Array("d", [
                self._screen_width / 2.0,
                self._screen_height / 2.0,
                0.0,
                0.0,
            ])
            self._proc = mp.Process(
                target=_run_overlay_process,
                args=(self._shared, self._params),
                daemon=True,
            )
            self._proc.start()
            self._active = True
            return True
        except Exception as e:  # noqa: BLE001 — オーバーレイは必須ではない
            print(f"[virtual_cursor] 仮想カーソルを開始できませんでした: {e}", file=sys.stderr)
            self._active = False
            self._shared = None
            self._proc = None
            return False

    def update_position(self, x: float, y: float, visible: bool = True,
                        dwell_progress: float = 0.0) -> None:
        """仮想カーソルの目標位置を更新する（実マウスは動かさない）。"""
        if self._shared is None:
            return
        with self._shared.get_lock():
            self._shared[_IDX_X] = float(x)
            self._shared[_IDX_Y] = float(y)
            self._shared[_IDX_VISIBLE] = 1.0 if visible else 0.0
            self._shared[_IDX_DWELL] = float(dwell_progress)

    def hide(self) -> None:
        """一時的に非表示にする（フェードアウト）。"""
        if self._shared is None:
            return
        with self._shared.get_lock():
            self._shared[_IDX_VISIBLE] = 0.0

    def stop(self) -> None:
        """オーバーレイプロセスを停止する。"""
        if self._proc is None:
            self._active = False
            return
        try:
            if self._shared is not None:
                with self._shared.get_lock():
                    self._shared[_IDX_VISIBLE] = _SHUTDOWN
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._active = False
            self._proc = None
            self._shared = None
