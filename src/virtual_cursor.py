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


def make_rect_sprite(
    w: int,
    h: int,
    color_rgb: Tuple[int, int, int],
    pad: int = config.SNAP_SHAPE_PAD,
    corner_ratio: float = config.SNAP_SHAPE_CORNER_RATIO,
    blur_ratio: float = config.SNAP_SHAPE_BLUR_RATIO,
    max_alpha: float = config.SNAP_SHAPE_MAX_OPACITY,
) -> np.ndarray:
    """ボタン/カードの形に合わせた角丸グローのスプライトを RGBA で生成する。

    内側 (w, h) の角丸矩形に外側 pad ぶんのブラーグローを付ける。

    Returns:
        (h + 2*pad, w + 2*pad, 4) uint8 の RGBA。チャンネル順は R, G, B, A。
    """
    import cv2

    w = max(2, int(w))
    h = max(2, int(h))
    pad = max(2, int(pad))
    canvas_w = w + 2 * pad
    canvas_h = h + 2 * pad

    r = int(max(1, round(min(w, h) * max(0.0, corner_ratio))))
    r = min(r, min(w, h) // 2) if min(w, h) >= 2 else 1
    r = max(1, r)

    x0, y0 = pad, pad
    x1, y1 = pad + w - 1, pad + h - 1

    mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    cv2.rectangle(mask, (x0 + r, y0), (x1 - r, y1), 1.0, -1)
    cv2.rectangle(mask, (x0, y0 + r), (x1, y1 - r), 1.0, -1)
    for (cx, cy) in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(mask, (cx, cy), r, 1.0, -1, lineType=cv2.LINE_AA)

    ksize = int(pad * 2 * max(0.1, blur_ratio))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    sigma = max(1.0, pad * blur_ratio)
    glow = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    peak = float(glow.max())
    if peak > 0:
        glow = glow / peak

    alpha = np.clip(np.maximum(mask, glow) * float(max_alpha), 0.0, 1.0)

    rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    cr, cg, cb = (int(c) for c in color_rgb)
    rgba[..., 0] = cr
    rgba[..., 1] = cg
    rgba[..., 2] = cb
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


def advance_cursor(
    cx: float, cy: float,
    tx: float, ty: float,
    responsiveness: float, dt: float,
    max_step: float,
    screen_w: float, screen_h: float,
    half_w: float, half_h: float,
) -> Tuple[float, float]:
    """1ティック分カーソルを進める純粋関数（テスト対象）。

    指数スムージングで目標へ寄せ、`max_step`(>0)で1ティックの移動量を上限制限し、
    スプライト全体が画面内に残るようクランプする。
    """
    nx = exp_smooth(cx, tx, responsiveness, dt)
    ny = exp_smooth(cy, ty, responsiveness, dt)

    # 速度キャップ（2D距離で制限）
    if max_step > 0.0:
        dx = nx - cx
        dy = ny - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_step and dist > 0.0:
            scale = max_step / dist
            nx = cx + dx * scale
            ny = cy + dy * scale

    # カーソル全体を画面内に保つ
    if screen_w > 2 * half_w:
        nx = min(max(nx, half_w), screen_w - half_w)
    if screen_h > 2 * half_h:
        ny = min(max(ny, half_h), screen_h - half_h)

    return (nx, ny)


def transit_scale(step_dist: float, ref_step: float, dip: float) -> float:
    """移動量に応じたスケール係数を返す（移動中は縮み、静止で膨らむ）。

    静止(step=0)で 1.0、ref_step 以上の速さで (1 - dip)。点と点の距離を
    視覚的に目立たなくするための「サイズ・イージング」。
    """
    if ref_step <= 0.0 or dip <= 0.0:
        return 1.0
    m = min(1.0, step_dist / ref_step)
    return 1.0 - dip * m


# ---------------------------------------------------------------------------
# オーバーレイ（Qt — 子プロセス側で実行）
# ---------------------------------------------------------------------------
# 共有メモリ Array のインデックス
_IDX_X = 0          # 目標 x (px)
_IDX_Y = 1          # 目標 y (px)
_IDX_VISIBLE = 2    # 1.0=表示, 0.0=非表示, 負値=シャットダウン要求
_IDX_DWELL = 3      # dwell進捗 0.0〜1.0（将来用・現状は表示のみ）
_IDX_EX = 4         # スナップ要素の中心 x（0=要素なし）
_IDX_EY = 5         # スナップ要素の中心 y
_IDX_EW = 6         # スナップ要素の幅（0=要素なし＝円のまま）
_IDX_EH = 7         # スナップ要素の高さ
_SHARED_LEN = 8

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

    def to_pixmap(rgba):
        rgba = np.ascontiguousarray(rgba)
        ih, iw = rgba.shape[:2]
        img = QtGui.QImage(
            rgba.data, iw, ih, 4 * iw, QtGui.QImage.Format_RGBA8888
        ).copy()
        return QtGui.QPixmap.fromImage(img)

    pixmap = to_pixmap(sprite)

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
            self._scale = 1.0

        def set_opacity(self, value: float) -> None:
            self._opacity = value

        def set_scale(self, value: float) -> None:
            self._scale = value

        def paintEvent(self, _event) -> None:  # noqa: N802 — Qt命名規約
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.setOpacity(max(0.0, min(1.0, self._opacity)))
            s = self._scale
            if s >= 0.999:
                painter.drawPixmap(0, 0, self._pixmap)
            else:
                sw = w * s
                sh = h * s
                target_rect = QtCore.QRectF((w - sw) / 2.0, (h - sh) / 2.0, sw, sh)
                src_rect = QtCore.QRectF(0.0, 0.0, float(w), float(h))
                painter.drawPixmap(target_rect, self._pixmap, src_rect)
            painter.end()

    class _RectHighlight(QtWidgets.QWidget):
        """スナップ要素を覆う角丸グロー（ボタン/カードの形のハイライト）。"""

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
            self._pixmap = None
            self._opacity = 0.0

        def set_pixmap(self, pix) -> None:
            self._pixmap = pix
            self.resize(pix.width(), pix.height())

        def set_opacity(self, value: float) -> None:
            self._opacity = value

        def paintEvent(self, _event) -> None:  # noqa: N802
            if self._pixmap is None:
                return
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.setOpacity(max(0.0, min(1.0, self._opacity)))
            painter.drawPixmap(0, 0, self._pixmap)
            painter.end()

    overlay = _Overlay()
    overlay.show()

    rect_hl = _RectHighlight()
    rect_hl.hide()

    half_w = w / 2.0
    half_h = h / 2.0
    tick_dt = float(params["tick_dt"])
    responsiveness = float(params["smoothing"])
    max_opacity = float(params["max_opacity"])
    max_speed = float(params.get("max_speed", 0.0))
    screen_w = float(params.get("screen_w", 0) or 0)
    screen_h = float(params.get("screen_h", 0) or 0)
    max_step = max_speed * tick_dt if max_speed > 0 else 0.0
    scale_dip = float(params.get("scale_dip", 0.0))
    scale_resp = float(params.get("scale_resp", 0.3))
    # スケール基準: 速度キャップがあればその1ティック上限、無ければ概算
    scale_ref = max_step if max_step > 0 else max(2.0, (screen_w or 1920.0) * tick_dt * 6.0)

    color = tuple(params["color"])
    state = {
        "x": float(shared[_IDX_X]), "y": float(shared[_IDX_Y]),
        "scale": 1.0, "snap": 0.0,
        "rect_size": (0, 0), "rect_pix": None,
        "rx": float(shared[_IDX_X]), "ry": float(shared[_IDX_Y]),
    }

    def tick() -> None:
        with shared.get_lock():
            tx = float(shared[_IDX_X])
            ty = float(shared[_IDX_Y])
            vis = float(shared[_IDX_VISIBLE])
            ex = float(shared[_IDX_EX])
            ey = float(shared[_IDX_EY])
            ew = float(shared[_IDX_EW])
            eh = float(shared[_IDX_EH])

        if vis == _SHUTDOWN:
            app.quit()
            return

        prev_x, prev_y = state["x"], state["y"]
        nx, ny = advance_cursor(
            prev_x, prev_y, tx, ty,
            responsiveness, tick_dt, max_step,
            screen_w, screen_h, half_w, half_h,
        )
        state["x"] = nx
        state["y"] = ny
        overlay.move(int(round(nx - half_w)), int(round(ny - half_h)))

        # サイズ・イージング: 移動量に応じて縮み、止まると膨らむ
        step_dist = ((nx - prev_x) ** 2 + (ny - prev_y) ** 2) ** 0.5
        target_scale = transit_scale(step_dist, scale_ref, scale_dip)
        state["scale"] = exp_smooth(state["scale"], target_scale, scale_resp, tick_dt)
        overlay.set_scale(state["scale"])

        # --- 形状スナップ（ボタン/カードの矩形へ変形）---
        snapped = ew >= 1.0 and eh >= 1.0
        state["snap"] = exp_smooth(state["snap"], 1.0 if snapped else 0.0, 0.25, tick_dt)
        snap_amt = state["snap"]
        if snapped:
            size = (int(ew), int(eh))
            if size != state["rect_size"]:
                state["rect_pix"] = to_pixmap(make_rect_sprite(size[0], size[1], color))
                state["rect_size"] = size
                rect_hl.set_pixmap(state["rect_pix"])
            state["rx"] = exp_smooth(state["rx"], ex, 0.5, tick_dt)
            state["ry"] = exp_smooth(state["ry"], ey, 0.5, tick_dt)
            pw, ph = rect_hl.width(), rect_hl.height()
            rect_hl.move(int(round(state["rx"] - pw / 2.0)), int(round(state["ry"] - ph / 2.0)))

        base_visible = 1.0 if vis >= 0.5 else 0.0
        rect_op = config.SNAP_SHAPE_MAX_OPACITY * snap_amt * base_visible
        if state["rect_pix"] is not None:
            rect_hl.set_opacity(rect_op)
            if rect_op > 0.01:
                if not rect_hl.isVisible():
                    rect_hl.show()
                rect_hl.update()
            elif rect_hl.isVisible():
                rect_hl.hide()

        # 円の不透明度（スナップ時は薄く）
        target_op = max_opacity * base_visible * (1.0 - 0.6 * snap_amt)
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
        max_speed: float = config.VIRTUAL_CURSOR_MAX_SPEED,
        scale_dip: float = config.VIRTUAL_CURSOR_SCALE_DIP,
        scale_resp: float = config.VIRTUAL_CURSOR_SCALE_RESP,
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
            "max_speed": float(max_speed),
            "screen_w": int(screen_width),
            "screen_h": int(screen_height),
            "scale_dip": float(scale_dip),
            "scale_resp": float(scale_resp),
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
            init = [0.0] * _SHARED_LEN
            init[_IDX_X] = self._screen_width / 2.0
            init[_IDX_Y] = self._screen_height / 2.0
            self._shared = mp.Array("d", init)
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

    def set_shape(self, ex: float, ey: float, ew: float, eh: float) -> None:
        """スナップ先のUI要素矩形を設定する。ew/eh<=0 で解除（円に戻る）。"""
        if self._shared is None:
            return
        with self._shared.get_lock():
            self._shared[_IDX_EX] = float(ex)
            self._shared[_IDX_EY] = float(ey)
            self._shared[_IDX_EW] = float(ew)
            self._shared[_IDX_EH] = float(eh)

    def clear_shape(self) -> None:
        """形状スナップを解除する。"""
        self.set_shape(0.0, 0.0, 0.0, 0.0)

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
