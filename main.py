"""GazeControl — 視線追跡カーソル制御システム エントリーポイント（精度改善版）"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

# 安全装置: 必ず最初に有効化
pyautogui.FAILSAFE = True

from src import config
from src.accessibility_snap import AccessibilitySnap
from src.calibration import CalibrationOverlay
from src.camera_stream import CameraStream
from src.cursor_controller import CursorController
from src.gaze_estimator import GazeEstimator, GazeResult
from src.fixation import FixationTracker
from src.gaze_pointer import GazePointer
from src.gestures import EyesClosedTrigger, is_eye_closed
from src.hotkeys import HotkeyController
from src.virtual_cursor import VirtualCursorOverlay


def get_screen_size() -> Tuple[int, int]:
    """スクリーン解像度を取得する"""
    try:
        from screeninfo import get_monitors
        monitor = get_monitors()[0]
        return monitor.width, monitor.height
    except Exception:
        return pyautogui.size()


class GazeControlApp:
    """視線追跡カーソル制御のメインアプリケーション

    GazeEstimator, CursorController, CalibrationOverlay を統合し、
    カメラからの映像を処理してOSカーソルを制御する。
    """

    def __init__(
        self,
        debug: bool = False,
        skip_calib: bool = False,
        blink_click: bool = False,
        sensitivity: float = config.DEFAULT_SENSITIVITY,
        virtual_cursor: bool = False,
        threaded_camera: bool = False,
        show_window: bool = False,
        precision_mode: bool = False,
        camera_index: int = config.CAMERA_INDEX,
        no_hotkeys: bool = False,
        cursor_smoothing: float = config.VIRTUAL_CURSOR_SMOOTHING,
        cursor_max_speed: float = config.VIRTUAL_CURSOR_MAX_SPEED,
        dwell_time: float = config.FIXATION_DWELL_TIME,
        hold_radius: float = config.FIXATION_RELEASE_RADIUS,
        gaze_range: float = config.GAZE_RANGE_MULT,
        distance_adapt: float = config.DISTANCE_ADAPT,
        calib_file: str = config.CALIBRATION_FILE,
        recalibrate: bool = False,
        snap_shape: bool = False,
        v_gain: float = config.VERTICAL_GAIN,
        down_boost: float = config.VERTICAL_DOWN_BOOST,
        down_smooth: float = config.VERTICAL_DOWN_SMOOTH,
        snap_radius: float = config.SNAP_MAGNET_RADIUS,
        sweep_calib: bool = False,
    ) -> None:
        self._debug = debug
        self._skip_calib = skip_calib
        self._sensitivity = sensitivity
        self._virtual_cursor = virtual_cursor
        self._threaded_camera = threaded_camera
        self._camera_index = camera_index
        self._no_hotkeys = no_hotkeys
        self._calib_file = calib_file
        self._recalibrate = recalibrate
        self._sweep_calib = sweep_calib

        # プレビューウィンドウ: 既定で非表示（窓を見ると視線が引っ張られ制御が乱れるため）。
        # --debug または --show-window で初期表示、実行中は 'p' キーでトグルできる。
        self._preview_visible = debug or show_window
        # プレビュー窓を初回表示時に画面中央上部（mac内蔵カメラ位置）へ配置するためのフラグ
        self._preview_positioned = False

        # スクリーンサイズ取得
        self._screen_w, self._screen_h = get_screen_size()
        print(f"スクリーンサイズ: {self._screen_w} x {self._screen_h}")

        # コンポーネント初期化（眉上げで精密モード自動切替は opt-in）
        self._estimator = GazeEstimator(
            self._screen_w, self._screen_h, enable_precision=precision_mode
        )
        self._estimator.sensitivity = sensitivity
        self._estimator.range_mult = gaze_range
        self._estimator.distance_adapt = distance_adapt
        self._estimator.v_gain = v_gain
        self._estimator.down_boost = down_boost
        self._estimator.down_smooth = down_smooth

        # カーソル制御:
        #   通常モード       → OSの実カーソルを動かす CursorController
        #   仮想カーソルモード → OSカーソルには触れず、gazeで仮想カーソルだけを動かす
        self._controller: Optional[CursorController] = None
        self._vcursor: Optional[VirtualCursorOverlay] = None
        if virtual_cursor:
            self._vcursor = VirtualCursorOverlay(
                self._screen_w, self._screen_h,
                smoothing=cursor_smoothing, max_speed=cursor_max_speed,
            )
        else:
            self._controller = CursorController(blink_click=blink_click)

        # 視線ポインター（座標保持）
        self._pointer = GazePointer()

        # 注視ベースのターゲット確定（履歴重心+ドウェル）— 仮想カーソルモードで使用
        self._fixation: Optional[FixationTracker] = (
            FixationTracker(dwell_time=dwell_time, release_radius=hold_radius)
            if virtual_cursor else None
        )

        # グローバルホットキー（ウィンドウ非表示でも終了/表示切替を受け付ける）
        self._hotkeys = HotkeyController(
            toggle_char=config.HOTKEY_TOGGLE_PREVIEW,
            quit_chars=(config.HOTKEY_QUIT,),
        )

        # 3秒間目を閉じる→リセット のジェスチャ
        self._eyes_closed = EyesClosedTrigger()

        # 形状スナップ＋磁石スナップ（近くのボタン/カードに吸着＋形変形）— 仮想カーソルモードのみ
        self._shape_snap: Optional[AccessibilitySnap] = None
        self._shape_query_t = 0.0
        self._snap_cache: Optional[Tuple[float, float, float, float]] = None
        self._snap_radius = snap_radius
        if snap_shape and virtual_cursor:
            if AccessibilitySnap.is_available():
                self._shape_snap = AccessibilitySnap()
            else:
                print("注意: 形状スナップには pyobjc が必要です（無効化）")

        # カメラ（生キャプチャ or スレッド化ラッパー）
        self._cap: Optional[object] = None

        # 最新の視線比率（キャリブレーション用コールバック）
        self._latest_gaze_ratio: Optional[Tuple[float, float]] = None

    def run(self) -> None:
        """アプリケーションを実行する"""
        # カメラ初期化（高解像度）
        try:
            cap = cv2.VideoCapture(self._camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        except Exception as e:
            print(f"エラー: カメラを開けませんでした — {e}")
            self._print_camera_help()
            sys.exit(1)

        if not cap.isOpened():
            print(f"エラー: カメラ(index={self._camera_index})を開けませんでした。")
            self._print_camera_help()
            sys.exit(1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # スレッド化キャプチャ: 取得を別スレッドに逃がしFPSを底上げ
        if self._threaded_camera:
            self._cap = CameraStream(cap).start()
            print(f"カメラ初期化完了 ({actual_w}x{actual_h}, スレッド取得)")
        else:
            self._cap = cap
            print(f"カメラ初期化完了 ({actual_w}x{actual_h})")

        # キャリブレーション: 保存済みがあれば読み込んで dot をスキップ
        loaded = False
        if not self._recalibrate and not self._skip_calib:
            if os.path.exists(self._calib_file) and self._estimator.load_calibration(self._calib_file):
                print(f"保存済みキャリブを読み込みました（dotスキップ）: {self._calib_file}")
                print("  やり直すには --recalibrate")
                loaded = True

        if not loaded and not self._skip_calib:
            print("キャリブレーションを開始します...")
            success = self._run_calibration()
            if not success:
                print("キャリブレーション失敗またはキャンセルされました。")
                print("--skip-calib オプションでスキップできます。")
                self._cleanup()
                sys.exit(1)
            print("キャリブレーション完了")
            if self._estimator.save_calibration(self._calib_file):
                print(f"キャリブを保存しました（次回から自動読込でdotスキップ）: {self._calib_file}")
        elif self._skip_calib:
            print("キャリブレーションをスキップしました（簡易マッピングモード）")

        # ポインター / 仮想カーソル起動
        self._pointer.start()
        if self._vcursor is not None:
            if self._vcursor.start():
                print("仮想カーソル（広範囲ブラーの円形オーバーレイ）を起動しました — OSの実マウスには干渉しません")
            else:
                print("注意: 仮想カーソルを起動できませんでした（PySide6/PyQt5 未導入かGUI不可の可能性）")
        else:
            print("視線ポインターを起動しました")

        # メインループ
        print("視線追跡を開始します（Q キーまたは ESC で終了）")
        self._main_loop()

    def _print_camera_help(self) -> None:
        """カメラを開けない時の macOS 向けトラブルシュート案内"""
        print("確認してください:")
        print("  1. macOS: システム設定 > プライバシーとセキュリティ > カメラ で、")
        print("     使用中のターミナルアプリ(Terminal/iTerm/VS Code等)を許可し、")
        print("     ターミナルを完全に終了してから開き直す")
        print("  2. 他アプリ(Zoom/Photo Booth/ブラウザ/iPhone連係カメラ等)がカメラを使っていないか")
        print("  3. 別のカメラ番号を試す: --camera-index 1 （0,1,2... と順に）")

    def _do_reset(self) -> None:
        """カーソル位置と頭部基準をリセットして中央付近へ戻す。"""
        self._estimator.reset_runtime()
        if self._fixation is not None:
            self._fixation.reset()
        self._eyes_closed.reset()
        print("リセット: 頭部基準とカーソル位置を再設定しました")

    def _apply_snap(self, fx: float, fy: float) -> Tuple[float, float]:
        """近傍のUI要素に吸着（磁石）＋形状変形する。

        間引いて近傍要素を問い合わせ（キャッシュ）、見つかればその中心へ吸着して
        カーソルをボタン/カードの形に変形する。無ければ (fx, fy) を返す。
        """
        if self._shape_snap is None or self._vcursor is None:
            return (fx, fy)

        now = time.time()
        if now - self._shape_query_t >= config.SNAP_SHAPE_QUERY_INTERVAL:
            self._shape_query_t = now
            try:
                self._snap_cache = self._shape_snap.nearest_element_frame(
                    fx, fy, self._snap_radius
                )
            except Exception:  # noqa: BLE001
                self._snap_cache = None

        frame = self._snap_cache
        if frame is not None:
            ecx, ecy, ew, eh = frame
            if (config.SNAP_SHAPE_MIN_SIZE <= ew <= config.SNAP_SHAPE_MAX_SIZE
                    and config.SNAP_SHAPE_MIN_SIZE <= eh <= config.SNAP_SHAPE_MAX_SIZE):
                self._vcursor.set_shape(ecx, ecy, ew, eh)
                return (ecx, ecy)  # 磁石: 要素中心へ吸着

        self._vcursor.clear_shape()
        return (fx, fy)

    def _run_calibration(self) -> bool:
        """キャリブレーションを実行（--sweep-calib で一周なぞり方式）"""
        overlay = CalibrationOverlay(
            screen_width=self._screen_w,
            screen_height=self._screen_h,
            gaze_callback=self._get_gaze_ratio_for_calibration,
        )

        if self._sweep_calib:
            print("一周なぞりキャリブ: 動くドットを目で追ってください")
            success = overlay.run_sweep()
        else:
            success = overlay.run()

        if success:
            calib_ok = self._estimator.compute_calibration(
                overlay.gaze_points, overlay.screen_points
            )
            if not calib_ok:
                return False

        return success

    def _get_gaze_ratio_for_calibration(self) -> Optional[Tuple[float, float]]:
        """キャリブレーション用: カメラフレームを取得して視線比率を返す"""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # カメラ映像を左右反転（鏡像補正）
        if config.CAMERA_FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        result = self._estimator.process_frame(frame)
        if result is None:
            return None

        # キャリブレーション前なので、生の比率を返す
        ratio_x = result.x / self._screen_w
        ratio_y = result.y / self._screen_h
        return (ratio_x, ratio_y)

    def _main_loop(self) -> None:
        """メインフレームループ"""
        frame_count = 0
        fps_start = time.perf_counter()
        last_seq = -1

        # ホットキー開始。--no-hotkeys または pynput不在ならウィンドウ+waitKeyにフォールバック
        hotkeys_active = (not self._no_hotkeys) and self._hotkeys.start()
        if hotkeys_active:
            print(
                f"ホットキー: '{config.HOTKEY_TOGGLE_PREVIEW}'=プレビュー表示切替 / "
                f"'r'=リセット / '{config.HOTKEY_QUIT}' または ESC=終了"
            )
        else:
            reason = "--no-hotkeys 指定" if self._no_hotkeys else "pynput未導入/利用不可"
            print(f"注意: グローバルホットキー無効（{reason}）。プレビューウィンドウのキー(q/ESC)で操作します")
            self._preview_visible = True

        if not self._preview_visible:
            print(f"プレビューウィンドウ非表示で実行中（'{config.HOTKEY_TOGGLE_PREVIEW}' キーで表示切替）")

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._threaded_camera:
                    # スレッド取得では起動直後に未取得(None)が来うる → 少し待って継続
                    time.sleep(0.005)
                    if hotkeys_active and self._hotkeys.should_quit:
                        break
                    continue
                print("カメラからフレームを取得できませんでした。")
                break

            # スレッド取得時は同一フレームの無駄な再処理を避ける
            if self._threaded_camera:
                seq = self._cap.seq
                if seq == last_seq:
                    time.sleep(0.003)
                    if hotkeys_active and self._hotkeys.should_quit:
                        break
                    continue
                last_seq = seq

            # カメラ映像を左右反転（鏡像補正）
            if config.CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # 視線推定
            result = self._estimator.process_frame(frame)

            if result is not None:
                # 3秒間目を閉じる → リセット
                closed = is_eye_closed(result.left_ear, result.right_ear, result.blink_score)
                if self._eyes_closed.update(closed, time.time()):
                    print("3秒閉眼を検出 → リセット")
                    self._do_reset()

                if self._vcursor is not None:
                    # 仮想カーソルモード: 注視で確定した点へ。近くにUI要素があれば吸着＋形変形。
                    fx, fy, _ = self._fixation.update(result.x, result.y, time.time())
                    cx, cy = self._apply_snap(fx, fy)
                    self._vcursor.update_position(cx, cy, visible=True)
                    self._pointer.update_position(cx, cy)
                else:
                    # 通常モード: OSの実カーソルを制御
                    clicked = self._controller.update(
                        target_x=result.x,
                        target_y=result.y,
                        left_ear=result.left_ear,
                        right_ear=result.right_ear,
                        confidence=result.confidence,
                        blink_score=result.blink_score,
                    )
                    self._pointer.update_position(
                        result.x, result.y,
                        dwell_progress=self._controller.dwell_progress,
                    )
                    if clicked:
                        print("クリック!")

            # プレビュー表示トグル（ホットキー）
            if hotkeys_active and self._hotkeys.poll_toggle():
                self._preview_visible = not self._preview_visible
                if not self._preview_visible:
                    cv2.destroyWindow(config.DEBUG_WINDOW_NAME)
                    self._preview_positioned = False  # 再表示時に再び中央上部へ
                print(f"プレビュー: {'表示' if self._preview_visible else '非表示'}")

            # リセット（ホットキー）
            if hotkeys_active and self._hotkeys.poll_reset():
                self._do_reset()

            # プレビュー描画（表示時のみ。窓を見ると視線が引っ張られるため既定は非表示）
            if self._preview_visible:
                self._show_preview(frame, result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break
                elif key == ord("r"):  # リセット
                    self._do_reset()

            # FPS計算
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                if self._debug:
                    print(f"FPS: {fps:.1f}")
                frame_count = 0
                fps_start = time.perf_counter()

            # ホットキー終了
            if hotkeys_active and self._hotkeys.should_quit:
                break

        self._cleanup()

    def _show_preview(self, frame: np.ndarray, result: Optional[GazeResult]) -> None:
        """ミニプレビューを表示する。--debug時は詳細オーバーレイ、通常は小さな素のフレーム。

        初回表示時に画面中央上部（mac内蔵カメラの位置）へ配置する。
        """
        if self._debug:
            self._draw_debug(frame, result)
            win_w = 640
        else:
            small = cv2.resize(frame, (480, 270)) if frame.shape[1] > 480 else frame
            cv2.imshow(config.DEBUG_WINDOW_NAME, small)
            win_w = 480

        # 初回のみ: 画面中央上部に移動（mac内蔵カメラ位置に合わせる）
        if not self._preview_positioned:
            x = max(0, (self._screen_w - win_w) // 2)
            try:
                cv2.moveWindow(config.DEBUG_WINDOW_NAME, x, 0)
            except cv2.error:
                pass
            self._preview_positioned = True

    def _draw_debug(self, frame: np.ndarray, result: Optional[GazeResult]) -> None:
        """デバッグ情報をフレームに描画して表示"""
        # デバッグ用に表示サイズを縮小
        if frame.shape[1] > 640:
            debug_frame = cv2.resize(frame, (640, 480))
        else:
            debug_frame = frame.copy()

        if result is not None:
            # 視線座標テキスト
            cv2.putText(
                debug_frame,
                f"Gaze: ({result.x:.0f}, {result.y:.0f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # EAR表示
            cv2.putText(
                debug_frame,
                f"EAR L:{result.left_ear:.2f} R:{result.right_ear:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # 信頼度
            cv2.putText(
                debug_frame,
                f"Conf: {result.confidence:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # 頭部姿勢表示
            cv2.putText(
                debug_frame,
                f"Head Yaw:{result.head_yaw:.2f} Pitch:{result.head_pitch:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 150, 255),
                2,
            )

            # 推定距離（30〜50cmが最適。緑=最適範囲, 黄=範囲外）
            in_range = 30.0 <= result.distance_cm <= 50.0
            cv2.putText(
                debug_frame,
                f"Dist:~{result.distance_cm:.0f}cm {'OK' if in_range else '(30-50cm)'}",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if in_range else (0, 200, 255),
                2,
            )

            # Dwell進捗バー
            progress = self._controller.dwell_progress if self._controller is not None else 0.0
            if progress > 0:
                bar_w = int(200 * progress)
                cv2.rectangle(debug_frame, (10, 140), (10 + bar_w, 155), (0, 255, 0), -1)
                cv2.rectangle(debug_frame, (10, 140), (210, 155), (255, 255, 255), 1)

            # 虹彩ランドマーク描画
            if result.landmarks:
                self._draw_iris_landmarks(debug_frame, result.landmarks)
        else:
            cv2.putText(
                debug_frame,
                "顔未検出",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # キャリブレーション状態
        calib_text = "Calibrated (poly)" if self._estimator.is_calibrated else "Uncalibrated"
        cv2.putText(
            debug_frame,
            calib_text,
            (10, debug_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(config.DEBUG_WINDOW_NAME, debug_frame)

    def _draw_iris_landmarks(self, frame: np.ndarray, landmarks: object) -> None:
        """虹彩ランドマークをフレームに描画"""
        h, w = frame.shape[:2]
        lm = landmarks.landmark

        # 左目虹彩
        for idx in config.LEFT_IRIS_POINTS:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            color = (0, 255, 0) if idx == config.LEFT_IRIS_CENTER else (0, 200, 0)
            radius = 3 if idx == config.LEFT_IRIS_CENTER else 2
            cv2.circle(frame, (x, y), radius, color, -1)

        # 右目虹彩
        for idx in config.RIGHT_IRIS_POINTS:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            color = (0, 255, 0) if idx == config.RIGHT_IRIS_CENTER else (0, 200, 0)
            radius = 3 if idx == config.RIGHT_IRIS_CENTER else 2
            cv2.circle(frame, (x, y), radius, color, -1)

    def _cleanup(self) -> None:
        """リソース解放"""
        self._hotkeys.stop()
        self._pointer.stop()
        if self._vcursor is not None:
            self._vcursor.stop()
        if self._cap is not None:
            self._cap.release()
        self._estimator.release()
        cv2.destroyAllWindows()
        print("終了しました。")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="GazeControl — 視線追跡カーソル制御システム（精度改善版）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="プレビューウィンドウに詳細オーバーレイを表示する（初期表示ON、'p'でトグル）",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="起動時からプレビューウィンドウを表示する（既定は非表示、'p'でトグル）",
    )
    parser.add_argument(
        "--skip-calib",
        action="store_true",
        help="キャリブレーションをスキップする",
    )
    parser.add_argument(
        "--blink-click",
        action="store_true",
        help="瞬きクリックモードを有効にする（デフォルトはDwell Click）",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=config.DEFAULT_SENSITIVITY,
        help=f"視線感度 (1.0〜10.0, デフォルト: {config.DEFAULT_SENSITIVITY})",
    )
    parser.add_argument(
        "--virtual-cursor",
        action="store_true",
        help="OSカーソルを動かさず、gaze駆動の仮想カーソル（広範囲ブラーの円形オーバーレイ）を表示する",
    )
    parser.add_argument(
        "--threaded-camera",
        action="store_true",
        help="カメラ取得を別スレッド化して実効FPSを底上げする",
    )
    parser.add_argument(
        "--precision-mode",
        action="store_true",
        help="眉上げ（blendshape優先）で精密モードを自動切替する",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"カメラデバイス番号 (デフォルト: {config.CAMERA_INDEX}, 開けない場合は 1, 2 を試す)",
    )
    parser.add_argument(
        "--no-hotkeys",
        action="store_true",
        help="pynputグローバルホットキーを無効化（プレビュー窓のq/ESCで操作）。macでpynputがクラッシュする場合の回避用",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=config.VIRTUAL_CURSOR_SMOOTHING,
        help=f"仮想カーソルの追従応答性 (0<r<=1, 小さいほど遅くぬるっと, デフォルト: {config.VIRTUAL_CURSOR_SMOOTHING})",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=config.VIRTUAL_CURSOR_MAX_SPEED,
        help=f"仮想カーソルの最大速度 px/秒 (小さいほど遅く見失いにくい, 0で無制限, デフォルト: {config.VIRTUAL_CURSOR_MAX_SPEED})",
    )
    parser.add_argument(
        "--dwell-time",
        type=float,
        default=config.FIXATION_DWELL_TIME,
        help=f"注視確定までの滞留時間 秒 (この時間とどまると移動。0.2〜0.5推奨, デフォルト: {config.FIXATION_DWELL_TIME})",
    )
    parser.add_argument(
        "--hold-radius",
        type=float,
        default=config.FIXATION_RELEASE_RADIUS,
        help=f"ロック保持半径 px — 大きいほどピタッと固定（近くのブレで動かない）, デフォルト: {config.FIXATION_RELEASE_RADIUS}",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=config.GAZE_RANGE_MULT,
        help=f"可動域（ゲイン）倍率 — 大きいほど視線で広く届く (デフォルト: {config.GAZE_RANGE_MULT})",
    )
    parser.add_argument(
        "--distance-adapt",
        type=float,
        default=config.DISTANCE_ADAPT,
        help=f"距離適応の強さ 0〜1 (近い/遠いでゲイン自動調整。0で無効, デフォルト: {config.DISTANCE_ADAPT})",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="保存済みキャリブを無視して再キャリブする",
    )
    parser.add_argument(
        "--sweep-calib",
        action="store_true",
        help="一周なぞりキャリブ（ドットを画面の縁に沿って追う方式）。隅・下端をしっかり取得",
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default=config.CALIBRATION_FILE,
        help="キャリブ保存ファイルのパス",
    )
    parser.add_argument(
        "--snap-shape",
        action="store_true",
        help="近くのボタン/カードに吸着＋その形にカーソルを変形（磁石スナップ, macOS Accessibility, 実験的）",
    )
    parser.add_argument(
        "--snap-radius",
        type=float,
        default=config.SNAP_MAGNET_RADIUS,
        help=f"磁石スナップの吸着半径 px (デフォルト: {config.SNAP_MAGNET_RADIUS})",
    )
    parser.add_argument(
        "--v-gain",
        type=float,
        default=config.VERTICAL_GAIN,
        help=f"縦ゲイン倍率（>1で上下に広く届く, デフォルト: {config.VERTICAL_GAIN}）",
    )
    parser.add_argument(
        "--down-boost",
        type=float,
        default=config.VERTICAL_DOWN_BOOST,
        help=f"下を見るほど縦可動量を増やす量 (0で無効, 例 0.5, デフォルト: {config.VERTICAL_DOWN_BOOST})",
    )
    parser.add_argument(
        "--down-smooth",
        type=float,
        default=config.VERTICAL_DOWN_SMOOTH,
        help=f"下を見るほど縦を強く平滑化（下端の分散・ブレ抑制, 0で無効, 例 0.6, デフォルト: {config.VERTICAL_DOWN_SMOOTH})",
    )
    return parser.parse_args()


def main() -> None:
    """エントリーポイント"""
    args = parse_args()

    app = GazeControlApp(
        debug=args.debug,
        skip_calib=args.skip_calib,
        blink_click=args.blink_click,
        sensitivity=args.sensitivity,
        virtual_cursor=args.virtual_cursor,
        threaded_camera=args.threaded_camera,
        show_window=args.show_window,
        precision_mode=args.precision_mode,
        camera_index=args.camera_index,
        no_hotkeys=args.no_hotkeys,
        cursor_smoothing=args.smoothing,
        cursor_max_speed=args.max_speed,
        dwell_time=args.dwell_time,
        hold_radius=args.hold_radius,
        gaze_range=args.range,
        distance_adapt=args.distance_adapt,
        recalibrate=args.recalibrate,
        calib_file=args.calib_file,
        snap_shape=args.snap_shape,
        v_gain=args.v_gain,
        down_boost=args.down_boost,
        down_smooth=args.down_smooth,
        snap_radius=args.snap_radius,
        sweep_calib=args.sweep_calib,
    )

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n中断されました。")


if __name__ == "__main__":
    main()
