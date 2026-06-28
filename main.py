"""GazeControl — 視線追跡カーソル制御システム エントリーポイント（精度改善版）"""

import argparse
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

# 安全装置: 必ず最初に有効化
pyautogui.FAILSAFE = True

from src import config
from src.calibration import CalibrationOverlay
from src.camera_stream import CameraStream
from src.cursor_controller import CursorController
from src.gaze_estimator import GazeEstimator, GazeResult
from src.gaze_pointer import GazePointer
from src.hotkeys import HotkeyController
from src.smoothing import MotionStabilizer
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
    ) -> None:
        self._debug = debug
        self._skip_calib = skip_calib
        self._sensitivity = sensitivity
        self._virtual_cursor = virtual_cursor
        self._threaded_camera = threaded_camera
        self._camera_index = camera_index
        self._no_hotkeys = no_hotkeys

        # プレビューウィンドウ: 既定で非表示（窓を見ると視線が引っ張られ制御が乱れるため）。
        # --debug または --show-window で初期表示、実行中は 'p' キーでトグルできる。
        self._preview_visible = debug or show_window

        # スクリーンサイズ取得
        self._screen_w, self._screen_h = get_screen_size()
        print(f"スクリーンサイズ: {self._screen_w} x {self._screen_h}")

        # コンポーネント初期化（眉上げで精密モード自動切替は opt-in）
        self._estimator = GazeEstimator(
            self._screen_w, self._screen_h, enable_precision=precision_mode
        )
        self._estimator.sensitivity = sensitivity

        # カーソル制御:
        #   通常モード       → OSの実カーソルを動かす CursorController
        #   仮想カーソルモード → OSカーソルには触れず、gazeで仮想カーソルだけを動かす
        self._controller: Optional[CursorController] = None
        self._vcursor: Optional[VirtualCursorOverlay] = None
        if virtual_cursor:
            self._vcursor = VirtualCursorOverlay(
                self._screen_w, self._screen_h, smoothing=cursor_smoothing
            )
        else:
            self._controller = CursorController(blink_click=blink_click)

        # 視線ポインター（座標保持）
        self._pointer = GazePointer()

        # 動き安定化（「ぬるっと」感）— 仮想カーソルモードで使用
        self._stabilizer: Optional[MotionStabilizer] = (
            MotionStabilizer() if virtual_cursor else None
        )

        # グローバルホットキー（ウィンドウ非表示でも終了/表示切替を受け付ける）
        self._hotkeys = HotkeyController(
            toggle_char=config.HOTKEY_TOGGLE_PREVIEW,
            quit_chars=(config.HOTKEY_QUIT,),
        )

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

        # キャリブレーション
        if not self._skip_calib:
            print("キャリブレーションを開始します...")
            success = self._run_calibration()
            if not success:
                print("キャリブレーション失敗またはキャンセルされました。")
                print("--skip-calib オプションでスキップできます。")
                self._cleanup()
                sys.exit(1)
            print("キャリブレーション完了")
        else:
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
        if self._stabilizer is not None:
            self._stabilizer.reset()
        print("リセット: 頭部基準とカーソル位置を再設定しました")

    def _run_calibration(self) -> bool:
        """キャリブレーションを実行"""
        overlay = CalibrationOverlay(
            screen_width=self._screen_w,
            screen_height=self._screen_h,
            gaze_callback=self._get_gaze_ratio_for_calibration,
        )

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
                if self._vcursor is not None:
                    # 仮想カーソルモード: OSカーソルには触れず仮想カーソルだけを動かす（表示のみ）。
                    # 動き安定化で微小なブレを削り「ぬるっと」追従させる。
                    sx, sy = self._stabilizer.update(result.x, result.y, time.time())
                    self._vcursor.update_position(sx, sy, visible=True)
                    self._pointer.update_position(sx, sy)
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
        """ミニプレビューを表示する。--debug時は詳細オーバーレイ、通常は小さな素のフレーム。"""
        if self._debug:
            self._draw_debug(frame, result)
        else:
            small = cv2.resize(frame, (480, 270)) if frame.shape[1] > 480 else frame
            cv2.imshow(config.DEBUG_WINDOW_NAME, small)

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
    )

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n中断されました。")


if __name__ == "__main__":
    main()
