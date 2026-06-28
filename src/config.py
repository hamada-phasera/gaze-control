"""GazeControl 全設定パラメータ集約モジュール"""

import os

import numpy as np

# --- カメラ設定 ---
CAMERA_INDEX = 0  # カメラデバイス番号
CAMERA_WIDTH = 1280  # キャプチャ横幅 (px) — 高解像度で虹彩精度向上
CAMERA_HEIGHT = 720  # キャプチャ縦幅 (px)
CAMERA_FPS = 30  # 目標FPS
# MediaPipe処理用ダウンサンプル解像度
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# --- MediaPipe Face Mesh 設定 ---
MAX_NUM_FACES = 1  # 検出する顔の最大数
MIN_DETECTION_CONFIDENCE = 0.5  # 顔検出の最小信頼度 (0.0〜1.0)
MIN_TRACKING_CONFIDENCE = 0.5  # 顔追跡の最小信頼度 (0.0〜1.0)
REFINE_LANDMARKS = True  # 虹彩ランドマーク有効化（478点モード）

# --- MediaPipe Tasks FaceLandmarker 設定 ---
# 新しい Tasks API（同梱の .task モデルを使用）。旧 mp.solutions.face_mesh より
# 高速・高精度で 478点（虹彩含む）+ blendshapes を出力する。
USE_FACE_LANDMARKER_TASKS = True  # True=Tasks API優先 / 失敗時は旧FaceMeshへ自動フォールバック
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task",
)
MIN_FACE_PRESENCE_CONFIDENCE = 0.5  # 顔存在の最小信頼度 (Tasks API用)
OUTPUT_BLENDSHAPES = True  # 表情係数（瞬き・眉上げ等）を出力する

# --- 虹彩ランドマークインデックス ---
# MediaPipe Face Mesh refine_landmarks=True 時のインデックス
LEFT_IRIS_CENTER = 468   # 左目虹彩中心
LEFT_IRIS_POINTS = [468, 469, 470, 471, 472]  # 左目虹彩5点
RIGHT_IRIS_CENTER = 473  # 右目虹彩中心
RIGHT_IRIS_POINTS = [473, 474, 475, 476, 477]  # 右目虹彩5点

# --- 目の端点インデックス ---
LEFT_EYE_INNER = 362   # 左目内角
LEFT_EYE_OUTER = 263   # 左目外角
RIGHT_EYE_INNER = 133  # 右目内角
RIGHT_EYE_OUTER = 33   # 右目外角

# --- 瞼インデックス（EAR計算用） ---
LEFT_EYE_TOP = 386     # 左上瞼
LEFT_EYE_BOTTOM = 374  # 左下瞼
RIGHT_EYE_TOP = 159    # 右上瞼
RIGHT_EYE_BOTTOM = 145 # 右下瞼

# --- 頭部姿勢推定用ランドマークインデックス ---
NOSE_TIP = 1          # 鼻先
CHIN = 199            # 顎
LEFT_EYE_LEFT_CORNER = 263   # 左目外角
RIGHT_EYE_RIGHT_CORNER = 33  # 右目外角
LEFT_MOUTH_CORNER = 287      # 左口角
RIGHT_MOUTH_CORNER = 57      # 右口角

# --- カメラフリップ設定 ---
CAMERA_FLIP_HORIZONTAL = True  # カメラ映像を左右反転（鏡像補正）

# --- 顔位置補正設定 ---
# 顔の移動（鼻先のズレ）を打ち消して視線を安定させる重み
# 大きいほど顔の動きの影響を除去するが、大きすぎると視線移動も打ち消す
HEAD_POSE_WEIGHT_YAW = 0.5    # 横方向の顔移動を打ち消す重み
HEAD_POSE_WEIGHT_PITCH = 0.4  # 縦方向の顔移動を打ち消す重み

# --- One Euro Filter 設定（虹彩比率用 — ノイズの根源を断つ）---
ONE_EURO_MIN_CUTOFF = 0.3   # 最小カットオフ周波数 (Hz) — 低いほど安定、高いほど反応速い
ONE_EURO_BETA = 0.5          # 速度感度 — 大きいほど動き出しに素早く反応
ONE_EURO_D_CUTOFF = 1.0      # 微分のカットオフ周波数

# --- One Euro Filter 設定（画面座標用 — 最終出力の安定化）---
ONE_EURO_SCREEN_MIN_CUTOFF = 0.8  # 画面座標用は少し反応寄り
ONE_EURO_SCREEN_BETA = 0.3        # 画面座標用の速度感度

# --- 適応的スムージング設定 ---
SACCADE_THRESHOLD = 200.0   # サッケード検出閾値 (px/frame)
SMOOTH_ALPHA_FAST = 0.25    # サッケード時のEMA α
SMOOTH_ALPHA_MEDIUM = 0.08  # 中速移動時のEMA α
SMOOTH_ALPHA_SLOW = 0.02    # 静止時のEMA α
VELOCITY_MEDIUM_THRESHOLD = 50.0  # 中速と低速の境界 (px/frame)
ADAPTIVE_DEAD_ZONE_MIN = 5.0   # デッドゾーン最小半径 (px)
ADAPTIVE_DEAD_ZONE_MAX = 20.0  # デッドゾーン最大半径 (px)

# --- 慣性（イナーシャ）設定 ---
# カーソルはバネ+摩擦モデルで動く: ターゲットに向かって加速し、摩擦で減速する
INERTIA_ACCELERATION = 0.06   # バネ定数（小さく→ゆっくり滑らかに追従）
INERTIA_FRICTION = 0.82       # 摩擦係数（大きく→よく滑る、慣性が長く残る）
INERTIA_MAX_SPEED = 80.0      # カーソルの最大速度 (px/frame)
INERTIA_MIN_SPEED = 0.3       # この速度以下で停止 (px/frame)

# --- 視線ポインター設定 ---
POINTER_RADIUS = 30  # ポインター円の半径 (px)
POINTER_COLOR = (0, 200, 255)  # ポインター色 BGR (オレンジ系)
POINTER_ALPHA = 0.4  # ポインター透明度 (0.0=透明〜1.0=不透明)
POINTER_BORDER_WIDTH = 3  # ポインター枠線の太さ (px)
POINTER_CENTER_DOT = 4  # 中心ドットの半径 (px)

# --- Dwell Click 設定 ---
DWELL_RADIUS = 50.0  # Dwell判定半径 (px)
DWELL_TIME = 1.2  # Dwell判定時間 (秒)
DWELL_COOLDOWN = 0.8  # クリック後のクールダウン (秒)

# --- 瞬きクリック設定 ---
BLINK_EAR_THRESHOLD = 0.2  # EAR閾値 (これ以下で瞬き判定)
BLINK_FRAMES_MIN = 3  # 意図的な瞬きの最小フレーム数
BLINK_FRAMES_MAX = 12  # 意図的な瞬きの最大フレーム数（超えると目を閉じている）

# --- キャリブレーション設定 ---
CALIBRATION_POINTS = 16  # キャリブレーションポイント数 (4x4)
CALIBRATION_GRID_SIZE = 4  # グリッドサイズ (4x4)
CALIBRATION_MARGIN = 0.08  # 画面端からのマージン (0.0〜0.5, 8%)
CALIBRATION_DURATION = 1.5  # 各ポイントでのデータ収集時間 (秒)
CALIBRATION_MIN_POINTS = 8  # キャリブレーション成立に必要な最小データ点数（多項式回帰に合わせて増加）
CALIBRATION_RANDOMIZE = True  # キャリブレーション点の表示順をランダムにする

# キャリブレーション結果の保存先（1回やれば次回から自動読込でdotスキップ）
CALIBRATION_FILE = os.path.join(os.path.expanduser("~"), ".gaze_control_calibration.json")

# 再センタリング: 実行中に 'c' キーで「画面中央を見て」追加オフセットを自動補正。
# 一定方向のバイアス（見ている点から常に同じ方向にずれる）をその場で打ち消す。
RECENTER_SAMPLES = 18      # 平均に使う視線フレーム数（約0.6秒@30fps）

# --- 可動域・距離適応 ---
GAZE_RANGE_MULT = 1.0       # 可動域（ゲイン）倍率 — 大きいほど視線で広く届く（--rangeで上書き）
DISTANCE_ADAPT = 0.0        # 距離適応の強さ (0=無効〜1=フル, --distance-adaptで上書き)
# 距離推定の基準（目の外角どうしの正規化距離と、その時の概算距離cm）。
# 近いほど目が大きく写る＝inter_eyeが大きい。距離 ≒ DISTANCE_REF_CM * (REF_INTER_EYE / inter_eye)
DISTANCE_REF_INTER_EYE = 0.13  # 約40cm時の目外角間の正規化距離（概算・実機で調整可）
DISTANCE_REF_CM = 40.0         # その基準距離 (cm)

# --- 感度設定 ---
DEFAULT_SENSITIVITY = 3.0  # デフォルト感度 (1.0〜10.0, 低めの方が安定)

# --- 利き目（ドミナントアイ）---
# 人は利き目で狙う。両目平均だと利き目と非利き目の中間にズレる（利き目が右なら左へズレやすい）。
DOMINANT_EYE = "both"        # "right" / "left" / "both"（--eye）
DOMINANT_EYE_WEIGHT = 0.5    # 利き目の重み 0.5=両目均等〜1.0=利き目のみ（--eye-weight）

# --- 出力オフセット補正（系統的なズレの微調整, px）---
OFFSET_X = 0.0   # 右が正（--offset-x。カーソルが左にズレるなら正の値）
OFFSET_Y = 0.0   # 下が正（--offset-y。カーソルが下にズレるなら負の値）

# --- 縦方向の可動域調整（カメラが画面上部 → 下方向が届きにくい/精度低下の対策）---
VERTICAL_GAIN = 1.0          # 縦ゲイン倍率（>1で上下に広く届く。--v-gain で上書き）
VERTICAL_DOWN_BOOST = 0.0    # 下を見るほど縦可動量を追加で増やす量 (0で無効。--down-boost で上書き)
VERTICAL_DOWN_SIGN = 1.0     # 「下方向」に対応する avg_y の符号（ブーストが上方向に効くなら -1.0 に）
VERTICAL_DOWN_REF = 0.5      # down_boost が最大になる avg_y の大きさ（正規化基準）
VERTICAL_DOWN_SMOOTH = 0.0   # 下を見るほど縦を強く平滑化（下端の分散・ブレ抑制）。0で無効, 0.6前後推奨。--down-smooth

# --- 磁石スナップ（近くのボタン/カード/ウィンドウにカーソルを吸着）---
SNAP_MAGNET_RADIUS = 130.0   # この距離内に要素があれば吸着 (px, --snap-radius で上書き)

# --- 頭部姿勢推定設定 ---
# solvePnP用の3D顔モデル座標（mm単位、鼻先が原点）
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # 鼻先 (NOSE_TIP)
    [0.0, -63.6, -12.5],      # 顎 (CHIN)
    [-43.3, 32.7, -26.0],     # 左目外角 (LEFT_EYE_LEFT_CORNER)
    [43.3, 32.7, -26.0],      # 右目外角 (RIGHT_EYE_RIGHT_CORNER)
    [-28.9, -28.9, -24.1],    # 左口角 (LEFT_MOUTH_CORNER)
    [28.9, -28.9, -24.1],     # 右口角 (RIGHT_MOUTH_CORNER)
], dtype=np.float64)

# 頭部姿勢→画面座標の感度 (px/degree)
HEAD_SENSITIVITY_YAW = 50.0    # 横方向: 1度あたり50px移動
HEAD_SENSITIVITY_PITCH = 40.0  # 縦方向: 1度あたり40px移動

# 縦の頭部アシスト（横は視線のまま、縦だけ頭のピッチで可動域を広げる）。
# 目の上下の動きは構造的に小さいので、頭の傾き(うなずき)で縦リーチを補助する。
# px/度。0で無効（純粋視線）。--head-assist で上書き。
HEAD_PITCH_ASSIST = 45.0

# 頭ドリフト補正: キャリブ時の頭の向き(基準)から横にズレた分を差し引く。
# px/度。0で無効。符号は実機で調整（逆なら負値）。--head-comp で上書き。
HEAD_COMP_X = 0.0

# 頭部正規化（推奨）: キャリブ時の頭の向き(基準)からのヨー/ピッチのズレを、
# 多項式キャリブの「手前」で虹彩比率から差し引く。面外回転（顔の向き）で
# 見かけの虹彩位置がズレるのを打ち消すため、head-comp（画面後段）より素直に効く。
# strength=0 で無効。横と縦をまとめて 1 つの強さで調整し、定数 K で deg→比率に変換。
# K は「ヨー/ピッチ 1 度あたり、0.5中心の比率がどれだけ動くか」の概算。
GAZE_HEAD_NORM = 0.0          # 強さ（0=無効, 1.0前後から試す, 逆効きなら負値）。--head-normalize
GAZE_HEAD_NORM_KX = 0.0067    # ヨー deg→比率（15度で約0.10=画面1割ぶん）
GAZE_HEAD_NORM_KY = 0.0067    # ピッチ deg→比率

# 3D眼内ベクトル（④-lite）: 虹彩比率を「目のローカル3D平面」で測る。
# MediaPipeが既に出す3D(z)座標を使い、頭の向きが変わってもu/v基底ごと回るので
# 見かけの虹彩ズレが原理的に相殺される（2Dの画像投影で起きる前後ズレを除去）。
# さらに左右の目で符号をそろえ、両目が補強し合う（2Dは目頭→目尻が左右逆で相殺しがち）。
# True=有効。信号の符号/スケールが2Dと変わるので有効化時は要 --recalibrate。--gaze-3d
GAZE_3D = False

# 追加ランドマークインデックス
GLABELLA_INDEX = 168           # 眉間（鼻梁上部）
FOREHEAD_CENTER_INDEX = 10     # 額中心

# One Euro Filter（頭部姿勢用）
ONE_EURO_HEAD_MIN_CUTOFF = 0.5
ONE_EURO_HEAD_BETA = 0.3

# --- 視線+頭部融合設定 ---
# GAZE_ONLY=True なら頭部融合を使わず純粋に視線で決める（視線カーソルの既定）。
# 頭部融合を使う場合でも、視線が支配的になるよう重みを高めに設定。
GAZE_ONLY = True             # True=視線オンリー（頭部融合なし）。--fuse-head で融合に切替
FUSION_W_GAZE_MIN = 0.85      # 静止時の視線重み（視線を主、頭部は微補助 ≤15%）
FUSION_W_GAZE_MAX = 0.97      # サッケード時の視線重み
FUSION_SACCADE_THRESHOLD = 300.0  # サッケード検出閾値 (px/s)

# --- 精密モード設定（眉上げトリガー）---
PRECISION_MODE_BROW_THRESHOLD = 0.025  # 眉上げ検出閾値（正規化距離）
PRECISION_MODE_ENTER_DURATION = 0.3    # 精密モード開始に必要な持続時間 (秒)
PRECISION_MODE_EXIT_DURATION = 0.5     # 精密モード終了に必要な持続時間 (秒)
PRECISION_MODE_SENSITIVITY_MULT = 2.5  # 精密モード時の頭部感度倍率

# 眉ランドマークインデックス
LEFT_EYEBROW_INDICES = [282, 283, 285, 293, 295]
RIGHT_EYEBROW_INDICES = [52, 53, 55, 63, 65]

# --- Blendshape（表情係数）ベース判定 ---
# Tasks FaceLandmarker が出力する blendshapes は ARKit 互換スコア (0〜1)。
# EAR/幾何より照明・距離・個人差に強い。blendshape が無い（旧FaceMesh）場合は
# 自動的に従来手法（EAR / 眉-目間距離）へフォールバックする。
USE_BLENDSHAPE_BLINK = True                 # 瞬きを blendshape で判定する
BLINK_BLENDSHAPE_THRESHOLD = 0.5            # eyeBlink スコアがこれ以上で「閉眼」
USE_BLENDSHAPE_BROW = True                  # 眉上げを blendshape で判定する
PRECISION_MODE_BLENDSHAPE_THRESHOLD = 0.4   # browUp スコアがこれ以上で「眉上げ」

# 視線信号に eyeLook blendshape を使う（生の虹彩比率より頭ブレに強い・縦も素直）。実験的。
USE_BLEND_GAZE = False        # True で有効（--blend-gaze）
BLEND_GAZE_GAIN = 4.0         # blendshape視線量([-1,1])→正規化座標へのゲイン

BLENDSHAPE_EYE_BLINK_LEFT = "eyeBlinkLeft"
BLENDSHAPE_EYE_BLINK_RIGHT = "eyeBlinkRight"
BLENDSHAPE_BROW_INNER_UP = "browInnerUp"
BLENDSHAPE_BROW_OUTER_UP_LEFT = "browOuterUpLeft"
BLENDSHAPE_BROW_OUTER_UP_RIGHT = "browOuterUpRight"

# --- Accessibility スナップ設定 ---
SNAP_ENABLED = True            # UI要素スナップ有効/無効
SNAP_RADIUS = 80.0             # スナップ検索半径 (px)
SNAP_SEARCH_STEP = 10          # グリッド検索ステップ (px)

# --- 形状スナップ（カーソルがボタン/カードの形に変化）---
SNAP_SHAPE_QUERY_INTERVAL = 0.12  # AX要素問い合わせの間隔 (秒) — 毎フレームは重いので間引く
SNAP_SHAPE_PAD = 14               # 要素矩形の外側に足すグロー余白 (px)
SNAP_SHAPE_CORNER_RATIO = 0.28    # 角丸半径 / min(w,h)
SNAP_SHAPE_BLUR_RATIO = 0.5       # 矩形グローのブラー幅（パッド比）
SNAP_SHAPE_MAX_OPACITY = 0.55     # 形状ハイライトの最大不透明度
SNAP_SHAPE_MIN_SIZE = 24          # これ未満の小さな要素にはスナップしない (px)
SNAP_SHAPE_MAX_SIZE = 900         # これを超える巨大要素にはスナップしない (px)

# --- 長い閉眼ジェスチャ（リセット）---
EYES_CLOSED_RESET_SEC = 3.0    # この秒数だけ目を閉じ続けるとリセット発火

# --- 仮想カーソル設定（gaze駆動・OSカーソルとは独立した表示専用カーソル）---
# 視線で動かす「仮想カーソル」を、広範囲ブラーの円形グローとして
# 透明・クリックスルーのオーバーレイに描画する。OSの実マウスには一切干渉しない。
VIRTUAL_CURSOR_DIAMETER = 180        # スプライト全体の一辺 (px) — 広範囲ブラーを内包するため大きめ
VIRTUAL_CURSOR_CORE_RATIO = 0.28     # くっきり見える中心円の半径比 (0.0〜1.0)
VIRTUAL_CURSOR_BLUR_RATIO = 0.55     # ブラーの広がり (大きいほど halo が広範囲に)
VIRTUAL_CURSOR_COLOR = (0, 200, 255)  # カーソル色 RGB (水色系のグロー)
VIRTUAL_CURSOR_MAX_OPACITY = 0.85    # 最大不透明度 (0.0〜1.0)
VIRTUAL_CURSOR_SMOOTHING = 0.22      # 追従応答性 (0<r<=1) — 大きいほど移動が速い（--smoothingで上書き可）
VIRTUAL_CURSOR_TICK_DT = 1.0 / 120.0  # オーバーレイ再描画間隔 (秒) — 滑らかな動きのため高頻度
VIRTUAL_CURSOR_MAX_SPEED = 1600.0    # 最大速度 (px/秒) — 点間の移動の速さ。固定は注視ロックが担うので速めでOK（--max-speed, 0で無制限）

# --- サイズ・イージング（移動中は縮み、止まると膨らむ。点と点の距離を目立たなくする）---
VIRTUAL_CURSOR_SCALE_DIP = 0.45      # 最大速度時にどれだけ縮むか (0〜1, 0.45=55%まで縮小)
VIRTUAL_CURSOR_SCALE_RESP = 0.30     # スケールのイージング応答性 (小さいほどゆっくり伸縮)

# --- 注視（フィキセーション）ベースのターゲット確定 ---
# 視線が dwell_time 秒とどまったら、その滞留中の履歴の重心へカーソルを移す。
# 移動中・ノイズ中はターゲットを保持し、途中の距離情報は無視する（情報を削る）。
FIXATION_DWELL_TIME = 0.3            # 注視確定までの滞留時間 (秒, 0.2〜0.5推奨, --dwell-timeで上書き可)
FIXATION_RADIUS = 80.0              # 「とどまっている」とみなす重心からの最大ばらつき (px)
# ロック解除（カーソル乗り換え）に必要な視線移動距離 (px)。
# 大きいほど「一度決めた所にピタッと固定」され、近くのブレ・頭ドリフトでは動かない。
FIXATION_RELEASE_RADIUS = 110.0     # （--hold-radius で上書き可）

# --- 動き安定化（「ぬるっと」感 / ラグ由来のブレ抑制）---
# リアルタイムすぎると微小なラグ・ノイズで視線が落ち着かないため、
# データを間引き＋注視デッドゾーンで安定させ、滑らかな追従を作る。
STAB_DEADZONE = 22.0          # この半径内の揺れは無視して注視点を保持 (px)
STAB_SACCADE = 280.0          # この距離以上の移動は即追従（素早い視線移動を妨げない, px）
STAB_COMMIT_INTERVAL = 0.07   # 中間移動の目標確定間隔 (秒) — フレームを間引いてブレを削る
STAB_COMMIT_RATIO = 0.45      # 確定時に新座標へ寄せる割合 (0〜1, 小さいほど緩やか)

# --- ホットキー設定 ---
HOTKEY_TOGGLE_PREVIEW = "p"   # プレビューウィンドウ表示/非表示の切替キー
HOTKEY_QUIT = "q"             # 終了キー（ESCも可）

# --- デバッグ設定 ---
DEBUG_WINDOW_NAME = "GazeControl Debug"  # デバッグウィンドウ名
