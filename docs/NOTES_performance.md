# 性能・精度 向上ロードマップ

視線推定の精度／レイテンシ／FPS を高めるための施策メモ。
コードを読んで判明した実装可能項目と、参考になる外部リポジトリをまとめる。
**★** は費用対効果が高いと考える順。

## 本ブランチで着手済み

- **✅ MediaPipe Tasks `FaceLandmarker` への移行** (`src/gaze_estimator.py`)
  同梱の `models/face_landmarker.task` を使う新 Tasks API（VIDEOモード）へ移行。
  478点（虹彩含む）+ blendshapes を出力。旧 `mp.solutions.face_mesh` には自動フォールバック。
  下流の index ベース処理は `_LandmarksAdapter` で無改修。blendshapes は `last_blendshapes`
  で取得可能（瞬き・眉上げ判定の置換は次段）。
- **✅ スレッド化カメラ取得** (`src/camera_stream.py`, `--threaded-camera`)
  取得 (`VideoCapture.read`) を専用スレッドへ逃がし、メインループは常に最新フレームのみ処理。
  フレーム連番で同一フレームの再処理も回避。実効FPS・体感遅延を改善。
- **✅ 動き安定化「ぬるっと」** (`src/smoothing.py`)
  注視デッドゾーン + 間引き確定 + サッケード即追従で、ラグ由来のブレを抑えつつ滑らかに追従。

## 未着手（推奨度順）

### ★1. blendshapes ベースのジェスチャ判定
- 現状: 瞬きは EAR、眉上げは幾何比で判定。
- 改善: Tasks API の blendshapes（`eyeBlinkLeft/Right`, `browInnerUp` 等）に置換すると
  照明・個人差に強くなる。`GazeEstimator.last_blendshapes` で既に取得可能。
- 対象: `src/cursor_controller.py`（瞬き）、`src/precision_mode.py`（眉上げ）。

### ★2. キャリブレーションの正則化（Ridge 回帰）
- 現状: `compute_calibration` が多項式特徴量 + `np.linalg.lstsq`（最小二乗）。
- 問題: 16点に対し6次特徴量は過学習気味で、端で外れやすい。
- 改善: Ridge（L2正則化）で係数を安定化。`(AᵀA + λI)⁻¹ Aᵀ y` を解くだけで導入可能。
- 参考: `aciderix/React-Eye-Tracker-V1`（9点キャリブ + Ridge + head-pose補正）。

### ★3. カーソル移動の低レイテンシ化（macOS）
- 現状: 毎フレーム `pyautogui.moveTo`。pyautogui は内部オーバーヘッドが大きい。
- 改善: Quartz の `CGWarpCursorPosition` / `CGEventPost` を直接呼ぶ。
  既に PyObjC/Quartz に依存しているため追加依存なし。
- 対象: `src/cursor_controller.py`（通常モードのみ。`--virtual-cursor` は対象外）。

### 4. AIベース視線推定（appearance-based）への置換／併用
- 現状: 虹彩の幾何（目内相対位置）ベース。頭部・照明・個人差に弱い。
- 改善: L2CS-Net 等の appearance-based モデルで視線ベクトルを直接回帰。
- 参考:
  - `Chundurirohan/eye-tracking-cursor`（L2CS-Net + Kalman + dwell/blink、構成が酷似）
  - `yihuacheng/Gaze-Net` / `Full-face` / `Dilated-Net`（MPIIGaze 系 PyTorch 実装）
- 注意: モデル推論コストとのトレードオフ。MediaPipe と二段構えにするのが現実的。

### 5. Kalman フィルタの選択肢追加
- 現状: One Euro Filter（良好）。
- 改善: 等速度モデルの Kalman を代替として用意し、用途で切替。
- 参考: `Chundurirohan/eye-tracking-cursor`。

## 参考リポジトリ一覧

| リポジトリ | 着目点 |
|---|---|
| `Chundurirohan/eye-tracking-cursor` | L2CS-Net + Kalman + dwell/blink。最も構成が近い |
| `aciderix/React-Eye-Tracker-V1` | head-pose補正 + 9点 Ridge キャリブ + visual cursor |
| `yihuacheng/Gaze-Net` / `Full-face` / `Dilated-Net` | MPIIGaze 系 appearance-based（PyTorch） |
| `Vadson159/PhantomScript` | PyQt6 透明・クリックスルーオーバーレイ実装の手本 |
| `Sid-V5/GestureHud` | MediaPipe + 透明HUDオーバーレイでマウス操作 |
