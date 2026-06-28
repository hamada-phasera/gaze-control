# 実機チューニングガイド

実機（mac + Webカメラ）で触りながら調整するためのチートシート。
パラメータは基本 `src/config.py`。CLIフラグは `python main.py --help` でも確認できる。

## 1. まず動かす

```bash
pip install -r requirements.txt   # PySide6 / mediapipe / pynput を含む

python main.py --virtual-cursor                        # 仮想カーソルだけ動かす（窓なし・実マウス非干渉）
python main.py --virtual-cursor --threaded-camera      # ＋取得スレッド化でFPS底上げ
python main.py --virtual-cursor --show-window --debug  # 様子を見たい時（'p' で窓トグル）
python main.py --blink-click                           # OSカーソル＋瞬きクリック
python main.py --precision-mode                        # 眉上げで精密モード自動切替
```

操作中のキー: **`r`** = リセット（頭部基準・カーソル位置を中央へ再設定）、**`p`** = プレビュー窓 表示/非表示、**`q` / ESC** = 終了。
ジェスチャ: **目を3秒間閉じる → リセット**（`EYES_CLOSED_RESET_SEC` で調整）。
形状スナップ: **`--snap-shape`** で、注視先のボタン/カードの形にカーソルが変形（macOS Accessibility, 実験的）。

よく使うフラグ:
- `--dwell-time 0.3` … **注視確定までの滞留時間（秒）**。この時間ある場所を見続けると、その間の履歴の重心へカーソルが移る。移動中・ノイズ中は移動しない（0.2〜0.5推奨）
- `--max-speed 600` … 最大速度 px/秒（小さいほど遅く見失いにくい。0で無制限）
- `--no-hotkeys` … pynput を使わない（macでメインループ開始時に `trace trap` する場合の回避。プレビュー窓のキーで操作）
- `--sensitivity 1.0` … 視線→移動量の倍率（端に飛ぶのを抑える。既定 3.0）
- `--smoothing 0.1` … 仮想カーソルの追従の滑らかさ（既定 0.12, 小さいほど遅い）
- `--camera-index 1` … 内蔵カメラが 0 で開かない時に別番号
- `--skip-calib` … キャリブ省略（**動作確認用**。実用では付けない＝キャリブした方が圧倒的に安定）
- `--range 1.5` … 可動域（ゲイン）倍率。大きいほど視線で広く届く（注視ベースなので上げても端に飛ばない）
- `--distance-adapt 0.5` … 距離適応 0〜1。近い/遠いでゲインを自動調整（0=無効）
- `--recalibrate` … 保存済みキャリブを無視してやり直す（既定は**1回やれば次回から自動読込でdotスキップ**）
- `--v-gain 1.3` … 縦ゲイン倍率（上下に広く届く）
- `--down-boost 0.5` … **下を見るほど縦可動量が増える**（カメラが上にある＝下が届きにくい対策）。⚠️ `--v-gain`/`--down-boost` を変えたら **`--recalibrate` 推奨**（キャリブと整合させるため）
- `--snap-radius 130` … 磁石スナップの吸着半径（近くのボタン/カードに吸い付く距離）

### 下方向が届かない/精度が落ちるとき

カメラが画面上部にあるため、下を見るほど目が下を向き精度が落ちやすい。`--down-boost` で「下ほど可動量を増やす」と届きやすくなる。例（キャリブもやり直す）:
```bash
./run.sh --down-boost 0.6 --v-gain 1.2 --recalibrate
```
ブーストが上方向に効いてしまう場合は `src/config.py` の `VERTICAL_DOWN_SIGN = -1.0`。

### 磁石スナップ（近くの要素に吸着）

`--snap-shape` で、近く（`--snap-radius` 内）のボタン/カード/ウィンドウにカーソルが**吸い付き**、その形に変形する。吸着が強すぎ/弱すぎなら `--snap-radius` を調整。

### キャリブは1回でOK（毎回のdotが不要）

初回にキャリブすると `~/.gaze_control_calibration.json` に保存され、次回から**自動読込**して dot を出しません。やり直したい時だけ `--recalibrate`。`--debug` 表示の `Dist:~XXcm` が **30〜50cm（OK表示）** になる位置で使うと最適です。

### 動作モデル（注視ベース）

仮想カーソルは「常に視線を追う」のではなく、**視線がとどまった所だけへ移る**:
1. 視線が `--dwell-time` 秒、半径 `FIXATION_RADIUS`(px) 内にとどまる → その間の**履歴の重心**を確定ターゲットにする（毎回ピタリ合わせなくてよい）。
2. 視線移動中・ノイズ中はターゲットを**保持**（途中の距離は無視＝情報を削る）。
3. 確定ターゲットへは `--max-speed`/`--smoothing` でゆっくり滑る。移動中はカーソルが**縮み**、止まると**膨らむ**（`VIRTUAL_CURSOR_SCALE_DIP`）。

調整(`src/config.py`): `FIXATION_RADIUS`（とどまり判定の広さ）, `FIXATION_MIN_MOVE`（再確定の最小距離）, `VIRTUAL_CURSOR_SCALE_DIP`/`SCALE_RESP`（縮み量と伸縮の速さ）。

> 窓を出すと自分の目を見てしまい視線が引っ張られるので、基本は**窓なし**で使い、確認したい時だけ `p` で一瞬出す。

## 2. 症状 → いじる場所（`src/config.py`）

### 動き・追従感
| 症状 | パラメータ | 方向 |
|---|---|---|
| 揺れる・落ち着かない | `STAB_DEADZONE` ↑ / `VIRTUAL_CURSOR_SMOOTHING` ↓ | 安定寄り |
| 反応が鈍い・遅れすぎ | `VIRTUAL_CURSOR_SMOOTHING` ↑ / `STAB_DEADZONE` ↓ / `STAB_COMMIT_INTERVAL` ↓ | 機敏寄り |
| 速い視線移動で置いていかれる | `STAB_SACCADE` ↓ | 即追従しやすく |
| 「ぬるっと」感が足りない | `STAB_COMMIT_INTERVAL` ↑ / `STAB_COMMIT_RATIO` ↓ / `VIRTUAL_CURSOR_SMOOTHING` ↓ | より滑る |
| 注視時に少しずつズレる | `STAB_DEADZONE` ↑ | 注視保持を強く |

### 精度（座標が合わない）
| 症状 | 対応 |
|---|---|
| 全体的にズレる | キャリブをやり直す（`--skip-calib` を付けない）。明るい環境で顔を動かさず |
| 端だけ大きくズレる | `--sensitivity` を下げる（既定 `DEFAULT_SENSITIVITY=3.0`）。将来 Ridge キャリブで改善予定 |
| 顔を動かすとカーソルも動く | `HEAD_POSE_WEIGHT_YAW` / `HEAD_POSE_WEIGHT_PITCH`、融合重み `FUSION_W_GAZE_*` |

### 見た目（仮想カーソル）
| 項目 | パラメータ |
|---|---|
| 円の大きさ | `VIRTUAL_CURSOR_DIAMETER` |
| 中心円の明瞭さ | `VIRTUAL_CURSOR_CORE_RATIO` |
| ブラーの広がり | `VIRTUAL_CURSOR_BLUR_RATIO` |
| 色 / 濃さ | `VIRTUAL_CURSOR_COLOR`（RGB）/ `VIRTUAL_CURSOR_MAX_OPACITY` |

### クリック・ジェスチャ（blendshape）
| 症状 | パラメータ | 方向 |
|---|---|---|
| 瞬きクリックが誤爆する | `BLINK_BLENDSHAPE_THRESHOLD` ↑ | 鈍く |
| 瞬きクリックが反応しない | `BLINK_BLENDSHAPE_THRESHOLD` ↓ | 敏感に |
| 眉上げ精密モードが入らない | `PRECISION_MODE_BLENDSHAPE_THRESHOLD` ↓ / `PRECISION_MODE_ENTER_DURATION` ↓ | 入りやすく |
| 精密モードが入りっぱなし | `PRECISION_MODE_BLENDSHAPE_THRESHOLD` ↑ / `PRECISION_MODE_EXIT_DURATION` ↓ | 抜けやすく |

> blendshape が出ない環境では EAR/幾何へ自動フォールバック。その場合は `BLINK_EAR_THRESHOLD` や
> `PRECISION_MODE_BROW_THRESHOLD` 側を調整する。

## 3. 実機で詰まりやすい所

- **macOS 権限**: システム設定 →「アクセシビリティ」「カメラ」、`pynput` ホットキー用に「入力監視」を許可。
- **依存**: `PySide6`（仮想カーソル）/ `mediapipe`（推定）/ `pynput`（ホットキー）が入っているか。
- **起動ログ**を確認: `FaceLandmarker (Tasks API) を初期化しました` が出れば新API。出ず
  `MediaPipe FaceMesh (legacy)` ならフォールバック（blendshape無効＝EAR/幾何判定）。
- 仮想カーソルが出ない場合: PySide6 未導入か GUI 不可。ログに警告が出る（本体は動作継続）。

## 4. フィードバックの渡し方

「ぬるっとしすぎ／カクつく／瞬きが誤爆する／端がズレる」など**症状を一言**もらえれば、
上表の該当パラメータ調整、または安定化・判定ロジック自体の改修で対応する。
