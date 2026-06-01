# gaze-control — 視線 + 頭部姿勢でカーソルを操作

> Webカメラだけで、視線と頭の向きを推定して Mac のマウスカーソルを動かすハンズフリー操作システム。アクセシビリティ／ハンズフリーUIの実験。

[![Python](https://img.shields.io/badge/Python-3.11-3776ab)](https://www.python.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5c3ee8)](https://opencv.org/) [![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-00a37a)](https://mediapipe.dev/)

---

## 概要

特別なハードウェア無しで、**Webカメラ映像から視線方向と頭部姿勢を推定**し、それらを融合してカーソル位置を決定する。キャリブレーションで個人差を吸収し、`accessibility_snap` で画面上のUI要素にスナップさせる工夫を持つ。

---

## 構成

| モジュール | 役割 |
|---|---|
| `gaze_estimator.py` | 視線方向の推定 |
| `head_pose_estimator.py` | 頭部姿勢（向き）の推定 |
| `fusion.py` | 視線 × 頭部姿勢の融合ロジック |
| `calibration.py` | 個人差を補正するキャリブレーション |
| `cursor_controller.py` / `gaze_pointer.py` | カーソル制御 |
| `accessibility_snap.py` | UI要素へのスナップ |

---

## 技術スタック

`Python 3.11` `OpenCV` `MediaPipe (FaceMesh)` `NumPy` `PyAutoGUI` `pynput` `PyObjC (Quartz / ApplicationServices)`

---

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src        # 実行（カメラ権限が必要）
pytest               # テスト
```

> ⚠️ macOS では「アクセシビリティ」「カメラ」のシステム権限が必要です。

---

## このプロジェクトで見せられること

- **コンピュータビジョン × リアルタイム制御**（推定→融合→アクチュエーション）
- **アクセシビリティ志向**のハンズフリーUI設計
- OS低レイヤ（PyObjC/Quartz）との連携

---

*※ ポートフォリオ目的の公開リポジトリです。*
