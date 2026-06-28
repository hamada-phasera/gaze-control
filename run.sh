#!/usr/bin/env bash
# GazeControl ランチャー — venv を自動で有効化して起動する。
#
#   ./run.sh                  # 推奨プリセットで起動（仮想カーソル＋形状スナップ等）
#   ./run.sh --recalibrate    # 追加/上書きフラグを渡す（プリセットを使わず全部指定も可）
#   ./run.sh --sensitivity 1.5 --max-speed 500
#
# 毎回 `source .venv/bin/activate` する必要はありません。
set -e
cd "$(dirname "$0")"

# 初回のみ: 仮想環境を作って依存をインストール
if [ ! -x ".venv/bin/python" ]; then
  echo "初回セットアップ: .venv を作成して依存をインストールします..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

# 引数が無ければ推奨プリセットで起動。引数があればそれをそのまま渡す。
if [ "$#" -eq 0 ]; then
  set -- --virtual-cursor --show-window --debug --no-hotkeys --snap-shape \
        --sensitivity 1.0 --range 1.5 --distance-adapt 0.5 --dwell-time 0.3 --max-speed 700
fi

exec ./.venv/bin/python main.py "$@"
