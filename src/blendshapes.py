"""Blendshape（表情係数）ユーティリティ — Tasks FaceLandmarker 出力の解釈

MediaPipe Tasks の FaceLandmarker は ARKit 互換の blendshapes を Category の
リストとして返す（各要素は `category_name` と `score`(0〜1) を持つ）。
本モジュールはそれを名前→スコアの辞書に変換し、瞬き・眉上げのスコアを抽出する。

EAR（目の縦横比）や眉-目間距離の幾何計算と違い、blendshape はモデルが学習した
正規化スコアなので、照明・カメラ距離・個人差に強い。

mediapipe に依存しない純粋ロジックで、フェイクの Category でユニットテストできる。
"""

from __future__ import annotations

from typing import Dict, Optional

from . import config


def score_map(blendshapes: object) -> Dict[str, float]:
    """blendshapes(Categoryのリスト)を {名前: スコア} の辞書に変換する。

    None・空・解釈不能な要素は無視する。`category_name`/`categoryName` の
    どちらの命名にも対応する。
    """
    if not blendshapes:
        return {}
    out: Dict[str, float] = {}
    for cat in blendshapes:
        name = getattr(cat, "category_name", None)
        if name is None:
            name = getattr(cat, "categoryName", None)
        score = getattr(cat, "score", None)
        if name is not None and score is not None:
            try:
                out[name] = float(score)
            except (TypeError, ValueError):
                continue
    return out


def blink_score(scores: Dict[str, float]) -> Optional[float]:
    """両目の eyeBlink 平均スコアを返す。該当が無ければ None。"""
    if not scores:
        return None
    left = scores.get(config.BLENDSHAPE_EYE_BLINK_LEFT)
    right = scores.get(config.BLENDSHAPE_EYE_BLINK_RIGHT)
    vals = [v for v in (left, right) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def brow_raise_score(scores: Dict[str, float]) -> Optional[float]:
    """眉上げスコアを返す。browInnerUp と 左右 browOuterUp 平均の大きい方。

    該当ブレンドシェイプが一つも無ければ None。
    """
    if not scores:
        return None
    candidates = []
    inner = scores.get(config.BLENDSHAPE_BROW_INNER_UP)
    if inner is not None:
        candidates.append(inner)
    outer_l = scores.get(config.BLENDSHAPE_BROW_OUTER_UP_LEFT)
    outer_r = scores.get(config.BLENDSHAPE_BROW_OUTER_UP_RIGHT)
    outers = [v for v in (outer_l, outer_r) if v is not None]
    if outers:
        candidates.append(sum(outers) / len(outers))
    if not candidates:
        return None
    return max(candidates)
