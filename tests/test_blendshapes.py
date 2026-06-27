"""blendshapes ユーティリティのテスト（mediapipe非依存）"""

from src import blendshapes, config


class FakeCat:
    """Tasks API の Category を模したフェイク"""

    def __init__(self, name: str, score: float, alias: bool = False) -> None:
        if alias:
            self.categoryName = name  # noqa: N815 — API互換名
        else:
            self.category_name = name
        self.score = score


class TestScoreMap:
    def test_none_and_empty(self) -> None:
        assert blendshapes.score_map(None) == {}
        assert blendshapes.score_map([]) == {}

    def test_builds_dict(self) -> None:
        cats = [FakeCat("eyeBlinkLeft", 0.8), FakeCat("browInnerUp", 0.3)]
        m = blendshapes.score_map(cats)
        assert m["eyeBlinkLeft"] == 0.8
        assert m["browInnerUp"] == 0.3

    def test_supports_camel_alias(self) -> None:
        cats = [FakeCat("eyeBlinkRight", 0.5, alias=True)]
        m = blendshapes.score_map(cats)
        assert m["eyeBlinkRight"] == 0.5


class TestBlinkScore:
    def test_average_of_both_eyes(self) -> None:
        scores = {
            config.BLENDSHAPE_EYE_BLINK_LEFT: 0.8,
            config.BLENDSHAPE_EYE_BLINK_RIGHT: 0.6,
        }
        assert blendshapes.blink_score(scores) == 0.7

    def test_single_eye(self) -> None:
        scores = {config.BLENDSHAPE_EYE_BLINK_LEFT: 0.9}
        assert blendshapes.blink_score(scores) == 0.9

    def test_none_when_absent(self) -> None:
        assert blendshapes.blink_score({}) is None
        assert blendshapes.blink_score({"browInnerUp": 0.5}) is None


class TestBrowRaiseScore:
    def test_inner_up(self) -> None:
        scores = {config.BLENDSHAPE_BROW_INNER_UP: 0.6}
        assert blendshapes.brow_raise_score(scores) == 0.6

    def test_max_of_inner_and_outer_avg(self) -> None:
        scores = {
            config.BLENDSHAPE_BROW_INNER_UP: 0.3,
            config.BLENDSHAPE_BROW_OUTER_UP_LEFT: 0.8,
            config.BLENDSHAPE_BROW_OUTER_UP_RIGHT: 0.6,
        }
        # max(0.3, (0.8+0.6)/2=0.7) = 0.7
        assert blendshapes.brow_raise_score(scores) == 0.7

    def test_none_when_absent(self) -> None:
        assert blendshapes.brow_raise_score({}) is None
        assert blendshapes.brow_raise_score({"eyeBlinkLeft": 0.5}) is None
