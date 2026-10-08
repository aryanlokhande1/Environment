"""Materiality regression tests independent of simulator/model behavior."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('certification_statistics', Path(__file__).parents[1]/'tools/certification_statistics.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


@pytest.mark.parametrize('difference,interval,margin,expected', [
    (0., [-.02, .02], .05, rules.PASS),
    (.028, [.0135, .042], .05, rules.LIMITATION),
    (-.028, [-.042, -.0135], .05, rules.LIMITATION),
    (.07, [.06, .08], .05, rules.FAIL),
    (-.07, [-.08, -.06], .05, rules.FAIL),
    (.028, [.01, .07], .05, rules.INCONCLUSIVE),
    (0., [-.2, .2], .05, rules.INCONCLUSIVE),
    (.15, [.01, .30], .70, rules.LIMITATION),
])
def test_practical_equivalence_retains_uncertainty(difference, interval, margin, expected):
    assert rules.equivalence(difference, interval, margin) == expected


@pytest.mark.parametrize('difference,interval,expected', [
    (.00788, [-.02931, .04507], rules.PASS),
    (.02710, [-.07712, .13131], rules.PASS),
    (-.10, [-.15, -.05], rules.PASS),
    (.10, [.05, .15], rules.INCONCLUSIVE),
])
def test_loop_point_increase_is_not_a_material_failure(difference, interval, expected):
    assert rules.loop_inflation(difference, interval) == expected


def test_loop_material_increase_requires_effect_size_and_uncertainty():
    assert rules.loop_inflation(.20, [.15, .25], .10) == rules.FAIL
    assert rules.loop_inflation(.02, [.01, .03], .10) == rules.LIMITATION


@pytest.mark.parametrize('difference,interval,margin', [(float('nan'), [0, 1], .1), (0, [1, -1], .1), (0, [-1, 1], 0), (.2, [0, .1], .1)])
def test_invalid_uncertainty_cannot_certify(difference, interval, margin):
    with pytest.raises(ValueError):
        rules.equivalence(difference, interval, margin)
