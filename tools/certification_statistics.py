"""Validation-only materiality decisions; never imported by the simulator."""
from __future__ import annotations
import math

PASS = 'PASS'
LIMITATION = 'PASS WITH DOCUMENTED LIMITATION'
FAIL = 'FAIL'
INCONCLUSIVE = 'INCONCLUSIVE'
INTENSITY_RELATIVE_MARGIN = .10  # docs/VALIDATION.md, pre-declared Intensity gate
PTP_ABSOLUTE_MARGIN = .05  # existing eligible-episode materiality audit


def intensity_margin(historical_mean):
    if not math.isfinite(historical_mean) or historical_mean <= 0:
        raise ValueError('positive historical intensity required')
    return INTENSITY_RELATIVE_MARGIN * historical_mean


def equivalence(difference, interval, margin):
    """Require the whole confidence interval within a pre-specified margin.

    Nonzero but practically equivalent effects remain visible as limitations.
    Intervals crossing a margin cannot certify equivalence. FAIL requires the
    interval to establish an effect outside the margin, not just outside zero.
    """
    low, high = map(float, interval)
    difference, margin = float(difference), float(margin)
    if not all(math.isfinite(x) for x in [difference, low, high, margin]):
        raise ValueError('finite effect, interval and margin required')
    if margin <= 0 or low > high or not low <= difference <= high:
        raise ValueError('positive margin and ordered interval containing effect required')
    if low >= -margin and high <= margin:
        return PASS if low <= 0 <= high else LIMITATION
    if low > margin or high < -margin:
        return FAIL
    return INCONCLUSIVE


def loop_inflation(difference, interval, materiality_margin=None):
    """One-sided evidence: positive point estimates alone never imply inflation.

    There is no pre-declared numeric loop-count margin. A supported positive
    increase without an independently justified margin requires a further
    materiality audit; it cannot silently pass or be called a material failure.
    """
    low, high = map(float, interval)
    if not all(math.isfinite(x) for x in [difference, low, high]) or low > high or not low <= difference <= high:
        raise ValueError('valid effect and interval required')
    if low <= 0:
        return PASS
    if materiality_margin is None:
        return INCONCLUSIVE
    return equivalence(difference, interval, materiality_margin)


def accepted(status):
    return status in {PASS, LIMITATION}
