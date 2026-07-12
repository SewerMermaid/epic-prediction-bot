"""Pure accuracy/scoring logic for MiniBench analysis.

Everything here is a pure function of an already-parsed forecast + resolution —
no network, no Metaculus SDK — so it is unit-testable in isolation. The Metaculus
API plumbing that *produces* these inputs lives in ``client.py``.

Terminology:
- ``forecast`` shapes by type:
    binary          -> float p_yes in [0, 1]
    multiple_choice -> list[float] probabilities aligned to the option order
    numeric         -> a ``NumericCdf`` (the stored 201-point CDF + its x-axis)
- ``resolution`` shapes by type:
    binary          -> bool (True == resolved YES)
    multiple_choice -> int index of the resolved option
    numeric         -> float resolved value (in question units)

Annulled / ambiguous / out-of-bounds resolutions are represented as ``None`` at
the call site and MUST be excluded from denominators by the caller — they score
as null on Metaculus and should count neither for nor against a bot.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Numeric CDF handling
# ---------------------------------------------------------------------------


def generate_cdf_locations(range_min: float, range_max: float, zero_point: float | None, cdf_size: int) -> list[float]:
    """X-axis (question-unit values) for the ``cdf_size``-point Metaculus CDF.

    Mirrors ``forecasting_tools.data_models.numeric_report.generate_cdf_locations``
    so our percentile inversion lands on the exact same grid Metaculus scores on.
    Linear when ``zero_point`` is None, log-spaced otherwise.
    """
    if zero_point is None:

        def scale(x: float) -> float:
            return range_min + (range_max - range_min) * x

    else:
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)

        def scale(x: float) -> float:
            return range_min + (range_max - range_min) * (deriv_ratio**x - 1) / (deriv_ratio - 1)

    return [float(scale(x)) for x in np.linspace(0, 1, cdf_size)]


@dataclass(frozen=True)
class NumericCdf:
    """A stored numeric forecast: the cumulative distribution over ``x_axis``.

    ``cdf[i]`` is the forecast probability that the outcome is <= ``x_axis[i]``.
    Both lists have length ``cdf_size`` and ``cdf`` is non-decreasing from ~0 to ~1.
    """

    x_axis: list[float]
    cdf: list[float]

    @classmethod
    def from_forecast_values(
        cls,
        forecast_values: list[float],
        *,
        range_min: float,
        range_max: float,
        zero_point: float | None,
    ) -> "NumericCdf":
        x_axis = generate_cdf_locations(range_min, range_max, zero_point, len(forecast_values))
        return cls(x_axis=x_axis, cdf=list(forecast_values))

    def value_at_percentile(self, q: float) -> float:
        """Inverse-CDF: the question-unit value at cumulative probability ``q``.

        Linear interpolation between the two CDF grid points that bracket ``q``.
        Clamps to the axis ends when ``q`` falls in a flat tail.
        """
        cdf = self.cdf
        x = self.x_axis
        if q <= cdf[0]:
            return x[0]
        if q >= cdf[-1]:
            return x[-1]
        # cdf is non-decreasing; find first index where cdf[i] >= q.
        i = bisect.bisect_left(cdf, q)
        lo_c, hi_c = cdf[i - 1], cdf[i]
        lo_x, hi_x = x[i - 1], x[i]
        if hi_c == lo_c:  # flat segment; take the left edge
            return float(lo_x)
        frac = (q - lo_c) / (hi_c - lo_c)
        return float(lo_x + frac * (hi_x - lo_x))

    def mass_in_bin_containing(self, value: float) -> float | None:
        """Forecast probability mass in the CDF interval that contains ``value``.

        Returns None when ``value`` is outside the axis (an out-of-bounds
        resolution), which the caller should treat as non-scorable here.
        """
        x = self.x_axis
        if value < x[0] or value > x[-1]:
            return None
        j = bisect.bisect_right(x, value) - 1
        j = min(max(j, 0), len(x) - 2)
        return self.cdf[j + 1] - self.cdf[j]


# ---------------------------------------------------------------------------
# Tier 1 — "beat chance" (sign of the Metaculus baseline score). All bots.
# ---------------------------------------------------------------------------


def beat_chance_binary(p_yes: float, resolved_yes: bool) -> bool:
    """Binary baseline > 0 iff more than 50% was placed on the resolved side."""
    p_correct = p_yes if resolved_yes else 1.0 - p_yes
    return bool(p_correct > 0.5)


def beat_chance_multiple_choice(probs: list[float], resolved_index: int) -> bool:
    """MC baseline > 0 iff the resolved option got more than the uniform 1/N."""
    n = len(probs)
    if n == 0 or not (0 <= resolved_index < n):
        raise ValueError(f"resolved_index {resolved_index} out of range for {n} options")
    return bool(probs[resolved_index] > 1.0 / n)


def beat_chance_numeric(cdf: NumericCdf, resolved_value: float) -> bool | None:
    """Numeric baseline > 0 iff forecast density at the outcome beats uniform.

    On the discretized 201-point grid Metaculus scores against, the uniform
    reference mass per interval is ``1/(cdf_size-1)``. Returns None for
    out-of-bounds resolutions (not scorable on the in-range grid).
    """
    mass = cdf.mass_in_bin_containing(resolved_value)
    if mass is None:
        return None
    uniform_mass = 1.0 / (len(cdf.x_axis) - 1)
    # Epsilon so a perfectly-uniform forecast (mass == uniform_mass up to float
    # noise on the grid) reads as "did not beat chance", matching the strict
    # >0.5 / >1/N boundary used for binary and MC.
    return bool(mass > uniform_mass * (1.0 + 1e-9))


# ---------------------------------------------------------------------------
# Tier 2 — intuitive breakdown (my bot only).
# ---------------------------------------------------------------------------


def directional_binary(p_yes: float, resolved_yes: bool) -> bool:
    """Same test as Tier 1 for binary; named for the Tier-2 report column."""
    return beat_chance_binary(p_yes, resolved_yes)


def argmax_multiple_choice(probs: list[float], resolved_index: int) -> bool:
    """MC correct iff the highest-probability option is the one that resolved.

    Ties (multiple options share the max) count as correct only if the resolved
    option is among the tied maxima.
    """
    if not probs:
        raise ValueError("empty probability vector")
    top = max(probs)
    return bool(probs[resolved_index] == top)


def within_iqr_numeric(cdf: NumericCdf, resolved_value: float) -> bool:
    """Numeric 'accurate' (my-bot Tier 2) iff the outcome fell within P25-P75."""
    p25 = cdf.value_at_percentile(0.25)
    p75 = cdf.value_at_percentile(0.75)
    lo, hi = (p25, p75) if p25 <= p75 else (p75, p25)
    return bool(lo <= resolved_value <= hi)
