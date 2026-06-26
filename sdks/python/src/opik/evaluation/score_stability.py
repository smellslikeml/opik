"""Stability-aware scoring helpers for prompt/experiment selection.

When several prompts or experiment variants are compared, ranking them by the
mean score alone is fragile: the paper "On the Stability of Prompt Ranking in
Large Language Model Evaluation" (arXiv:2606.24381) shows that the identity of
the top-ranked prompt frequently changes under minor evaluation variation
(random seeds, limited subsets), even when overall rank correlation stays high.

The paper proposes a simple, robust alternative to "pick the highest mean":
select on a lower confidence bound (LCB) that subtracts a variance penalty from
the mean. A variant with a slightly lower mean but much tighter spread can be
the safer pick, and the LCB makes that trade-off explicit while staying
competitive with the mean in already-stable regimes.

This module computes that single number from the mean/std that the evaluation
aggregation already produces, so callers can rank by ``lower_confidence_bound``
instead of (or alongside) ``mean``.
"""

import math
from typing import Optional

# One standard error below the mean. z=1.0 is a conservative, easy-to-reason
# default (roughly a one-sided ~84% bound); callers wanting a tighter guarantee
# can pass e.g. 1.645 for a one-sided 95% bound.
DEFAULT_Z = 1.0


def lower_confidence_bound(
    mean: float,
    std: Optional[float],
    count: int,
    z: float = DEFAULT_Z,
) -> float:
    """Compute a variance-penalized lower confidence bound for a score.

    The bound is ``mean - z * standard_error``, where the standard error is
    ``std / sqrt(count)``. Ranking candidates by this value rather than by the
    raw mean penalizes high-variance candidates and yields more stable
    selection decisions under evaluation noise (arXiv:2606.24381).

    Args:
        mean: Mean score across trials.
        std: Sample standard deviation of the scores, or ``None`` when fewer
            than two values are available (no variance estimate).
        count: Number of score values the statistics were computed from.
        z: Multiplier controlling how many standard errors to subtract. Larger
            values are more conservative. Defaults to ``DEFAULT_Z`` (1.0).

    Returns:
        The lower confidence bound. When variance cannot be estimated
        (``std`` is ``None`` or ``count`` < 2) the mean is returned unchanged,
        since there is no evidence on which to apply a penalty.
    """
    if std is None or count < 2 or not math.isfinite(std):
        return mean

    standard_error = std / math.sqrt(count)
    return mean - z * standard_error
