"""Tests for stability-aware (lower confidence bound) score aggregation.

These exercise the wiring between ``score_stability`` and the existing
``score_statistics.calculate_aggregated_statistics`` call site, verifying that
the variance-penalized lower confidence bound proposed in arXiv:2606.24381 is
populated on the aggregated statistics and behaves as expected for selection.
"""

import math

import pytest

from opik.evaluation import score_stability, score_statistics, test_case, test_result
from opik.evaluation.metrics import score_result


def _make_test_result(metric_values, trial_id):
    """Build a TestResult carrying one ScoreResult per (name, value) pair."""
    return test_result.TestResult(
        test_case=test_case.TestCase(
            trace_id=f"trace-{trial_id}",
            dataset_item_id=f"item-{trial_id}",
            task_output={"output": "x"},
        ),
        score_results=[
            score_result.ScoreResult(name=name, value=value)
            for name, value in metric_values.items()
        ],
        trial_id=trial_id,
    )


def test_lower_confidence_bound__single_value__returns_mean():
    # No variance estimate available -> no penalty applied.
    assert score_stability.lower_confidence_bound(mean=0.8, std=None, count=1) == 0.8


def test_lower_confidence_bound__penalizes_variance():
    # mean - z * std / sqrt(n), with default z = 1.0
    lcb = score_stability.lower_confidence_bound(mean=0.8, std=0.2, count=4)
    assert lcb == pytest.approx(0.8 - 1.0 * 0.2 / math.sqrt(4))
    assert lcb < 0.8


def test_lower_confidence_bound__larger_z_is_more_conservative():
    base = score_stability.lower_confidence_bound(mean=0.8, std=0.2, count=4, z=1.0)
    strict = score_stability.lower_confidence_bound(mean=0.8, std=0.2, count=4, z=1.645)
    assert strict < base


def test_calculate_aggregated_statistics__populates_lower_confidence_bound():
    # Two scores share the same mean (0.5) but different spreads.
    stable_values = [0.5, 0.5, 0.5, 0.5]
    noisy_values = [0.0, 1.0, 0.0, 1.0]

    results = [
        _make_test_result({"stable_metric": s, "noisy_metric": n}, trial_id=i)
        for i, (s, n) in enumerate(zip(stable_values, noisy_values))
    ]

    aggregated = score_statistics.calculate_aggregated_statistics(results)

    stable = aggregated["stable_metric"]
    noisy = aggregated["noisy_metric"]

    # Same mean...
    assert stable.mean == pytest.approx(0.5)
    assert noisy.mean == pytest.approx(0.5)

    # ...but the lower confidence bound separates them: the stable metric (no
    # variance) keeps its mean while the noisy one is penalized below it.
    assert stable.lower_confidence_bound == pytest.approx(0.5)
    assert noisy.lower_confidence_bound < noisy.mean

    # Stability-aware selection therefore prefers the stable metric even though
    # a mean-only ranking would call it a tie.
    assert stable.lower_confidence_bound > noisy.lower_confidence_bound

    # The aggregated value matches the standalone helper for the same inputs.
    assert noisy.lower_confidence_bound == pytest.approx(
        score_stability.lower_confidence_bound(
            mean=noisy.mean, std=noisy.std, count=len(noisy.values)
        )
    )
