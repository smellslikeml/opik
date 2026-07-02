import logging
from typing import Any, Dict, List, Optional

import opik.exceptions as exceptions
from opik.evaluation.metrics import score_result
from opik.evaluation.metrics.llm_judges import parsing_helpers

LOGGER = logging.getLogger(__name__)

_AFFIRMATIVE = {"yes", "true", "1", "y"}


def _match_verdict(
    question: str,
    index: int,
    raw_verdicts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the model verdict for a configured question.

    Prefer matching on the echoed question text; fall back to positional order
    when the model paraphrases or omits the question field.
    """
    normalized = question.strip().lower()
    for entry in raw_verdicts:
        if str(entry.get("question", "")).strip().lower() == normalized:
            return entry
    if index < len(raw_verdicts):
        return raw_verdicts[index]
    return None


def parse_model_output(
    content: str,
    name: str,
    questions: List[str],
) -> score_result.ScoreResult:
    """Aggregate per-question binary verdicts into an interpretable ScoreResult.

    The numeric value is the fraction of questions answered "yes" (higher is
    better). The per-question verdicts are preserved in ``metadata`` so callers
    get transparent, multi-dimensional feedback rather than an opaque score.
    """
    try:
        dict_content = parsing_helpers.extract_json_content_or_raise(content)
        raw_verdicts = dict_content["verdicts"]
        if not isinstance(raw_verdicts, list):
            raise ValueError("'verdicts' must be a list")

        verdicts: List[Dict[str, Any]] = []
        satisfied = 0
        for index, question in enumerate(questions):
            entry = _match_verdict(question, index, raw_verdicts)
            if entry is None:
                raise exceptions.MetricComputationError(
                    f"No verdict returned for question: {question!r}"
                )
            answer = str(entry.get("answer", "")).strip().lower()
            passed = answer in _AFFIRMATIVE
            satisfied += int(passed)
            verdicts.append(
                {
                    "question": question,
                    "answer": "yes" if passed else "no",
                    "reason": str(entry.get("reason", "")),
                }
            )

        total = len(questions)
        value = satisfied / total

        return score_result.ScoreResult(
            name=name,
            value=value,
            reason=f"{satisfied}/{total} binary evaluation questions satisfied.",
            metadata={
                "verdicts": verdicts,
                "satisfied": satisfied,
                "total": total,
            },
        )
    except exceptions.MetricComputationError:
        raise
    except Exception as e:
        LOGGER.error(f"Failed to parse model output: {e}", exc_info=True)
        raise exceptions.MetricComputationError(
            "Binary-question evaluation failed: could not parse model output."
        )
