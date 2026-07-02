"""Tests for the BinaryQuestions judge and its wiring into the public metrics API.

The metric is imported through the existing ``opik.evaluation.metrics`` package
(not the new module directly) to exercise the ``__init__`` export edit that wires
the BINEVAL-style judge into the engine's public surface.
"""

import json
from unittest.mock import Mock

import pytest

from opik.evaluation.metrics import BinaryQuestions
from opik.evaluation.metrics import score_result
from opik.evaluation.metrics.llm_judges.binary_questions import parser
from opik.evaluation.models import base_model


def _mock_model(verdicts) -> Mock:
    """A model stub that returns a structured binary-question response."""
    model = Mock(spec=base_model.OpikBaseModel)
    content = json.dumps({"verdicts": verdicts})
    model.generate_chat_completion.return_value = {
        "role": "assistant",
        "content": content,
    }
    return model


def test_binary_questions_is_exported_from_public_metrics_api():
    # The public package import is the integration point the evaluation engine uses.
    from opik.evaluation import metrics

    assert "BinaryQuestions" in metrics.__all__
    assert metrics.BinaryQuestions is BinaryQuestions


def test_binary_questions_aggregates_verdicts_into_score_and_metadata():
    questions = [
        "Is the answer factually correct?",
        "Does the answer directly address the question?",
        "Is the answer free of unsupported claims?",
    ]
    model = _mock_model(
        [
            {"question": questions[0], "answer": "yes", "reason": "Matches the facts."},
            {"question": questions[1], "answer": "no", "reason": "Off topic."},
            {"question": questions[2], "answer": "yes", "reason": "No extra claims."},
        ]
    )
    metric = BinaryQuestions(questions=questions, model=model, track=False)

    result = metric.score(
        output="The capital of France is Paris.",
        input="What is the capital of France?",
    )

    assert isinstance(result, score_result.ScoreResult)
    # 2 of 3 questions satisfied -> aggregated, calibrated score.
    assert result.value == pytest.approx(2 / 3)
    # Per-question verdicts are preserved for interpretable, actionable feedback.
    assert result.metadata["satisfied"] == 2
    assert result.metadata["total"] == 3
    answers = {v["question"]: v["answer"] for v in result.metadata["verdicts"]}
    assert answers[questions[1]] == "no"
    model.generate_chat_completion.assert_called_once()


def test_binary_questions_requires_at_least_one_question():
    with pytest.raises(ValueError):
        BinaryQuestions(questions=[], track=False)


def test_parser_matches_verdicts_by_question_text_when_reordered():
    questions = ["Is it concise?", "Is it polite?"]
    # Model returns verdicts in a different order than configured.
    content = json.dumps(
        {
            "verdicts": [
                {"question": "Is it polite?", "answer": "yes", "reason": "Friendly."},
                {"question": "Is it concise?", "answer": "no", "reason": "Too long."},
            ]
        }
    )

    result = parser.parse_model_output(
        content=content, name="binary_questions_metric", questions=questions
    )

    by_question = {v["question"]: v["answer"] for v in result.metadata["verdicts"]}
    assert by_question["Is it concise?"] == "no"
    assert by_question["Is it polite?"] == "yes"
    assert result.value == pytest.approx(0.5)
