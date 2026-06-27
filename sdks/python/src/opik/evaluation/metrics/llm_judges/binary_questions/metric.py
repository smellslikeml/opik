from typing import Any, List, Optional, Union
import pydantic

from opik.evaluation.models import base_model, models_factory
from opik.evaluation.metrics import score_result, base_metric

from . import template, parser


class _BinaryVerdict(pydantic.BaseModel):
    question: str
    answer: str
    reason: str


class BinaryQuestionsResponseFormat(pydantic.BaseModel):
    verdicts: List[_BinaryVerdict]


class BinaryQuestions(base_metric.BaseMetric):
    """Evaluate an output by decomposing the criteria into atomic yes/no questions.

    Instead of asking the judge for a single opaque score, ``BinaryQuestions``
    asks a list of fine-grained binary questions, has the judge answer each one
    independently, and aggregates the verdicts into a calibrated score. The
    per-question verdicts are returned in ``ScoreResult.metadata`` so the result
    is interpretable and directly actionable for prompt iteration.

    Adapted from "Ask, Don't Judge: Binary Questions for Interpretable LLM
    Evaluation and Self-Improvement" (BINEVAL, arXiv:2606.27226).

    Args:
        questions: The atomic yes/no questions to evaluate the output against.
            Each question should be answerable with a clear "yes" (satisfied) or
            "no" (not satisfied).
        task_introduction: Short description of the task being evaluated, shown to
            the judge for context. Defaults to a generic quality instruction.
        model: The LLM to use for evaluation. Can be a string (model name) or an
            ``opik.evaluation.models.OpikBaseModel`` subclass instance.
            ``opik.evaluation.models.LiteLLMChatModel`` is used by default.
        name: The name of the metric.
        track: Whether to track the metric. Defaults to True.
        project_name: Optional project name to track the metric in for the cases
            when there is no parent span/trace to inherit project name from.
        seed: Optional seed value for reproducible model generation.
        temperature: Optional temperature value for model generation.

    Example:
        >>> from opik.evaluation.metrics import BinaryQuestions
        >>> metric = BinaryQuestions(
        ...     questions=[
        ...         "Is the answer factually correct?",
        ...         "Does the answer directly address the question?",
        ...     ],
        ... )
        >>> result = metric.score(
        ...     input="What is the capital of France?",
        ...     output="The capital of France is Paris.",
        ... )  # doctest: +SKIP
        >>> result.value  # doctest: +SKIP
        1.0
        >>> result.metadata["verdicts"]  # doctest: +SKIP
        [{'question': 'Is the answer factually correct?', 'answer': 'yes', ...}, ...]
    """

    def __init__(
        self,
        questions: List[str],
        task_introduction: str = "Evaluate the quality of the OUTPUT for the given task.",
        model: Optional[Union[str, base_model.OpikBaseModel]] = None,
        name: str = "binary_questions_metric",
        track: bool = True,
        project_name: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__(name=name, track=track, project_name=project_name)
        if not questions:
            raise ValueError("`questions` must contain at least one binary question")
        self.questions = list(questions)
        self.task_introduction = task_introduction
        self._seed = seed
        self._init_model(model, temperature=temperature)

    def _init_model(
        self,
        model: Optional[Union[str, base_model.OpikBaseModel]],
        temperature: Optional[float],
    ) -> None:
        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            model_kwargs = {}
            if temperature is not None:
                model_kwargs["temperature"] = temperature
            if self._seed is not None:
                model_kwargs["seed"] = self._seed

            self._model = models_factory.get(
                model_name=model, track=self.track, **model_kwargs
            )

    def score(
        self,
        output: str,
        input: Optional[str] = None,
        context: Optional[List[str]] = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        """Score ``output`` against the configured binary questions.

        Args:
            output: The LLM's output to evaluate.
            input: Optional original input/question, shown to the judge for context.
            context: Optional list of grounding context strings (e.g. for factual
                consistency questions).
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: Value is the fraction of questions answered
            "yes" (0.0-1.0); ``metadata["verdicts"]`` holds the per-question feedback.
        """
        messages = template.build_messages(
            task_introduction=self.task_introduction,
            questions=self.questions,
            output=output,
            input=input,
            context=context,
        )
        message = self._model.generate_chat_completion(
            messages=messages, response_format=BinaryQuestionsResponseFormat
        )

        return parser.parse_model_output(
            content=message["content"], name=self.name, questions=self.questions
        )

    async def ascore(
        self,
        output: str,
        input: Optional[str] = None,
        context: Optional[List[str]] = None,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        """Async variant of :meth:`score`."""
        messages = template.build_messages(
            task_introduction=self.task_introduction,
            questions=self.questions,
            output=output,
            input=input,
            context=context,
        )
        message = await self._model.agenerate_chat_completion(
            messages=messages, response_format=BinaryQuestionsResponseFormat
        )

        return parser.parse_model_output(
            content=message["content"], name=self.name, questions=self.questions
        )
