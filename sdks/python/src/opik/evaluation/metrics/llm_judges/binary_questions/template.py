from typing import List, Optional

from opik.evaluation.models import base_model

_SYSTEM_PROMPT = """You are an expert evaluator. You will be given a TASK description, the OUTPUT produced for that task, and a list of atomic yes/no evaluation QUESTIONS.

Answer EACH question independently with a strict binary verdict:
- "yes" if the OUTPUT satisfies the question.
- "no" if it does not.

Guidelines:
1. Judge every question on its own merits. Do not collapse the questions into a single holistic judgment.
2. Do not give partial credit. A question is either satisfied ("yes") or not ("no").
3. Ground each verdict in concrete evidence from the OUTPUT and give a short reason.
4. Answer the questions in the same order they are given, one verdict per question.

It is crucial that you provide your answer in the following JSON format only:
{{
    "verdicts": [
        {{"question": "<the question text>", "answer": "yes" or "no", "reason": "<short justification>"}}
    ]
}}
Include exactly one entry per question. Output must be JSON format only."""


def _format_questions(questions: List[str]) -> str:
    return "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, 1)
    )


def build_messages(
    task_introduction: str,
    questions: List[str],
    output: str,
    input: Optional[str] = None,
    context: Optional[List[str]] = None,
) -> List[base_model.ConversationDict]:
    """Build the [system, user] message pair for a binary-question judgment.

    The static instructions and JSON output spec live in the system message so
    providers can cache the prefix across calls. The per-call ``task_introduction``,
    ``questions``, ``input``/``context`` and ``output`` go in the user message.
    """
    sections = [f"TASK:\n{task_introduction}"]
    if input is not None:
        sections.append(f"INPUT:\n{input}")
    if context is not None:
        sections.append("CONTEXT:\n" + "\n".join(context))
    sections.append(f"OUTPUT:\n{output}")
    sections.append(f"QUESTIONS:\n{_format_questions(questions)}")

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
