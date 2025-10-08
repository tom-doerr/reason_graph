"""Utility module that wraps DSPy to extract reasons for a fact."""

from __future__ import annotations

import os
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import dspy
from pydantic import BaseModel, Field

# Ensure DSPy can write its cache inside the workspace, even when $HOME is read-only.
_CACHE_DIR = Path(__file__).resolve().parent / ".dspy_cache"
os.environ.setdefault("DSPY_CACHEDIR", str(_CACHE_DIR))
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # If the directory cannot be created we leave the env var in place and allow DSPy to handle it.
    pass

DEFAULT_MODEL_SLUG = "openrouter/deepseek/deepseek-chat-v3.1"
DEFAULT_MAX_REASONS = 4
DEFAULT_MAX_REASON_LENGTH = 80
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 320
def _default_thread_count() -> int:
    raw = os.environ.get("REASON_THREADS", "")
    try:
        value = int(raw) if raw else 16
    except ValueError:
        value = 16
    return max(1, value)


DEFAULT_THREAD_COUNT = _default_thread_count()
DEFAULT_MAX_STEPS = int(os.environ.get("REASON_MAX_STEPS", "32") or 32)

_REASON_PREFIX = re.compile(r"^\s*(?:[-*]\s*|\d+\s*[\.\-\)]\s*)")
_TRACE_PREVIEW = 120


class ReasonEdge(BaseModel):
    source: int
    target: int


class ReasonNode(BaseModel):
    id: int
    text: str
    parents: list[int] = Field(default_factory=list)
    children: list[int] = Field(default_factory=list)


class ReasonGraph(BaseModel):
    fact: str
    nodes: list[ReasonNode] = Field(default_factory=list)
    edges: list[ReasonEdge] = Field(default_factory=list)

    @property
    def reason_texts(self) -> list[str]:
        return [node.text for node in self.nodes[1:]] if len(self.nodes) > 1 else []


class _IterativeReasonSignature(dspy.Signature):
    """Generate the next reason in an iterative chain."""

    fact: str = dspy.InputField(
        desc="The factual statement the user provided. Keep it unchanged."
    )
    reason_chain: str = dspy.InputField(
        desc=(
            "A numbered list of concise reasons generated so far. "
            "Include the previous reasons in order, ending with the current focus."
        )
    )
    constraints: str = dspy.InputField(
        desc="Guidelines for how to phrase the next reason and when to stop."
    )
    child_reasons: list[str] = dspy.OutputField(
        desc="A list of new concise reasons that directly support the current focus node.",
        type=list[str],
    )


class ReasonGenerator(dspy.Module):
    """Iteratively expand a fact into a concise chain of supporting reasons."""

    def __init__(
        self,
        *,
        model_slug: str = DEFAULT_MODEL_SLUG,
        max_reasons: int = DEFAULT_MAX_REASONS,
        max_reason_length: int = DEFAULT_MAX_REASON_LENGTH,
        max_steps: int = DEFAULT_MAX_STEPS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_threads: int = DEFAULT_THREAD_COUNT,
        predictor: dspy.Module | None = None,
    ) -> None:
        super().__init__()
        if max_reasons < 1:
            raise ValueError("max_reasons must be at least 1.")
        if max_reason_length < 10:
            raise ValueError("max_reason_length must be at least 10 characters.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1.")

        self.model_slug = model_slug
        self.max_reasons = max_reasons
        self.max_reason_length = max_reason_length
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self._final_state: ReasonGraph | None = None

        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Please provide it to call the DeepSeek model."
            )

        dspy.settings.configure(
            lm=_lazy_lm(model_slug, temperature, max_tokens),
            num_threads=num_threads,
        )

        self._predict = predictor or dspy.ChainOfThought(_IterativeReasonSignature)

    def forward(self, fact: str) -> ReasonGraph:
        fact = fact.strip()
        if not fact:
            return ReasonGraph(fact="", nodes=[], edges=[])

        self._final_state = None
        for _ in self.iter_reasoning(fact):
            pass

        return self._final_state or ReasonGraph(fact=fact, nodes=[], edges=[])

    def iter_reasoning(
        self,
        fact: str,
        progress_callback: Callable[[int], None] | None = None,
        trace_callback: Callable[[str], None] | None = None,
    ) -> Iterable[ReasonGraph]:
        fact = fact.strip()
        if not fact:
            return

        self._final_state = None
        node_texts: list[str] = [fact]
        parents_map: dict[int, set[int]] = {0: set()}
        children_map: dict[int, set[int]] = {0: set()}
        edges: list[tuple[int, int]] = []
        edge_set: set[tuple[int, int]] = set()
        reason_to_index: dict[str, int] = {fact: 0}
        pending: deque[int] = deque([0])
        steps = 0
        latest_state: ReasonGraph | None = None

        while pending and len(node_texts) - 1 < self.max_reasons and steps < self.max_steps:
            focus_index = pending[0]
            children_map.setdefault(focus_index, set())
            graph_summary = _format_graph_summary(node_texts, edges, focus_index)
            focus_text = node_texts[focus_index]
            remaining_slots = self.max_reasons - (len(node_texts) - 1)
            constraints = _build_constraints(
                step=steps,
                max_length=self.max_reason_length,
                remaining=min(remaining_slots, self.max_steps - steps),
                focus_text=focus_text,
            )
            if progress_callback:
                progress_callback(1)
            if trace_callback:
                trace_callback(
                    _format_trace_request(
                        fact=fact,
                        focus_index=focus_index,
                        focus_text=focus_text,
                        node_count=len(node_texts) - 1,
                        remaining=remaining_slots,
                        max_length=self.max_reason_length,
                        model=self.model_slug,
                    )
                )
            prediction = self._predict(
                fact=fact,
                reason_chain=graph_summary,
                constraints=constraints,
            )

            child_texts = _normalise_child_reasons(
                getattr(prediction, "child_reasons", None),
                self.max_reason_length,
            )
            if trace_callback:
                trace_callback(
                    _format_trace_response(
                        focus_index=focus_index,
                        reasons=child_texts,
                    )
                )

            if not child_texts:
                pending.popleft()
                if progress_callback:
                    progress_callback(1 if pending else 0)
                continue

            added_any = False
            limit_reached = False
            for child_text in child_texts:
                if not child_text:
                    continue

                existing_index = reason_to_index.get(child_text)

                if existing_index is None:
                    if len(node_texts) - 1 >= self.max_reasons:
                        limit_reached = True
                        break
                    target_index = len(node_texts)
                    node_texts.append(child_text)
                    reason_to_index[child_text] = target_index
                    parents_map[target_index] = {focus_index}
                    children_map[target_index] = set()
                    children_map[focus_index].add(target_index)
                    edge_key = (focus_index, target_index)
                    edges.append(edge_key)
                    edge_set.add(edge_key)
                    pending.append(target_index)
                    added_any = True
                else:
                    target_index = existing_index
                    parents_map.setdefault(target_index, set()).add(focus_index)
                    children_map[focus_index].add(target_index)
                    edge_key = (focus_index, target_index)
                    if focus_index != target_index and edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append(edge_key)
                        added_any = True
            if added_any:
                latest_state = _assemble_graph(node_texts, parents_map, children_map, edges)
                yield latest_state

            pending.popleft()
            if progress_callback:
                progress_callback(1 if pending else 0)
            steps += 1
            if limit_reached:
                break

        if latest_state is None:
            latest_state = _assemble_graph(node_texts, parents_map, children_map, edges)

        self._final_state = latest_state
        if progress_callback:
            progress_callback(0)


@lru_cache(maxsize=4)
def _lazy_lm(model_slug: str, temperature: float, max_tokens: int) -> dspy.LM:
    """Create or reuse an LM client configured for OpenRouter DeepSeek."""
    return dspy.LM(
        model=model_slug,
        provider="openrouter",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _format_graph_summary(
    node_texts: Sequence[str],
    edges: Sequence[tuple[int, int]],
    focus_index: int,
) -> str:
    lines = []
    for idx, text in enumerate(node_texts):
        parents = [src for src, dst in edges if dst == idx]
        if parents:
            parent_labels = ", ".join(
                "Fact" if src == 0 else f"Reason {src}"
                for src in parents
            )
        else:
            parent_labels = "Fact"
        display = "Fact" if idx == 0 else f"Reason {idx}"
        prefix = "*" if focus_index == idx else "-"
        lines.append(
            f"{prefix} {display}: {text} (supported by {parent_labels})"
        )

    if focus_index == 0:
        focus_line = f"Current focus: Fact ({node_texts[0]})"
    else:
        focus_line = f"Current focus: Reason {focus_index} ({node_texts[focus_index]})"

    return focus_line + "\n" + "\n".join(lines)


def _build_constraints(
    *,
    step: int,
    max_length: int,
    remaining: int,
    focus_text: str,
) -> str:
    limit_clause = (
        f"Keep each reason under {max_length} characters, without numbering or bullet markers."
    )
    progression_clause = (
        "Make them specific, factual, and causally linked to the current focus and existing graph. "
        "Do not restate earlier reasons verbatim."
    )
    stopping_clause = (
        "If no additional concise reasons exist, return an empty list."
        if remaining <= 1
        else "Return an empty list when there are no further succinct reasons; otherwise list each remaining reason."
    )
    if step == 0:
        intro_clause = "Start by listing the strongest direct supports for the fact."
    else:
        intro_clause = "Provide additional reasons that support the current focus while respecting causal flow."

    focus_clause = f"We are expanding: {focus_text}."

    return " ".join([intro_clause, focus_clause, limit_clause, progression_clause, stopping_clause])


def _summarise_for_trace(text: str, *, limit: int = _TRACE_PREVIEW) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _format_trace_request(
    *,
    fact: str,
    focus_index: int,
    focus_text: str,
    node_count: int,
    remaining: int,
    max_length: int,
    model: str,
) -> str:
    return (
        "[ReasonGenerator] Request -> DSPy module\n"
        f"- Model: {model}\n"
        f"- Focus node: {focus_index} ({_summarise_for_trace(focus_text) or 'Fact'})\n"
        f"- Reason count: {node_count} collected; remaining budget: {remaining}\n"
        f"- Max reason length: {max_length}\n"
        f"- Fact: {_summarise_for_trace(fact) or '(empty)'}"
    )


def _format_trace_response(
    *,
    focus_index: int,
    reasons: Sequence[str],
) -> str:
    if reasons:
        preview = ", ".join(f"\"{_summarise_for_trace(reason)}\"" for reason in reasons)
    else:
        preview = "(none)"
    return (
        "[ReasonGenerator] Response <- DSPy module\n"
        f"- Focus node: {focus_index}\n"
        f"- Reasons: {preview}"
    )


def _assemble_graph(
    node_texts: Sequence[str],
    parents_map: dict[int, set[int]],
    children_map: dict[int, set[int]],
    edges: Sequence[tuple[int, int]],
) -> ReasonGraph:
    nodes: list[ReasonNode] = []
    for idx, text in enumerate(node_texts):
        parents = sorted(parents_map.get(idx, set()))
        children = sorted(children_map.get(idx, set()))
        nodes.append(ReasonNode(id=idx, text=text, parents=parents, children=children))

    edge_models = [ReasonEdge(source=src, target=dst) for src, dst in edges]
    return ReasonGraph(fact=node_texts[0], nodes=nodes, edges=edge_models)


def _clean_reason(text: str, max_length: int) -> str:
    candidate = _REASON_PREFIX.sub("", text or "").strip()
    if not candidate:
        return ""
    if len(candidate) > max_length:
        candidate = candidate[:max_length].rstrip(" ,;:-")
    return candidate


def _normalise_child_reasons(raw: object, max_length: int) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        text = str(raw).replace("\r", "")
        if "\n" in text:
            items = [part.strip() for part in text.split("\n") if part.strip()]
        else:
            items = [part.strip() for part in text.split(";") if part.strip()]
            if len(items) <= 1:
                items = [part.strip() for part in text.split(",") if part.strip()]
    children: list[str] = []
    for item in items:
        cleaned = _clean_reason(str(item), max_length)
        if cleaned:
            children.append(cleaned)
    return children


__all__ = [
    "ReasonGenerator",
    "ReasonGraph",
    "ReasonNode",
    "ReasonEdge",
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_MAX_REASON_LENGTH",
    "DEFAULT_THREAD_COUNT",
    "DEFAULT_MAX_STEPS",
]
