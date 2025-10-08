import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reason_engine  # noqa: E402
import dspy  # noqa: E402


@dataclass
class StubPrediction:
    child_reasons: list[str]


class FakePredictor(dspy.Module):
    def __init__(self, predictions):
        super().__init__()
        self._predictions = list(predictions)
        self.calls = []

    def __call__(self, **kwargs):
        idx = len(self.calls)
        self.calls.append(kwargs)
        try:
            return self._predictions[idx]
        except IndexError:
            return self._predictions[-1]


class DummyLM:
    """Minimal stand-in to satisfy dspy.settings.configure during tests."""

    def __init__(self, model):
        self.model = model


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DSPY_CACHEDIR", str(reason_engine._CACHE_DIR))


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch):
    calls = []

    def fake_configure(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(dspy.settings, "configure", fake_configure)
    return calls


def test_reason_generator_builds_iterative_chain(monkeypatch, _stub_settings):
    predictor = FakePredictor(
        [
            StubPrediction(
                child_reasons=[
                    "It receives solar energy.",
                    "Solar panels convert sunlight to electricity.",
                ]
            ),
            StubPrediction(child_reasons=[]),
            StubPrediction(child_reasons=[]),
        ]
    )

    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    generator = reason_engine.ReasonGenerator(
        predictor=predictor,
        max_reasons=5,
        max_reason_length=80,
        num_threads=8,
    )
    result = generator("The house uses solar power.")

    assert result.fact == "The house uses solar power."
    assert [node.text for node in result.nodes] == [
        "The house uses solar power.",
        "It receives solar energy.",
        "Solar panels convert sunlight to electricity.",
    ]
    edge_set = {(edge.source, edge.target) for edge in result.edges}
    assert edge_set >= {(0, 1), (0, 2)}

    predictor_stream = FakePredictor(
        [
            StubPrediction(
                child_reasons=[
                    "It receives solar energy.",
                    "Solar panels convert sunlight to electricity.",
                ]
            ),
            StubPrediction(child_reasons=[]),
            StubPrediction(child_reasons=[]),
        ]
    )
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )
    streaming_generator = reason_engine.ReasonGenerator(
        predictor=predictor_stream, max_reasons=5, max_reason_length=80
    )
    snapshots = list(streaming_generator.iter_reasoning("The house uses solar power."))
    assert len(snapshots) == 1
    assert [node.text for node in snapshots[0].nodes] == [
        "The house uses solar power.",
        "It receives solar energy.",
        "Solar panels convert sunlight to electricity.",
    ]

    assert "Current focus" in predictor.calls[0]["reason_chain"]
    assert "Fact:" in predictor.calls[0]["reason_chain"]


def test_reason_generator_truncates_long_reason(monkeypatch, _stub_settings):
    long_reason = "A" * 200
    predictor = FakePredictor([StubPrediction(child_reasons=[long_reason])])
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    generator = reason_engine.ReasonGenerator(
        predictor=predictor, max_reasons=3, max_reason_length=25
    )
    result = generator("Short fact.")

    assert len(result.nodes) == 2
    assert len(result.nodes[1].text) <= 25
    assert result.nodes[1].text.startswith("A")
    edge_set = {(edge.source, edge.target) for edge in result.edges}
    assert edge_set == {(0, 1)}


def test_reason_generator_requires_api_key(monkeypatch, _stub_settings):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    with pytest.raises(RuntimeError):
        reason_engine.ReasonGenerator(
            predictor=FakePredictor([StubPrediction(child_reasons=["reason"])])
        )
    assert not _stub_settings


def test_reason_generator_rejects_invalid_threads(monkeypatch, _stub_settings):
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )
    with pytest.raises(ValueError):
        reason_engine.ReasonGenerator(
            predictor=FakePredictor([StubPrediction(child_reasons=["reason"])]),
            num_threads=0,
        )


def test_reason_generator_blank_fact(monkeypatch, _stub_settings):
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )
    generator = reason_engine.ReasonGenerator(
        predictor=FakePredictor([StubPrediction(child_reasons=["unused"])])
    )
    result = generator("   ")
    assert result.fact == ""
    assert result.nodes == []
    assert result.edges == []
    assert generator.num_threads == reason_engine.DEFAULT_THREAD_COUNT


def test_reason_generator_deduplicates_reasons(monkeypatch, _stub_settings):
    predictor = FakePredictor(
        [
            StubPrediction(child_reasons=["Reason A", "Reason B"]),
            StubPrediction(child_reasons=["Reason B"]),
            StubPrediction(child_reasons=[]),
        ]
    )
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    generator = reason_engine.ReasonGenerator(
        predictor=predictor,
        max_reasons=5,
        max_reason_length=80,
    )

    result = generator("Duplicated fact")

    assert [node.text for node in result.nodes] == ["Duplicated fact", "Reason A", "Reason B"]
    edge_set = {(edge.source, edge.target) for edge in result.edges}
    assert edge_set >= {(0, 1), (0, 2), (1, 2)}
    assert all(node.text in {"Duplicated fact", "Reason A", "Reason B"} for node in result.nodes)

    streaming_generator = reason_engine.ReasonGenerator(
        predictor=FakePredictor(
            [
                StubPrediction(child_reasons=["Reason A", "Reason B"]),
                StubPrediction(child_reasons=["Reason B"]),
                StubPrediction(child_reasons=[]),
            ]
        ),
        max_reasons=5,
        max_reason_length=80,
    )

    snapshots = list(streaming_generator.iter_reasoning("Duplicated fact"))
    assert [node.text for node in snapshots[-1].nodes] == ["Duplicated fact", "Reason A", "Reason B"]
    assert {(edge.source, edge.target) for edge in snapshots[-1].edges} >= {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert streaming_generator._final_state is not None
    final_edge_set = {
        (edge.source, edge.target)
        for edge in streaming_generator._final_state.edges
    }
    assert final_edge_set >= {(0, 1), (0, 2), (1, 2)}


def test_iter_reasoning_reports_progress(monkeypatch, _stub_settings):
    predictor = FakePredictor(
        [
            StubPrediction(child_reasons=["Reason A"]),
            StubPrediction(child_reasons=[]),
        ]
    )
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    generator = reason_engine.ReasonGenerator(
        predictor=predictor,
        max_reasons=3,
        max_reason_length=80,
    )

    events: list[int] = []
    list(generator.iter_reasoning("Progress fact", progress_callback=events.append))

    assert events
    assert events[0] == 1
    assert events[-1] == 0
    assert all(event in (0, 1) for event in events)
    assert events.count(1) >= 2


def test_iter_reasoning_emits_trace(monkeypatch, _stub_settings):
    predictor = FakePredictor(
        [
            StubPrediction(child_reasons=["Reason A"]),
            StubPrediction(child_reasons=[]),
        ]
    )
    monkeypatch.setattr(
        reason_engine, "_lazy_lm", lambda *args, **kwargs: DummyLM("fake-model")
    )

    generator = reason_engine.ReasonGenerator(
        predictor=predictor,
        max_reasons=3,
        max_reason_length=80,
    )

    traces: list[str] = []
    list(
        generator.iter_reasoning(
            "Traceable fact",
            progress_callback=None,
            trace_callback=traces.append,
        )
    )

    assert any("Request" in entry for entry in traces)
    assert any("Response" in entry for entry in traces)

    request_entry = next(entry for entry in traces if entry.startswith("[ReasonGenerator] Request"))
    assert "Model: openrouter/deepseek/deepseek-chat-v3.1" in request_entry
    assert "Focus node: 0" in request_entry
    assert "Reason count: 0" in request_entry
    assert "Fact: Traceable fact" in request_entry

    response_entry = next(entry for entry in traces if entry.startswith("[ReasonGenerator] Response"))
    assert 'Reasons: "Reason A"' in response_entry


def test_summarise_for_trace_truncates(monkeypatch, _stub_settings):
    long_text = "A" * 200
    summary = reason_engine._summarise_for_trace(long_text, limit=30)
    assert summary.endswith("...")
    assert len(summary) <= 30
