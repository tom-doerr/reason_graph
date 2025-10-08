import sys
from pathlib import Path
from typing import Iterable

import pytest
from textual.widgets import Input, Log, Tree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reason_engine  # noqa: E402
from fact_reason_app import FactReasonApp  # noqa: E402


class _StubPrediction:
    def __init__(self, child_reasons: Iterable[str]):
        self.child_reasons = list(child_reasons)


class _FakePredictor:
    def __init__(self):
        self._responses = [
            _StubPrediction(["Reason one", "Reason two"]),
            _StubPrediction(["Downstream support"]),
            _StubPrediction([]),
            _StubPrediction([]),
        ]
        self.calls = 0

    def __call__(self, **kwargs):
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


@pytest.mark.asyncio
async def test_app_integration_with_reason_generator(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(reason_engine, "_lazy_lm", lambda *_, **__: None)
    settings_cls = type(reason_engine.dspy.settings)
    monkeypatch.setattr(settings_cls, "_ensure_configure_allowed", lambda self: None, raising=False)
    monkeypatch.setattr(settings_cls, "configure", lambda self, **__: None, raising=False)
    monkeypatch.setenv("DSPY_CACHEDIR", "/tmp/dspy-cache")

    generator = reason_engine.ReasonGenerator(
        predictor=_FakePredictor(),
        max_reasons=5,
        max_reason_length=80,
        num_threads=4,
        max_steps=8,
    )
    snapshots = list(generator.iter_reasoning("Testing integration."))
    assert snapshots, "Expected generator to yield snapshots"
    final_graph = snapshots[-1]

    app = FactReasonApp()
    async with app.run_test() as pilot:
        await pilot.pause(0)

        app.busy = True
        app.thread_count = generator.num_threads
        app._start_generation_ui(
            "Testing integration.",
            generator.max_reasons,
            generator.num_threads,
            generator.max_steps,
        )
        app._set_active_workers(1)
        await pilot.pause(0)

        for snapshot in snapshots:
            app._handle_intermediate_result(snapshot)
            await pilot.pause(0)

        app._handle_generation_complete(final_graph)
        await pilot.pause(0)

        tree = app.query_one("#reason-tree", Tree)
        children = tree.root.children
        assert len(children) == len(final_graph.nodes) - 1
        labels = {child.label.plain for child in children}
        assert any("Reason one" in label for label in labels)
        assert any("Reason two" in label for label in labels)
        assert any("Downstream support" in label for label in labels)

        status_text = str(app.query_one("#status").render())
        assert status_text.startswith("Done.")

        thread_status = str(app.query_one("#thread-status").render())
        assert thread_status == "Threads active: 0/4"

        log_widget = app.query_one("#detail-log", Log)
        assert log_widget.lines
        activity_log = app.query_one("#activity-log", Log)
        activity_lines = list(activity_log.lines)
        assert any("Starting generation" in line for line in activity_lines)
        assert any("Added" in line for line in activity_lines)
        assert any("Completed generation" in line for line in activity_lines)
