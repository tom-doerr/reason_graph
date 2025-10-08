import sys
from pathlib import Path

import pytest

from textual.widgets import Input, Log, Tree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_reason_app import FactReasonApp
from reason_engine import ReasonEdge, ReasonGraph, ReasonNode


@pytest.mark.asyncio
async def test_full_app_flow(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured_inits: list[tuple[int, int, int]] = []

    class StubReasonGenerator:
        def __init__(
            self,
            *,
            max_reasons: int,
            max_reason_length: int,
            num_threads: int,
            **kwargs,
        ):
            captured_inits.append((max_reasons, max_reason_length, num_threads))

        def iter_reasoning(self, fact: str, progress_callback=None, trace_callback=None):
            if trace_callback:
                trace_callback(f"[StubReasonGenerator] Request fact='{fact}'")
            if progress_callback:
                progress_callback(1)
            yield ReasonGraph(
                fact=fact,
                nodes=[
                    ReasonNode(id=0, text=fact, children=[1]),
                    ReasonNode(id=1, text="Seed reason", parents=[0]),
                ],
                edges=[ReasonEdge(source=0, target=1)],
            )
            if progress_callback:
                progress_callback(1)
            yield ReasonGraph(
                fact=fact,
                nodes=[
                    ReasonNode(id=0, text=fact, children=[1]),
                    ReasonNode(id=1, text="Seed reason", parents=[0], children=[2]),
                    ReasonNode(id=2, text="Expanded reason", parents=[1]),
                ],
                edges=[
                    ReasonEdge(source=0, target=1),
                    ReasonEdge(source=1, target=2),
                ],
            )
            if trace_callback:
                trace_callback("[StubReasonGenerator] Response reasons=['Seed reason', 'Expanded reason']")
            if progress_callback:
                progress_callback(0)

    monkeypatch.setattr("fact_reason_app.ReasonGenerator", StubReasonGenerator)

    app = FactReasonApp()
    async with app.run_test() as pilot:
        fact_input = app.query_one("#fact-input", Input)
        fact_input.value = "Planets orbit stars."

        max_length_input = app.query_one("#max-length", Input)
        max_length_input.value = "32"

        max_iter_input = app.query_one("#max-iterations", Input)
        max_iter_input.value = "2"

        thread_input = app.query_one("#thread-count", Input)
        thread_input.value = "5"

        await pilot.click("#fact-input")
        await pilot.press("enter")

        for _ in range(40):
            if not app.busy:
                break
            await pilot.pause(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("Generation did not finish.")

        assert captured_inits == [(2, 32, 5)]

        tree = app.query_one("#reason-tree", Tree)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].label.plain.endswith("Seed reason")
        assert tree.root.children[1].label.plain.endswith("Expanded reason")

        status_text = str(app.query_one("#status").render())
        assert status_text.startswith("Done.")
        assert "≤2 reason" in status_text
        assert app._active_thread_count == 5
        thread_status = str(app.query_one("#thread-status").render())
        assert thread_status.startswith("Threads active: 0/5")

        log_widget = app.query_one("#detail-log", Log)
        log_lines = list(log_widget.lines)
        assert log_lines
        activity_log = app.query_one("#activity-log", Log)
        activity_lines = list(activity_log.lines)
        assert any("Starting generation" in line for line in activity_lines)
        assert any("StubReasonGenerator" in line for line in activity_lines)
        assert any("Completed generation" in line for line in activity_lines)

        history = app.query_one("#history-list")
        entries = [child for child in history.children if hasattr(child, "fact") and not getattr(child, "placeholder", False)]
        assert entries and entries[0].fact == "Planets orbit stars."
