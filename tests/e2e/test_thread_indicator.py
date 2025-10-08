import sys
import time
from pathlib import Path

import pytest
from textual.widgets import Input, Log, Static

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_reason_app import FactReasonApp  # noqa: E402
from reason_engine import ReasonEdge, ReasonGraph, ReasonNode  # noqa: E402


@pytest.mark.asyncio
async def test_thread_indicator_shows_activity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured_threads: list[int] = []

    class StubReasonGenerator:
        def __init__(self, *_, num_threads: int, **__):
            captured_threads.append(num_threads)

        def iter_reasoning(self, fact: str, progress_callback=None, trace_callback=None):
            if trace_callback:
                trace_callback(f"[StubReasonGenerator] Request fact='{fact}'")
            if progress_callback:
                progress_callback(2)
            time.sleep(0.05)
            yield ReasonGraph(
                fact=fact,
                nodes=[
                    ReasonNode(id=0, text=fact, children=[1]),
                    ReasonNode(id=1, text="Primary support", parents=[0]),
                ],
                edges=[ReasonEdge(source=0, target=1)],
            )
            if progress_callback:
                progress_callback(1)
            time.sleep(0.05)
            yield ReasonGraph(
                fact=fact,
                nodes=[
                    ReasonNode(id=0, text=fact, children=[1]),
                    ReasonNode(id=1, text="Primary support", parents=[0], children=[2]),
                    ReasonNode(id=2, text="Secondary insight", parents=[1]),
                ],
                edges=[
                    ReasonEdge(source=0, target=1),
                    ReasonEdge(source=1, target=2),
                ],
            )
            if trace_callback:
                trace_callback("[StubReasonGenerator] Response reasons=['Primary support', 'Secondary insight']")
            if progress_callback:
                progress_callback(0)

    monkeypatch.setattr("fact_reason_app.ReasonGenerator", StubReasonGenerator)

    app = FactReasonApp()
    async with app.run_test() as pilot:
        fact_input = app.query_one("#fact-input", Input)
        fact_input.value = "Moons orbit planets."

        thread_input = app.query_one("#thread-count", Input)
        thread_input.value = "6"

        await pilot.click("#fact-input")
        await pilot.press("enter")

        for _ in range(40):
            if app.active_workers > 0:
                break
            await pilot.pause(0.05)

        for _ in range(40):
            if not app.busy:
                break
            await pilot.pause(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("Background generation did not finish.")

        status_text = str(app.query_one("#status").render())
        assert status_text.startswith("Done.")

        thread_widget = app.query_one("#thread-status", Static)
        thread_text = str(thread_widget.render())
        assert thread_text == "Threads active: 0/6"
        assert app.active_workers == 0
        assert captured_threads == [6]
        activity_log = app.query_one("#activity-log", Log)
        assert any("StubReasonGenerator" in line for line in activity_log.lines)
