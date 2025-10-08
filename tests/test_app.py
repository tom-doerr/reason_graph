import sys
import time
from pathlib import Path

import pytest

from textual.widgets import Input, Log, Tree

import fact_reason_app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_reason_app import FactReasonApp
from reason_engine import ReasonEdge, ReasonGraph, ReasonNode


@pytest.mark.asyncio
async def test_app_initial_placeholder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    app = FactReasonApp()
    async with app.run_test() as pilot:
        tree = app.query_one("#reason-tree", Tree)
        assert tree.root.label.plain == "Reason graph will appear here."
        status_widget = app.query_one("#status")
        assert "Waiting for input" in str(status_widget.render())
        activity_log = app.query_one("#activity-log", Log)
        assert any("Activity log ready" in line for line in activity_log.lines)


@pytest.mark.asyncio
async def test_app_streams_reasons(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured_threads: list[int] = []

    class StubReasonGenerator:
        def __init__(self, *_, num_threads: int, **__):
            captured_threads.append(num_threads)

        def iter_reasoning(self, fact: str, progress_callback=None, trace_callback=None):
            if trace_callback:
                trace_callback(f"[StubReasonGenerator] Request fact='{fact}'")
            if progress_callback:
                progress_callback(1)
            time.sleep(0.05)
            yield ReasonGraph(
                fact=fact,
                nodes=[
                    ReasonNode(id=0, text=fact, children=[1]),
                    ReasonNode(id=1, text="First", parents=[0]),
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
                    ReasonNode(id=1, text="First", parents=[0], children=[2]),
                    ReasonNode(id=2, text="Second", parents=[1]),
                ],
                edges=[
                    ReasonEdge(source=0, target=1),
                    ReasonEdge(source=1, target=2),
                ],
            )
            if trace_callback:
                trace_callback("[StubReasonGenerator] Response reasons=['First', 'Second']")
            if progress_callback:
                progress_callback(0)

    monkeypatch.setattr("fact_reason_app.ReasonGenerator", StubReasonGenerator)

    app = FactReasonApp()
    async with app.run_test() as pilot:
        fact_input = app.query_one("#fact-input", Input)
        fact_input.value = "Gravity keeps planets in orbit."

        threads_input = app.query_one("#thread-count", Input)
        threads_input.value = "3"

        await pilot.click("#fact-input")
        await pilot.press("enter")

        # Allow background worker to finish streaming snapshots.
        for _ in range(40):
            if not app.busy:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("App never finished generating reasons.")

        tree = app.query_one("#reason-tree", Tree)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].label.plain.endswith("First")
        assert tree.root.children[1].label.plain.endswith("Second")

        assert captured_threads == [3]

        history = app.query_one("#history-list")
        items = [child for child in history.children if hasattr(child, "fact") and not getattr(child, "placeholder", False)]
        assert items and items[0].fact == "Gravity keeps planets in orbit."

        await pilot.click("#history-list")
        await pilot.press("enter")
        assert app.query_one("#fact-input", Input).value == "Gravity keeps planets in orbit."

        thread_status = str(app.query_one("#thread-status").render())
        assert thread_status.startswith("Threads active: 0/3")
        activity_log = app.query_one("#activity-log", Log)
        activity_lines = list(activity_log.lines)
        assert any("Starting generation" in line for line in activity_lines)
        assert any("StubReasonGenerator" in line for line in activity_lines)
        assert any("Completed generation" in line for line in activity_lines)


@pytest.mark.asyncio
async def test_help_toggle(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    app = FactReasonApp()
    async with app.run_test() as pilot:
        app.action_show_help()
        help_log = app.query_one("#detail-log")
        assert any("Reason Graph Help" in line for line in help_log.lines)
        assert app._showing_help

        app.action_show_help()
        assert not app._showing_help


@pytest.mark.asyncio
async def test_thread_status_updates_with_reactive_changes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    app = FactReasonApp()
    async with app.run_test() as pilot:
        thread_status = app.query_one("#thread-status")
        assert str(thread_status.render()).startswith("Threads active: 0/")

        app.thread_count = 9
        await pilot.pause(0)
        assert str(thread_status.render()) == "Threads active: 0/9"

        app._set_active_workers(3)
        await pilot.pause(0)
        assert str(thread_status.render()) == "Threads active: 3/9"


@pytest.mark.asyncio
async def test_generation_error_resets_thread_status(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class ErrorReasonGenerator:
        def __init__(self, *_, **__):
            pass

        def iter_reasoning(self, fact: str, progress_callback=None, trace_callback=None):
            if trace_callback:
                trace_callback(f"[ErrorReasonGenerator] Request fact='{fact}'")
            if progress_callback:
                progress_callback(1)
            raise RuntimeError("Test failure")
            if False:  # pragma: no cover - generator formality
                yield None

    monkeypatch.setattr("fact_reason_app.ReasonGenerator", ErrorReasonGenerator)

    app = FactReasonApp()
    async with app.run_test() as pilot:
        fact_input = app.query_one("#fact-input", Input)
        fact_input.value = "Unstable fact."

        thread_input = app.query_one("#thread-count", Input)
        thread_input.value = "2"

        await pilot.click("#fact-input")
        await pilot.press("enter")

        for _ in range(40):
            if not app.busy:
                break
            await pilot.pause(0.05)
        else:  # pragma: no cover - defensive
            raise AssertionError("Error path did not finish.")

        status_text = str(app.query_one("#status").render())
        assert "Failed" in status_text

        thread_status = str(app.query_one("#thread-status").render())
        assert thread_status == "Threads active: 0/2"

        assert app.active_workers == 0
        activity_log = app.query_one("#activity-log", Log)
        assert any("Generation failed" in line for line in activity_log.lines)


@pytest.mark.asyncio
async def test_settings_menu_updates(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    app = FactReasonApp()
    async with app.run_test() as pilot:
        app.action_show_settings()
        await pilot.pause(0.05)

        settings_screen = next(child for child in app.children if isinstance(child, fact_reason_app.SettingsScreen))
        fact_input = settings_screen.query_one("#settings-fact", Input)
        max_length_input = settings_screen.query_one("#settings-max-length", Input)
        max_reasons_input = settings_screen.query_one("#settings-max-reasons", Input)
        max_steps_input = settings_screen.query_one("#settings-max-steps", Input)
        threads_input = settings_screen.query_one("#settings-threads", Input)

        fact_input.value = "Updated fact"
        max_length_input.value = "90"
        max_reasons_input.value = "6"
        max_steps_input.value = "11"
        threads_input.value = "7"

        values = settings_screen._collect_values()
        assert values == {"fact": "Updated fact", "max_length": 90, "max_reasons": 6, "max_steps": 11, "threads": 7}
        settings_screen._dismiss(values)
        await pilot.pause(0.05)

        assert app.query_one("#fact-input", Input).value == "Updated fact"
        assert app.max_reason_length == 90
        assert app.max_iterations == 6
        assert app.max_steps == 11
        assert app.thread_count == 7

        assert app.query_one("#max-length", Input).value == "90"
        assert app.query_one("#max-iterations", Input).value == "6"
        assert app.query_one("#max-steps", Input).value == "11"
        assert app.query_one("#thread-count", Input).value == "7"
