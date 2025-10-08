import os
import sys
from pathlib import Path

import pytest
from textual.widgets import Log, Tree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_reason_app import FactReasonApp


missing_key = not os.getenv("OPENROUTER_API_KEY")


@pytest.mark.asyncio
@pytest.mark.skipif(missing_key, reason="OPENROUTER_API_KEY is not set; live OpenRouter call skipped.")
async def test_textual_live_generation():
    fact = "Solar panels generate electricity for residential homes."

    app = FactReasonApp(
        initial_fact=fact,
        initial_max_length=160,
        initial_max_reasons=3,
        initial_max_steps=6,
        initial_threads=1,
        auto_generate=True,
    )

    async with app.run_test() as pilot:
        # Wait until generation starts.
        for _ in range(50):
            if app.busy:
                break
            await pilot.pause(0.1)

        # Wait for generation to finish and the tree to populate.
        for _ in range(180):
            tree = app.query_one("#reason-tree", Tree)
            if not app.busy and tree.root.children:
                break
            await pilot.pause(0.25)
        else:  # pragma: no cover - defensive
            raise AssertionError("Live generation did not complete in time.")

        status_text = str(app.query_one("#status").render())
        assert status_text.startswith("Done.")
        assert "No reasons produced" not in status_text

        tree = app.query_one("#reason-tree", Tree)
        assert tree.root.children, "Expected at least one supporting reason."
        first_reason_label = tree.root.children[0].label.plain
        assert first_reason_label

        log_widget = app.query_one("#detail-log", Log)
        assert log_widget.lines, "Detail log should contain reasoning context."
        activity_log = app.query_one("#activity-log", Log)
        activity_lines = list(activity_log.lines)
        assert any("Starting generation" in line for line in activity_lines)
        assert any("Request -> DSPy module" in line for line in activity_lines)
        assert any("Response <- DSPy module" in line for line in activity_lines)
        assert any("Completed generation" in line for line in activity_lines)

        result = app._last_result
        assert result is not None
        assert len(result.nodes) >= 2, "Live model should provide supporting reasons."

        history_view = app.query_one("#history-list")
        history_entries = [
            item for item in history_view.children if hasattr(item, "fact") and not getattr(item, "placeholder", False)
        ]
        assert history_entries and history_entries[0].fact == fact
