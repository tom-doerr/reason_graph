# Instructions for Claude

When working in this repository, please follow these guidelines:

- Maintain the existing Textual TUI (`FactReasonApp`) and extend it rather than replacing it.
- Treat the fact as node 0 in the reasoning graph. All reasoning data should flow through `ReasonGenerator` and the Pydantic models in `reason_engine.py`.
- Expectations for the inference loop:
  * Each DSPy call should return all child reasons for the current node.
  * Empty child lists mean the node is complete and the queue should advance.
- Keep the settings/help overlays accessible via `Ctrl+S` and `Ctrl+H`; if you add new controls, surface them there too.
- Preserve the live thread activity indicator (`#thread-status`) so it reflects `active_workers`/thread limits during generation.
- Keep the activity log pane (`#activity-log`) wired to `_log_activity`; log entries should remain concise, single-line summaries so background threads can append safely.
- Mirror the shared numeric field metadata (`FactReasonApp._NUMERIC_FIELDS`) when adding/removing limits — that table drives both the main UI and settings modal.
- Respect environment-driven defaults: `OPENROUTER_API_KEY`, `REASON_MAX_LENGTH`, `REASON_MAX_REASONS`, and `REASON_THREADS`.
- When changing DSPy orchestration, maintain the `trace_callback` plumbing in `ReasonGenerator.iter_reasoning`; tests assert we log the module name, inputs, and outputs for every request/response pair.
- Ensure history and CLI entry points stay in sync with new features.
- Prefer headless Textual tests (`App.run_test`) and stub DSPy interactions; CI has no network access. Integration/E2E coverage lives in `tests/integration/` and `tests/e2e/`, so extend those suites when touching cross-module behaviour.
- Stay within Black-compatible formatting and use ASCII unless there’s a strong reason otherwise.
- Leave `.dspy_cache/` ignored in version control.
