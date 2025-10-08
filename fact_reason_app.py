"""Textual TUI that turns a fact into an iterative chain of reasons via DSPy."""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Mapping, Sequence
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
    Tree,
)


class HistoryItem(ListItem):
    def __init__(self, fact: str, *, placeholder: bool = False) -> None:
        label = Label(fact)
        super().__init__(label)
        self.fact = fact
        self.placeholder = placeholder


class SettingsScreen(ModalScreen[dict[str, object]]):
    DEFAULT_CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-panel {
        layout: vertical;
        background: $panel;
        border: panel $primary;
        padding: 1 2;
        width: 60;
    }

    #settings-title {
        content-align: center middle;
        text-style: bold;
    }

    .settings-field {
        margin-top: 1;
    }

    #settings-actions {
        layout: horizontal;
        margin-top: 1;
    }

    .settings-button {
        margin-right: 2;
    }

    #settings-error {
        color: $error;
        height: 1;
    }
    """

    def __init__(
        self,
        *,
        fact: str,
        numeric_specs: Sequence[NumericFieldSpec],
        numeric_values: Mapping[str, int],
    ) -> None:
        super().__init__()
        self._initial_fact = fact
        self._numeric_specs = list(numeric_specs)
        self._numeric_values = dict(numeric_values)
        self._fact_input: Input | None = None
        self._numeric_inputs: dict[str, Input] = {}
        self._error_label: Static | None = None

    def compose(self) -> ComposeResult:
        with Container(id="settings-panel"):
            yield Static("Settings", id="settings-title")
            yield Static("Fact", classes="settings-field")
            self._fact_input = Input(id="settings-fact", classes="settings-field")
            yield self._fact_input
            for spec in self._numeric_specs:
                yield Static(spec.label, classes="settings-field")
                field_input = Input(id=spec.settings_id, classes="settings-field")
                self._numeric_inputs[spec.settings_key] = field_input
                yield field_input
            self._error_label = Static("", id="settings-error", classes="settings-field")
            yield self._error_label
            with Horizontal(id="settings-actions"):
                yield Button("Save", id="settings-save", variant="primary", classes="settings-button")
                yield Button("Cancel", id="settings-cancel", classes="settings-button")

    def on_mount(self) -> None:
        assert (
            self._fact_input
            and self._numeric_inputs
        )
        self._fact_input.value = self._initial_fact
        for spec in self._numeric_specs:
            widget = self._numeric_inputs[spec.settings_key]
            value = self._numeric_values.get(spec.attr, spec.default)
            widget.value = str(value)
        self.set_focus(self._fact_input)

    def _dismiss(self, payload: dict[str, int] | None = None) -> None:
        self.dismiss(payload)

    def _show_error(self, message: str) -> None:
        if self._error_label:
            self._error_label.update(message)

    def _clear_error(self) -> None:
        if self._error_label:
            self._error_label.update("")

    def _collect_values(self) -> dict[str, int] | None:
        assert self._fact_input and self._numeric_inputs
        fact = self._fact_input.value.strip()
        parsed: dict[str, int] = {}
        try:
            for spec in self._numeric_specs:
                widget = self._numeric_inputs[spec.settings_key]
                value = int(widget.value.strip())
                if value < spec.minimum:
                    self._show_error(spec.error_message)
                    return None
                parsed[spec.settings_key] = value
        except ValueError:
            self._show_error("All values must be integers.")
            return None
        self._clear_error()
        return {
            "fact": fact,
            **parsed,
        }

    @on(Button.Pressed, "#settings-save")
    def handle_save_pressed(self, _: Button.Pressed) -> None:
        values = self._collect_values()
        if values is not None:
            self._dismiss(values)

    @on(Button.Pressed, "#settings-cancel")
    def handle_cancel_pressed(self, _: Button.Pressed) -> None:
        self._dismiss()

    def action_cancel(self) -> None:
        self._dismiss()

from reason_engine import (
    DEFAULT_MAX_REASON_LENGTH,
    DEFAULT_THREAD_COUNT,
    DEFAULT_MAX_STEPS,
    ReasonGenerator,
    ReasonGraph,
    ReasonNode,
)


def _read_int_env(name: str, fallback: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return max(value, minimum)


DEFAULT_MAX_LENGTH = _read_int_env(
    "REASON_MAX_LENGTH",
    DEFAULT_MAX_REASON_LENGTH,
    minimum=10,
)
DEFAULT_MAX_REASONS = _read_int_env("REASON_MAX_REASONS", 4, minimum=1)
DEFAULT_THREAD_LIMIT = _read_int_env("REASON_THREADS", DEFAULT_THREAD_COUNT, minimum=1)
DEFAULT_MAX_STEPS = _read_int_env("REASON_MAX_STEPS", DEFAULT_MAX_STEPS, minimum=1)

@dataclass(frozen=True)
class NumericFieldSpec:
    selector: str
    settings_id: str
    attr: str
    settings_key: str
    label: str
    minimum: int
    default: int
    error_message: str


class FactReasonApp(App[None]):
    """Interactive terminal UI for exploring fact-based reasoning."""

    TITLE = "Reason Graph"
    _last_result: ReasonGraph | None = None
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #main {
        layout: vertical;
        height: 1fr;
        padding: 1 2;
        overflow: hidden;
    }

    #input-panel {
        layout: vertical;
        padding: 1;
        border: panel $primary;
        background: $boost;
    }

    .stack-space {
        margin-top: 1;
    }

    #hint {
        color: $text-muted;
    }

    #config {
        layout: horizontal;
    }

    .config-column {
        layout: vertical;
        padding-right: 2;
    }

    .config-field {
        margin-top: 1;
    }

    #controls {
        layout: horizontal;
    }

    .control-button {
        margin-right: 2;
    }

    #status {
        height: auto;
        color: $secondary;
    }

    #thread-status {
        height: auto;
        color: $text-muted;
    }

    .config-label {
        color: $text-muted;
    }

    #max-length, #max-iterations, #thread-count {
        width: 12;
    }

    #results {
        layout: horizontal;
        height: 1fr;
    }

    #reason-tree {
        border: solid $primary;
        padding: 1;
        width: 1fr;
    }

    #detail-log {
        border: solid $secondary 50%;
        padding: 1;
        width: 1fr;
        margin-left: 1;
    }

    #activity-log {
        border: solid $accent 40%;
        padding: 1;
        width: 1fr;
        margin-left: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "focus_input", "Focus input"),
        ("ctrl+h", "show_help", "Help"),
        ("ctrl+s", "show_settings", "Settings"),
    ]

    busy: reactive[bool] = reactive(False)
    max_reason_length: reactive[int] = reactive(DEFAULT_MAX_LENGTH)
    max_iterations: reactive[int] = reactive(DEFAULT_MAX_REASONS)
    thread_count: reactive[int] = reactive(DEFAULT_THREAD_LIMIT)
    active_workers: reactive[int] = reactive(0)
    max_steps: reactive[int] = reactive(DEFAULT_MAX_STEPS)
    _NUMERIC_FIELD_ORDER = ("max-length", "max-iterations", "max-steps", "thread-count")
    _NUMERIC_FIELDS: dict[str, NumericFieldSpec] = {
        "max-length": NumericFieldSpec(
            selector="#max-length",
            settings_id="settings-max-length",
            attr="max_reason_length",
            settings_key="max_length",
            label="Max chars/reason",
            minimum=10,
            default=DEFAULT_MAX_LENGTH,
            error_message="Max length must be an integer ≥ 10.",
        ),
        "max-iterations": NumericFieldSpec(
            selector="#max-iterations",
            settings_id="settings-max-reasons",
            attr="max_iterations",
            settings_key="max_reasons",
            label="Max reasons",
            minimum=1,
            default=DEFAULT_MAX_REASONS,
            error_message="Max reasons must be an integer ≥ 1.",
        ),
        "max-steps": NumericFieldSpec(
            selector="#max-steps",
            settings_id="settings-max-steps",
            attr="max_steps",
            settings_key="max_steps",
            label="Max steps",
            minimum=1,
            default=DEFAULT_MAX_STEPS,
            error_message="Max steps must be ≥ 1.",
        ),
        "thread-count": NumericFieldSpec(
            selector="#thread-count",
            settings_id="settings-threads",
            attr="thread_count",
            settings_key="threads",
            label="Threads",
            minimum=1,
            default=DEFAULT_THREAD_LIMIT,
            error_message="Threads must be an integer ≥ 1.",
        ),
    }

    @classmethod
    def _numeric_specs(cls) -> list[NumericFieldSpec]:
        return [cls._NUMERIC_FIELDS[key] for key in cls._NUMERIC_FIELD_ORDER]

    def __init__(
        self,
        *,
        initial_fact: str | None = None,
        initial_max_length: int | None = None,
        initial_max_reasons: int | None = None,
        initial_threads: int | None = None,
        initial_max_steps: int | None = None,
        auto_generate: bool = False,
    ) -> None:
        super().__init__()
        self._selected_reason: int | None = None
        self._last_result: ReasonGraph | None = None
        self._active_max_iterations: int = DEFAULT_MAX_REASONS
        self._active_thread_count: int = DEFAULT_THREAD_LIMIT
        self._active_max_steps: int = DEFAULT_MAX_STEPS
        self._history: list[str] = []
        self._history_placeholder: bool = True
        self._initial_fact = initial_fact.strip() if initial_fact else None
        self._auto_generate = auto_generate and bool(self._initial_fact)
        self._showing_help = False
        self._thread_status: Static | None = None
        self._detail_log: Log | None = None
        self._activity_log: Log | None = None
        self._input_cache: dict[str, Input] = {}

        if initial_max_length is not None:
            self.max_reason_length = max(initial_max_length, 10)
        if initial_max_reasons is not None:
            self.max_iterations = max(initial_max_reasons, 1)
        if initial_threads is not None:
            self.thread_count = max(initial_threads, 1)
        if initial_max_steps is not None:
            self.max_steps = max(initial_max_steps, 1)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Container(id="input-panel"):
                yield Static("Enter a fact to explore its supporting reasons.", id="prompt")
                yield Static(
                    "Tip: keep the fact concise. Adjust limits below to control how deep and verbose the graph becomes.",
                    id="hint",
                    classes="stack-space",
                )
                yield Input(
                    placeholder="e.g. The Earth orbits the Sun",
                    id="fact-input",
                    classes="stack-space",
                )
                with Horizontal(id="config", classes="stack-space"):
                    for spec in self._numeric_specs():
                        with Container(classes="config-column"):
                            yield Static(spec.label, classes="config-label")
                            yield Input(
                                value=str(getattr(self, spec.attr)),
                                id=spec.selector.lstrip("#"),
                                classes="config-field",
                            )
                with Horizontal(id="controls", classes="stack-space"):
                    yield Button(
                        "Generate reasons",
                        id="generate",
                        variant="primary",
                        classes="control-button",
                    )
                    yield Button("Clear", id="clear", variant="default", classes="control-button")
                yield Static("Waiting for input…", id="status", classes="stack-space")
                thread_status = Static(
                    f"Threads active: {self.active_workers}/{self.thread_count}",
                    id="thread-status",
                    classes="stack-space",
                )
                self._thread_status = thread_status
                yield thread_status
                yield Static("History", classes="config-label stack-space", id="history-label")
                history_list = ListView(id="history-list")
                history_list.can_focus = True
                yield history_list
            with Container(id="results"):
                graph = Tree("Reason graph will appear here.", id="reason-tree")
                graph.show_root = True
                graph.root.expand()
                yield graph
                detail_log = Log(
                    id="detail-log",
                    highlight=False,
                )
                self._detail_log = detail_log
                yield detail_log
                activity_log = Log(
                    id="activity-log",
                    highlight=False,
                )
                self._activity_log = activity_log
                yield activity_log
        yield Footer()

    def on_mount(self) -> None:
        self._last_result = None
        fact_input = self._input("#fact-input")
        if self._initial_fact:
            fact_input.value = self._initial_fact
        fact_input.focus()
        self._set_placeholder(
            root_label="Reason graph will appear here.",
            detail_message="Enter a fact and press Enter to generate an influence graph.",
        )
        self._render_history()
        self._update_thread_status()
        self._log_activity("Activity log ready. Generate a fact to see progress.", reset=True)
        if self._auto_generate and self._initial_fact:
            asyncio.create_task(self._trigger_generation(self._initial_fact))

    def watch_busy(self, busy: bool) -> None:
        """Toggle interactive elements while work is running."""
        generate = self.query_one("#generate", Button)
        clear = self.query_one("#clear", Button)
        generate.disabled = busy
        generate.variant = "primary" if not busy else "default"
        clear.disabled = busy
        self._update_thread_status()

    def watch_thread_count(self, count: int) -> None:
        self._update_thread_status()

    def watch_active_workers(self, active: int) -> None:
        self._update_thread_status()

    def _input(self, selector: str) -> Input:
        cached = self._input_cache.get(selector)
        if cached is None:
            cached = self.query_one(selector, Input)
            self._input_cache[selector] = cached
        return cached

    def _field_spec(self, field_key: str) -> NumericFieldSpec:
        return self._NUMERIC_FIELDS[field_key]

    def _focus_field(self, field_key: str) -> None:
        spec = self._field_spec(field_key)
        self._input(spec.selector).focus()

    def _handle_numeric_change(self, field_key: str, raw_value: str) -> None:
        value = raw_value.strip()
        if not value:
            return
        spec = self._field_spec(field_key)
        parsed = self._parse_numeric(value, minimum=spec.minimum)
        if parsed is None:
            self._status_message(spec.error_message)
            return
        setattr(self, spec.attr, parsed)

    def _set_active_workers(self, count: int) -> None:
        total_capacity = self._active_thread_count if self.busy else self.thread_count
        total_capacity = max(total_capacity, 1)
        self.active_workers = max(0, min(count, total_capacity))

    def _update_thread_status(self) -> None:
        if self._thread_status is None:
            return
        total = self._active_thread_count if self.busy else self.thread_count
        total = max(total, 1)
        self._thread_status.update(f"Threads active: {self.active_workers}/{total}")

    def action_focus_input(self) -> None:
        """Keyboard binding to jump back to the input box."""
        self._input("#fact-input").focus()

    @on(Input.Submitted, "#fact-input")
    async def handle_fact_submitted(self, event: Input.Submitted) -> None:
        await self._trigger_generation(event.value)

    @on(Button.Pressed, "#generate")
    async def handle_generate_pressed(self, _: Button.Pressed) -> None:
        fact = self._input("#fact-input").value
        await self._trigger_generation(fact)

    @on(Button.Pressed, "#clear")
    def handle_clear_pressed(self, _: Button.Pressed) -> None:
        input_box = self._input("#fact-input")
        input_box.value = ""
        for spec in self._numeric_specs():
            widget = self._input(spec.selector)
            widget.value = str(spec.default)
            setattr(self, spec.attr, spec.default)
        self._status_message("Waiting for input…")
        self._set_placeholder(
            root_label="Reason graph will appear here.",
            detail_message="Cleared. Enter a new fact to regenerate the graph.",
        )
        input_box.focus()
        self._render_history()

    @on(Input.Changed, "#max-length")
    def handle_max_length_changed(self, event: Input.Changed) -> None:
        self._handle_numeric_change("max-length", event.value)

    @on(Input.Changed, "#max-iterations")
    def handle_max_iterations_changed(self, event: Input.Changed) -> None:
        self._handle_numeric_change("max-iterations", event.value)

    @on(Input.Changed, "#thread-count")
    def handle_thread_count_changed(self, event: Input.Changed) -> None:
        self._handle_numeric_change("thread-count", event.value)

    @on(Input.Changed, "#max-steps")
    def handle_max_steps_changed(self, event: Input.Changed) -> None:
        self._handle_numeric_change("max-steps", event.value)

    async def _trigger_generation(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            self._status_message("Please provide a fact before generating reasons.")
            return

        max_length = self._current_max_length()
        if max_length is None:
            spec = self._field_spec("max-length")
            self._status_message(spec.error_message)
            self._focus_field("max-length")
            return
        self.max_reason_length = max_length

        max_iterations = self._current_max_iterations()
        if max_iterations is None:
            spec = self._field_spec("max-iterations")
            self._status_message(spec.error_message)
            self._focus_field("max-iterations")
            return
        self.max_iterations = max_iterations

        thread_count = self._current_thread_count()
        if thread_count is None:
            spec = self._field_spec("thread-count")
            self._status_message(spec.error_message)
            self._focus_field("thread-count")
            return
        self.thread_count = thread_count

        max_steps = self._current_max_steps()
        if max_steps is None:
            spec = self._field_spec("max-steps")
            self._status_message(spec.error_message)
            self._focus_field("max-steps")
            return
        self.max_steps = max_steps

        if self.busy:
            return

        self.busy = True
        self._status_message(
            f"Generating reasons with DSPy (≤{max_iterations} reasons, ≤{max_steps} steps, {thread_count} threads)…"
        )
        self._start_generation_ui(fact, max_iterations, thread_count, max_steps)

        def run_generation() -> None:
            generator = ReasonGenerator(
                max_reasons=max_iterations,
                max_reason_length=max_length,
                max_steps=max_steps,
                num_threads=thread_count,
            )
            final_result = ReasonGraph(
                fact=fact,
                nodes=[ReasonNode(id=0, text=fact)],
                edges=[],
            )

            def progress_callback(active: int) -> None:
                self.call_from_thread(self._set_active_workers, active)
            def trace_callback(message: str) -> None:
                self.call_from_thread(self._log_activity, message)

            try:
                for snapshot in generator.iter_reasoning(
                    fact,
                    progress_callback=progress_callback,
                    trace_callback=trace_callback,
                ):
                    final_result = snapshot
                    self.call_from_thread(self._handle_intermediate_result, snapshot)
                self.call_from_thread(self._handle_generation_complete, final_result)
            except Exception as error:  # pragma: no cover - defensive
                self.call_from_thread(self._handle_generation_error, error)
            finally:
                self.call_from_thread(self._set_active_workers, 0)

        self.run_worker(
            run_generation,
            description=f"Generate reasons for: {fact}",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def _current_max_length(self) -> int | None:
        return self._read_numeric_value("max-length")

    @staticmethod
    def _parse_numeric(raw: str, *, minimum: int) -> int | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= minimum else None

    def _read_numeric_value(self, field_key: str) -> int | None:
        spec = self._field_spec(field_key)
        widget = self._input(spec.selector)
        raw = widget.value.strip()
        if not raw:
            return spec.default
        return self._parse_numeric(raw, minimum=spec.minimum)

    def _current_max_iterations(self) -> int | None:
        return self._read_numeric_value("max-iterations")

    def _current_thread_count(self) -> int | None:
        return self._read_numeric_value("thread-count")

    def _current_max_steps(self) -> int | None:
        return self._read_numeric_value("max-steps")

    def _status_message(self, message: str) -> None:
        status = self.query_one("#status", Static)
        status.update(message)

    def _render_history(self) -> None:
        history_view = self.query_one("#history-list", ListView)
        history_view.clear()
        if not self._history:
            history_view.append(HistoryItem("No history yet.", placeholder=True))
            self._history_placeholder = True
        else:
            self._history_placeholder = False
            for fact in self._history:
                history_view.append(HistoryItem(fact))

    def _append_history(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        if fact in self._history:
            self._history.remove(fact)
        self._history.insert(0, fact)
        if len(self._history) > 20:
            self._history = self._history[:20]
        self._render_history()

    def _start_generation_ui(self, fact: str, max_iterations: int, thread_count: int, max_steps: int) -> None:
        self._selected_reason = None
        self._last_result = ReasonGraph(
            fact=fact,
            nodes=[ReasonNode(id=0, text=fact)],
            edges=[],
        )
        self._active_max_iterations = max_iterations
        self._active_thread_count = thread_count
        self._active_max_steps = max_steps
        self._showing_help = False
        self._set_active_workers(0)
        tree = self.query_one("#reason-tree", Tree)
        tree.clear()
        root = tree.root
        root.label = f"Fact: {fact}"
        root.data = {"type": "fact"}
        root.expand()
        tree.select_node(root)
        self._write_detail_lines(
            [
                "Generating reasons…",
                "As each reason arrives it will appear in the graph.",
                f"This run allows up to {max_iterations} total reasons.",
                f"Iteration budget: {max_steps} node expansions.",
                f"Thread pool configured for {thread_count} worker(s).",
                "Use the arrow keys to explore earlier reasons while generation continues.",
            ]
        )
        self._log_activity(
            f"Starting generation for fact: {fact}",
            f"Budget: ≤{max_iterations} reason(s), ≤{max_steps} step(s), {thread_count} thread(s).",
            reset=True,
        )

    def _render_graph(
        self,
        outcome: ReasonGraph,
        *,
        focus_index: int | None = None,
        preserve_selection: bool = False,
    ) -> None:
        self._last_result = outcome
        tree = self.query_one("#reason-tree", Tree)
        tree.clear()
        root = tree.root
        root.label = f"Fact: {outcome.fact}"
        root.data = {"type": "fact"}

        for node_model in outcome.nodes[1:]:
            label = f"{node_model.id}. {node_model.text}"
            node = root.add_leaf(label)
            node.data = {"type": "reason", "index": node_model.id}

        root.expand()
        selected_reason_id: int | None
        if preserve_selection and self._selected_reason is not None:
            selected_reason_id = self._selected_reason
        elif focus_index is not None:
            selected_reason_id = focus_index
        elif self._selected_reason is not None:
            selected_reason_id = self._selected_reason
        elif root.children:
            first_child = root.children[0]
            selected_reason_id = first_child.data.get("index") if first_child.data else 1
        else:
            selected_reason_id = None

        if selected_reason_id is not None and root.children:
            child_index = max(0, min(selected_reason_id - 1, len(root.children) - 1))
            node = root.children[child_index]
            tree.select_node(node)
            self._selected_reason = selected_reason_id
            self._update_detail_for_reason(selected_reason_id)
        else:
            self._selected_reason = None
            tree.select_node(root)
            self._update_detail_for_fact()

    def _set_placeholder(self, *, root_label: str, detail_message: str) -> None:
        tree = self.query_one("#reason-tree", Tree)
        tree.clear()
        root = tree.root
        root.label = root_label
        root.data = {"type": "placeholder"}
        root.expand()
        tree.select_node(root)
        self._write_detail_lines([detail_message])
        self._last_result = None
        self._selected_reason = None
        self._active_max_iterations = self.max_iterations
        self._active_thread_count = self.thread_count
        self._active_max_steps = self.max_steps
        self._set_active_workers(0)

    def _handle_intermediate_result(self, snapshot: ReasonGraph) -> None:
        prior_count = (len(self._last_result.nodes) - 1) if self._last_result else 0
        follow_chain = (
            self._selected_reason is None
            or self._selected_reason == prior_count
        )
        current_count = len(snapshot.nodes) - 1
        focus_index = (snapshot.nodes[-1].id if follow_chain and current_count > 0 else None)
        self._render_graph(
            snapshot,
            focus_index=focus_index,
            preserve_selection=not follow_chain,
        )
        self._status_message(
            f"Generated {current_count} of ≤{self._active_max_iterations} reason(s) (≤{self._active_max_steps} step budget)…"
        )
        delta = current_count - prior_count
        if delta > 0:
            new_nodes = snapshot.nodes[-delta:]
            labels = ", ".join(f"Reason {node.id}" for node in new_nodes)
            self._log_activity(f"Added {delta} reason(s): {labels} (total {current_count}).")
        else:
            self._log_activity(f"Step completed with {current_count} reason(s); no new additions.")

    def _handle_generation_complete(self, outcome: ReasonGraph) -> None:
        self._set_active_workers(0)
        if len(outcome.nodes) > 1:
            self._render_graph(outcome, preserve_selection=True)
            self._status_message(
                f"Done. Generated {len(outcome.nodes) - 1} of ≤{self._active_max_iterations} reason(s) within ≤{self._active_max_steps} step(s)."
            )
            self._append_history(outcome.fact)
            self._log_activity(
                f"Completed generation with {len(outcome.nodes) - 1} reason(s) captured."
            )
        else:
            self._render_graph(outcome, preserve_selection=True)
            self._status_message("Done. No reasons produced; try rephrasing the fact.")
            self._write_detail_lines(
                [
                    "The model did not return any supporting reasons.",
                    "Try rephrasing the fact or increasing the max length.",
                ]
            )
            self._log_activity("Generation completed without new supporting reasons.")
        self.busy = False

    def _handle_generation_error(self, error: Exception) -> None:
        self._set_active_workers(0)
        self._set_placeholder(
            root_label="Generation failed.",
            detail_message=f"Failed to generate reasons: {error}",
        )
        self._status_message("Failed to generate reasons. See details above.")
        self.busy = False
        self._active_max_iterations = self.max_iterations
        self._active_thread_count = self.thread_count
        self._active_max_steps = self.max_steps
        self._log_activity(f"Generation failed: {error}")

    def _write_detail_lines(self, lines) -> None:
        log = self._detail_log
        if log is None:
            return
        log.clear()
        for line in lines:
            log.write(line)
        self._showing_help = False

    def _log_activity(self, *lines: str, reset: bool = False) -> None:
        log = self._activity_log
        if log is None:
            return
        if reset:
            log.clear()
        for line in lines:
            if not line:
                continue
            for segment in str(line).splitlines():
                if segment:
                    log.write(segment)

    def _update_detail_for_fact(self) -> None:
        result = self._last_result
        if result is None:
            self._write_detail_lines(["Enter a fact to generate its reasoning graph."])
            return

        direct_children = result.nodes[0].children if result.nodes else []
        lines = ["Fact:", result.fact]
        if direct_children:
            lines.append("")
            lines.append("Directly supported by:")
            for idx in direct_children:
                lines.append(f"- Reason {idx}: {result.nodes[idx].text}")
        else:
            lines.append("")
            lines.append("No supporting reasons identified yet.")
        self._write_detail_lines(lines)

    def _update_detail_for_reason(self, index: int) -> None:
        result = self._last_result
        if result is None or not (0 <= index < len(result.nodes)):
            self._write_detail_lines(["Reason details unavailable."])
            return

        reason_node = result.nodes[index]
        reason = reason_node.text
        predecessors = reason_node.parents
        successors = reason_node.children

        lines = [f"Reason {index}:", reason, ""]
        lines.append("Influenced by:")
        if predecessors:
            for src in predecessors:
                if src == 0:
                    lines.append("- Fact")
                else:
                    label = f"Reason {src}" if src > 0 else "Fact"
                    lines.append(f"- {label}: {result.nodes[src].text}")
        else:
            lines.append("- Fact")

        lines.append("")
        lines.append("Supports:")
        if successors:
            for dst in successors:
                label = f"Reason {dst}" if dst > 0 else "Fact"
                lines.append(f"- {label}: {result.nodes[dst].text}")
        else:
            lines.append("- This is a terminal reason.")

        self._write_detail_lines(lines)

    def _root_reasons(self, result: ReasonGraph) -> list[int]:
        referenced = {edge.target for edge in result.edges if edge.source != 0}
        return [idx for idx in range(1, len(result.nodes)) if idx not in referenced]

    def _handle_tree_node(self, node) -> None:
        if node is None:
            return
        data = getattr(node, "data", None) or {}
        node_type = data.get("type")
        if node_type == "reason":
            index = data.get("index")
            if isinstance(index, int):
                self._selected_reason = index
                self._update_detail_for_reason(index)
        elif node_type == "fact":
            self._selected_reason = None
            self._update_detail_for_fact()

    @on(Tree.NodeHighlighted, "#reason-tree")
    def handle_tree_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node and (event.node.data or {}).get("type") != "placeholder":
            self._handle_tree_node(event.node)

    @on(Tree.NodeSelected, "#reason-tree")
    def handle_tree_selected(self, event: Tree.NodeSelected) -> None:
        if event.node and (event.node.data or {}).get("type") != "placeholder":
            self._handle_tree_node(event.node)

    @on(ListView.Selected, "#history-list")
    def handle_history_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, HistoryItem) and not item.placeholder:
            input_box = self._input("#fact-input")
            input_box.value = item.fact
            input_box.focus()
            self._status_message("Loaded fact from history. Press Enter to regenerate.")
            self._showing_help = False

    HELP_LINES = (
        "Reason Graph Help",
        "",
        "Controls:",
        "  Enter       - Generate the graph for the current fact",
        "  Ctrl+R      - Focus the fact input",
        "  Ctrl+H      - Toggle this help panel",
        "  Up/Down     - Navigate history or graph nodes",
        "  Enter (history) - Load the selected historical fact",
        "",
        "Tips:",
        "  - Adjust character, reason, and thread limits before generating.",
        "  - Use the detail pane to inspect influences between reasons.",
        "  - The history keeps the 20 most recent facts for quick recall.",
        "  - Thread activity indicator shows how many workers are running at once.",
    )

    def action_show_help(self) -> None:
        log = self._detail_log
        if log is None:
            return
        if self._showing_help:
            self._showing_help = False
            if self._selected_reason is not None:
                self._update_detail_for_reason(self._selected_reason)
            else:
                self._update_detail_for_fact()
            self._status_message("Closed help panel.")
            return

        log.clear()
        for line in self.HELP_LINES:
            log.write(line)
        self._status_message("Help panel open. Press Ctrl+H again to close.")
        self._showing_help = True

    def action_show_settings(self) -> None:
        specs = self._numeric_specs()
        numeric_values = {spec.attr: getattr(self, spec.attr) for spec in specs}
        screen = SettingsScreen(
            fact=self._input("#fact-input").value,
            numeric_specs=specs,
            numeric_values=numeric_values,
        )
        self._showing_help = False
        self.push_screen(screen, self._apply_settings)

    def _apply_settings(self, result: dict[str, object] | None) -> None:
        if not result:
            self._status_message("Settings unchanged.")
            return

        self.max_reason_length = int(result["max_length"])
        self.max_iterations = int(result["max_reasons"])
        self.max_steps = int(result.get("max_steps", self.max_steps))
        self.thread_count = int(result["threads"])
        fact_value = str(result.get("fact", "")).strip()

        for spec in self._numeric_specs():
            widget = self._input(spec.selector)
            widget.value = str(getattr(self, spec.attr))
        fact_input = self._input("#fact-input")
        if fact_value:
            fact_input.value = fact_value
        fact_input.focus()
        self._active_max_steps = self.max_steps

        self._status_message(
            f"Settings updated (≤{self.max_iterations} reasons, ≤{self.max_steps} steps, {self.max_reason_length} chars, {self.thread_count} threads)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Textual fact reasoning explorer")
    parser.add_argument("fact", nargs="?", help="Fact to explain")
    parser.add_argument("--max-length", type=int, dest="max_length", help="Maximum characters per reason")
    parser.add_argument("--max-reasons", type=int, dest="max_reasons", help="Maximum number of reasons in the graph")
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="Maximum iteration steps")
    parser.add_argument("--threads", type=int, dest="threads", help="Number of DSPy threads to use")
    args = parser.parse_args(argv)

    app = FactReasonApp(
        initial_fact=args.fact,
        initial_max_length=args.max_length,
        initial_max_reasons=args.max_reasons,
        initial_threads=args.threads,
        initial_max_steps=args.max_steps,
        auto_generate=bool(args.fact),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
