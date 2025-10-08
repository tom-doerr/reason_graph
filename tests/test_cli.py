import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fact_reason_app


class _DummyApp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.ran = False

    def run(self):
        self.ran = True


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            [],
            {
                "initial_fact": None,
                "initial_max_length": None,
                "initial_max_reasons": None,
                "initial_max_steps": None,
                "initial_threads": None,
                "auto_generate": False,
            },
        ),
        (
            [
                "A fact",
                "--max-length",
                "70",
                "--max-reasons",
                "5",
                "--max-steps",
                "12",
                "--threads",
                "4",
            ],
            {
                "initial_fact": "A fact",
                "initial_max_length": 70,
                "initial_max_reasons": 5,
                "initial_max_steps": 12,
                "initial_threads": 4,
                "auto_generate": True,
            },
        ),
    ],
)
def test_cli_arguments(monkeypatch, argv, expected):
    created = {}

    def _fake_app(**kwargs):
        created.clear()
        created.update(kwargs)
        inst = _DummyApp(**kwargs)
        created["_instance"] = inst
        return inst

    monkeypatch.setattr(fact_reason_app, "FactReasonApp", _fake_app)
    assert fact_reason_app.main(argv) == 0
    inst = created.pop("_instance")
    assert inst.ran
    assert created == expected
