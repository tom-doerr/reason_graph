import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reason_engine import ReasonGenerator


missing_key = not os.getenv("OPENROUTER_API_KEY")


@pytest.mark.skipif(missing_key, reason="OPENROUTER_API_KEY is not set; live OpenRouter call skipped.")
def test_openrouter_live_reason_generation():
    generator = ReasonGenerator(
        max_reasons=3,
        max_reason_length=160,
        max_steps=6,
        num_threads=1,
    )
    fact = "Solar panels provide electricity to residential homes."

    graph = generator(fact)

    assert graph.fact == fact
    # Expect at least one supporting reason when using the live model.
    assert len(graph.nodes) >= 2
    first_reason = graph.nodes[1]
    assert first_reason.text
