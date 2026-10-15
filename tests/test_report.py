"""Markdown tables keep their row labels."""

import pandas as pd

from rotation_lab.report import _md_table


def test_named_index_is_a_column():
    frame = pd.DataFrame({"Jan": [1.25]}, index=pd.Index(["Gold"], name="asset"))
    text = _md_table(frame)
    assert "| asset | Jan |" in text
    assert "Gold" in text
    assert "1.25" in text


def test_range_index_is_not_rendered():
    text = _md_table(pd.DataFrame({"ticker": ["GLD"]}))
    assert text.splitlines()[0] == "| ticker |"
    assert "GLD" in text


def test_empty_frame():
    assert _md_table(pd.DataFrame()) == "_No data._"
