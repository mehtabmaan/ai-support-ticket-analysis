"""
Automated tests for Streamlit UI interactions using Streamlit's AppTest framework.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py")


def test_ui_initial_load():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Verify persistent banner
    banner_markdown = [m.value for m in at.markdown if "Dataset Snapshot Reference Date" in m.value]
    assert len(banner_markdown) > 0
    assert "2024-04-04 12:23" in banner_markdown[0]


def test_ui_query_interaction():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    # Select sample query
    query_selectbox = at.selectbox[0]
    query_selectbox.select("How many tickets are currently open?")
    at.button[0].click()
    at.run(timeout=30)

    assert len(at.exception) == 0
    answer_markdown = [m.value for m in at.markdown if "Executive Summary" in m.value]
    assert len(answer_markdown) > 0
    assert "111" in answer_markdown[0]


def test_ui_anomaly_filter_interaction():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    # Filter by Billing
    category_filter = at.selectbox[1]
    category_filter.select("Billing")
    at.run(timeout=30)

    assert len(at.exception) == 0
    filtered_text = [m.value for m in at.markdown if "Showing" in m.value and "matching anomalies" in m.value]
    assert len(filtered_text) > 0
    assert "30" in filtered_text[0]
