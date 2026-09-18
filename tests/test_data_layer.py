from pathlib import Path
"""
Tests for Data Layer: ingestion, schema validation, derived timestamps,
and dynamic reference timestamp behavior.
"""

import os
import tempfile
import pandas as pd
import pytest

from src.config import get_reference_timestamp, get_reference_timestamp_str
from src.data_layer.loader import (
    get_cached_dataframe,
    load_and_prepare_data,
    initialize_database,
)


@pytest.fixture(scope="module", autouse=True)
def setup_dataset():
    initialize_database()


def test_csv_loading_and_counts():
    df = get_cached_dataframe()
    assert len(df) == 500
    assert (df["status"] == "Resolved").sum() == 327
    assert (df["status"] == "Open").sum() == 111
    assert (df["status"] == "Escalated").sum() == 62
    assert df["resolution_time_hrs"].isna().sum() == 173
    assert df["customer_rating"].isna().sum() == 173


def test_timestamp_minute_precision():
    df = get_cached_dataframe()
    # Check format YYYY-MM-DD HH:MM (16 chars)
    for val in df["created_at"]:
        assert len(val) == 16
        assert val[4] == "-" and val[7] == "-" and val[10] == " " and val[13] == ":"

    resolved = df[df["resolved_at"].notna()]
    for val in resolved["resolved_at"]:
        assert len(val) == 16
        assert val[4] == "-" and val[7] == "-" and val[10] == " " and val[13] == ":"


def test_reference_now_bounds():
    df = get_cached_dataframe()
    ref_now = get_reference_timestamp()
    assert ref_now >= df["created_at_dt"].max()
    assert ref_now >= df["resolved_at_dt"].max()
    assert get_reference_timestamp_str() == "2024-04-04 12:23"


def test_resolved_equivalence_invariant_in_dataset():
    df = get_cached_dataframe()
    # In the real support_tickets.csv, status == 'Resolved' is 100% equivalent to resolution_time_hrs IS NOT NULL
    has_res = df["resolution_time_hrs"].notna()
    is_resolved_status = df["status"] == "Resolved"
    assert (has_res == is_resolved_status).all()


def test_data_driven_resolution_with_mock_escalated_resolved():
    """Confirms that an Escalated ticket with resolution time is correctly derived as resolved."""
    mock_csv_content = """ticket_id,created_at,category,priority,status,response_time_hrs,resolution_time_hrs,agent_id,customer_rating,issue_summary
TKT-998,2024-02-01 10:00,Technical,Critical,Escalated,1.0,12.0,AGT-01,3,Test escalated but resolved
TKT-999,2024-02-01 12:00,General,Low,Open,2.0,,AGT-02,,Test open
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(mock_csv_content)
        tmp_path = tmp.name

    try:
        mock_df = load_and_prepare_data(csv_path=Path(tmp_path))
        esc_row = mock_df[mock_df["ticket_id"] == "TKT-998"].iloc[0]
        assert esc_row["status"] == "Escalated"
        assert pd.notna(esc_row["resolution_time_hrs"])
        assert esc_row["resolved_at"] == "2024-02-01 22:00"
    finally:
        os.remove(tmp_path)
        # Restore real dataset
        initialize_database()


def test_dynamic_reference_now_mutation():
    """Confirms REFERENCE_NOW updates dynamically when a dataset with a later event is loaded."""
    mock_csv_content = """ticket_id,created_at,category,priority,status,response_time_hrs,resolution_time_hrs,agent_id,customer_rating,issue_summary
TKT-001,2025-08-15 09:30,Billing,High,Resolved,1.0,5.0,AGT-01,5,Future event
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(mock_csv_content)
        tmp_path = tmp.name

    try:
        load_and_prepare_data(csv_path=Path(tmp_path))
        assert get_reference_timestamp_str() == "2025-08-15 14:30"
    finally:
        os.remove(tmp_path)
        # Restore real dataset
        initialize_database()
