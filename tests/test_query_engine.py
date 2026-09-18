"""
Tests for Natural Language Query Engine: accuracy, verb disambiguation,
tie detection, and generic sparse-window caveats.
"""

import pytest
from src.data_layer.loader import initialize_database
from src.llm.query_engine import NLQueryEngine


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    initialize_database()


def test_open_tickets_query():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("How many tickets are currently open?")
    assert resp["row_count"] == 1
    assert resp["results"][0]["open_tickets_count"] == 111
    assert "111" in resp["answer"]


def test_critical_unresolved_query():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("How many critical tickets are unresolved?")
    assert resp["row_count"] == 1
    assert resp["results"][0]["unresolved_critical_count"] == 31
    assert "31" in resp["answer"]


def test_lowest_rated_agent_query():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("Which agent has the lowest average customer rating?")
    top = resp["results"][0]
    assert top["agent_id"] == "AGT-08"
    assert round(top["avg_rating"], 2) == 3.48


def test_most_resolved_this_month_sparse_caveat():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("Which agent resolved the most tickets this month?")
    # Under REFERENCE_NOW 2024-04-04, April has 1 ticket resolved by AGT-07
    top = resp["results"][0]
    assert top["agent_id"] == "AGT-07"
    assert top["resolved_tickets_count"] == 1
    # Check that the sparse data caveat and March context are surfaced
    assert resp["caveats"] is not None
    assert "statistically sparse" in resp["caveats"].lower()
    assert "AGT-01" in resp["caveats"]


def test_most_resolved_march_2024():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("Which agent resolved the most tickets in March 2024?")
    top = resp["results"][0]
    assert top["agent_id"] == "AGT-01"
    assert top["resolved_tickets_count"] == 16


def test_most_resolved_overall_tie_detection():
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("Which agent resolved the most tickets overall?")
    assert resp["results"][0]["agent_id"] in ["AGT-09", "AGT-12"]
    assert resp["results"][1]["agent_id"] in ["AGT-09", "AGT-12"]
    assert resp["results"][0]["total_resolved"] == 37
    assert resp["results"][1]["total_resolved"] == 37
    # Tie note must be present
    assert resp["caveats"] is not None
    assert "tied" in resp["caveats"].lower()
    assert "AGT-09" in resp["caveats"] and "AGT-12" in resp["caveats"]


def test_generic_weekly_sparse_window():
    """Confirms the generic sparse-window detector triggers for weekly queries as well."""
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("Are there any anomalies in resolution times this week?")
    assert resp["caveats"] is not None
    assert "past 7-day window" in resp["caveats"].lower()
    assert "dataset average" in resp["caveats"].lower()


def test_out_of_scope_query_graceful_handling():
    """Confirms off-topic questions return a helpful, polite message without crashing."""
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("What is the capital of France and what is the weather there?")
    assert resp["row_count"] == 0
    assert resp["sql"] is None
    assert "unable to translate your question" in resp["answer"]
    assert resp["error"] == "No query generated"


def test_offline_fallback_mode(monkeypatch):
    """Confirms queries execute in offline deterministic mode with zero stack traces."""
    monkeypatch.setattr("src.config.settings.GROQ_API_KEY", None)
    monkeypatch.setattr("src.config.settings.LLM_PROVIDER", "fallback")
    engine = NLQueryEngine()
    resp = engine.execute_nl_query("How many tickets are currently open?")
    assert resp["is_fallback"] is True
    assert resp["row_count"] == 1
    assert resp["results"][0]["open_tickets_count"] == 111
    assert "111" in resp["answer"]
    assert resp["error"] is None

