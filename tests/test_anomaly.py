"""
Tests for Anomaly Detection Engine: per-category IQR thresholding,
SLA breaches, and explanation generation.
"""

import pytest
from src.anomaly.detector import AnomalyDetector
from src.data_layer.loader import initialize_database


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    initialize_database()


def test_category_thresholds():
    detector = AnomalyDetector()
    stats = detector.get_category_resolution_stats()

    assert "Billing" in stats
    assert "General" in stats
    assert "Technical" in stats

    # Verify category baselines match verified values
    assert stats["Billing"]["count"] == 101
    assert stats["General"]["count"] == 122
    assert stats["Technical"]["count"] == 104

    assert stats["Billing"]["median"] == 11.4
    assert stats["General"]["median"] == 12.05
    assert stats["Technical"]["median"] == 13.15


def test_resolution_time_outliers_count():
    detector = AnomalyDetector()
    outliers = detector.detect_resolution_time_outliers()

    assert len(outliers) == 22

    billing_outliers = [o for o in outliers if o.category == "Billing"]
    general_outliers = [o for o in outliers if o.category == "General"]
    technical_outliers = [o for o in outliers if o.category == "Technical"]

    assert len(billing_outliers) == 5
    assert len(general_outliers) == 9
    assert len(technical_outliers) == 8


def test_sla_breaches_count():
    detector = AnomalyDetector()
    breaches = detector.detect_sla_breaches()

    assert len(breaches) == 80
    critical_breaches = [b for b in breaches if b.priority == "Critical"]
    high_breaches = [b for b in breaches if b.priority == "High"]

    assert len(critical_breaches) == 31
    assert len(high_breaches) == 49


def test_total_anomalies_and_summary():
    detector = AnomalyDetector()
    summary = detector.get_anomaly_summary()

    assert summary["total_anomalies"] == 102
    assert summary["resolution_time_outliers"] == 22
    assert summary["sla_breaches"] == 80
    assert summary["by_severity"]["CRITICAL"] == 40
    assert summary["by_severity"]["HIGH"] == 57
    assert summary["by_severity"]["MEDIUM"] == 5


def test_outlier_explanation_mentions_category_baseline():
    detector = AnomalyDetector()
    outliers = detector.detect_resolution_time_outliers()
    top = outliers[0]

    assert "General" in top.explanation
    assert "median" in top.explanation.lower()
    assert "Z-score" in top.explanation
    assert "Q3 + 1.5*IQR" in top.explanation


def test_per_category_anomaly_cross_tab():
    """
    Permanently asserts the exact cross-tab breakdown of anomalies by category and type:
    - Billing: 5 resolution outliers + 25 SLA breaches = 30 total
    - General: 9 resolution outliers + 33 SLA breaches = 42 total
    - Technical: 8 resolution outliers + 22 SLA breaches = 30 total
    - Total: 102 anomalies
    """
    detector = AnomalyDetector()
    res_outliers = detector.detect_resolution_time_outliers()
    sla_breaches = detector.detect_sla_breaches()

    cat_counts = {"Billing": {"outliers": 0, "sla": 0}, "General": {"outliers": 0, "sla": 0}, "Technical": {"outliers": 0, "sla": 0}}
    for o in res_outliers:
        cat_counts[o.category]["outliers"] += 1
    for b in sla_breaches:
        cat_counts[b.category]["sla"] += 1

    assert cat_counts["Billing"]["outliers"] == 5
    assert cat_counts["Billing"]["sla"] == 25
    assert cat_counts["Billing"]["outliers"] + cat_counts["Billing"]["sla"] == 30

    assert cat_counts["General"]["outliers"] == 9
    assert cat_counts["General"]["sla"] == 33
    assert cat_counts["General"]["outliers"] + cat_counts["General"]["sla"] == 42

    assert cat_counts["Technical"]["outliers"] == 8
    assert cat_counts["Technical"]["sla"] == 22
    assert cat_counts["Technical"]["outliers"] + cat_counts["Technical"]["sla"] == 30

    summary = detector.get_anomaly_summary()
    assert summary["by_category"]["Billing"] == 30
    assert summary["by_category"]["General"] == 42
    assert summary["by_category"]["Technical"] == 30
    assert summary["total_anomalies"] == 102
