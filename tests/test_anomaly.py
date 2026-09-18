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
