"""
Anomaly Detection Engine for support tickets.
Provides defensible, explainable anomaly detection combining:
1. Per-category statistical IQR outlier detection on resolution times.
2. Rule-based SLA breach detection for aging unresolved High/Critical tickets.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_reference_timestamp, settings
from src.data_layer.loader import get_cached_dataframe

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRecord:
    ticket_id: str
    category: str
    priority: str
    status: str
    anomaly_type: str  # "RESOLUTION_TIME_OUTLIER" or "SLA_BREACH"
    severity: str      # "CRITICAL", "HIGH", "MEDIUM"
    metric_name: str
    metric_value: float
    threshold_value: float
    explanation: str
    created_at: str
    agent_id: str
    issue_summary: str
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetector:
    """
    Detects statistical resolution-time outliers and operational SLA breaches.
    Decoupled from status strings: keys off resolution_time_hrs IS NOT NULL / IS NULL.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else get_cached_dataframe()

    def get_category_resolution_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Computes descriptive statistics and IQR upper fences per category
        for all tickets with non-null resolution_time_hrs.
        """
        resolved_df = self.df[self.df["resolution_time_hrs"].notna()].copy()
        stats = {}

        for category, group in resolved_df.groupby("category"):
            res_times = group["resolution_time_hrs"].dropna().values
            if len(res_times) == 0:
                continue

            q1 = float(np.percentile(res_times, 25))
            q3 = float(np.percentile(res_times, 75))
            iqr = q3 - q1
            upper_fence = q3 + (settings.IQR_MULTIPLIER * iqr)
            extreme_fence = q3 + (3.0 * iqr)
            median = float(np.median(res_times))
            mean = float(np.mean(res_times))
            std = float(np.std(res_times, ddof=1)) if len(res_times) > 1 else 0.0

            stats[str(category)] = {
                "count": len(res_times),
                "q1": round(q1, 2),
                "median": round(median, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "upper_fence": round(upper_fence, 2),
                "extreme_fence": round(extreme_fence, 2),
                "mean": round(mean, 2),
                "std": round(std, 2),
            }

        return stats

    def detect_resolution_time_outliers(self) -> List[AnomalyRecord]:
        """
        Identifies tickets whose resolution time exceeds the category IQR upper fence.
        Generates explainable narrative referencing the category baseline.
        """
        cat_stats = self.get_category_resolution_stats()
        resolved_df = self.df[self.df["resolution_time_hrs"].notna()].copy()
        outliers: List[AnomalyRecord] = []

        for _, row in resolved_df.iterrows():
            cat = str(row["category"])
            if cat not in cat_stats:
                continue

            c_stats = cat_stats[cat]
            res_hrs = float(row["resolution_time_hrs"])
            upper_fence = c_stats["upper_fence"]

            if res_hrs > upper_fence:
                # Severity determination
                if res_hrs > c_stats["extreme_fence"]:
                    severity = "CRITICAL"
                elif res_hrs > (c_stats["q3"] + 2.0 * c_stats["iqr"]):
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                # Z-score and median ratio
                std = c_stats["std"]
                z_score = (res_hrs - c_stats["mean"]) / std if std > 0 else 0.0
                ratio_to_median = res_hrs / c_stats["median"] if c_stats["median"] > 0 else 0.0

                explanation = (
                    f"Resolution time of {res_hrs:.1f} hrs exceeds the {cat} category statistical "
                    f"threshold of {upper_fence:.2f} hrs (Q3 + 1.5*IQR). This is {ratio_to_median:.1f}x "
                    f"the category median ({c_stats['median']:.2f} hrs) with a Z-score of {z_score:.2f}."
                )

                outliers.append(
                    AnomalyRecord(
                        ticket_id=str(row["ticket_id"]),
                        category=cat,
                        priority=str(row["priority"]),
                        status=str(row["status"]),
                        anomaly_type="RESOLUTION_TIME_OUTLIER",
                        severity=severity,
                        metric_name="resolution_time_hrs",
                        metric_value=res_hrs,
                        threshold_value=upper_fence,
                        explanation=explanation,
                        created_at=str(row["created_at"]),
                        agent_id=str(row["agent_id"]),
                        issue_summary=str(row["issue_summary"]),
                        resolved_at=str(row["resolved_at"]) if pd.notna(row.get("resolved_at")) else None,
                    )
                )

        # Sort descending by resolution time
        outliers.sort(key=lambda x: x.metric_value, reverse=True)
        return outliers

    def detect_sla_breaches(self) -> List[AnomalyRecord]:
        """
        Identifies unresolved High and Critical tickets older than the SLA threshold (24h)
        relative to the dataset REFERENCE_NOW.
        Decoupled from status: defined by resolution_time_hrs IS NULL.
        """
        ref_now = get_reference_timestamp()
        sla_hours = settings.SLA_BREACH_HOURS

        # Unresolved tickets of High or Critical priority
        unresolved_mask = (
            self.df["resolution_time_hrs"].isna() &
            self.df["priority"].isin(["High", "Critical"])
        )
        candidates = self.df[unresolved_mask].copy()
        breaches: List[AnomalyRecord] = []

        for _, row in candidates.iterrows():
            created_dt = row["created_at_dt"]
            age_hours = (ref_now - created_dt).total_seconds() / 3600.0

            if age_hours > sla_hours:
                priority = str(row["priority"])
                severity = "CRITICAL" if priority == "Critical" else "HIGH"
                overdue_by = age_hours - sla_hours

                explanation = (
                    f"Unresolved {priority} priority ticket has been open for {age_hours:.1f} hrs "
                    f"(created {row['created_at']}), exceeding the {sla_hours:.0f}-hour SLA "
                    f"threshold by {overdue_by:.1f} hrs relative to dataset reference time ({ref_now.strftime('%Y-%m-%d %H:%M')})."
                )

                breaches.append(
                    AnomalyRecord(
                        ticket_id=str(row["ticket_id"]),
                        category=str(row["category"]),
                        priority=priority,
                        status=str(row["status"]),
                        anomaly_type="SLA_BREACH",
                        severity=severity,
                        metric_name="age_hours",
                        metric_value=round(age_hours, 1),
                        threshold_value=sla_hours,
                        explanation=explanation,
                        created_at=str(row["created_at"]),
                        agent_id=str(row["agent_id"]),
                        issue_summary=str(row["issue_summary"]),
                        resolved_at=None,
                    )
                )

        # Sort descending by overdue age
        breaches.sort(key=lambda x: x.metric_value, reverse=True)
        return breaches

    def detect_all_anomalies(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        anomaly_type: Optional[str] = None,
    ) -> List[AnomalyRecord]:
        """
        Returns all detected anomalies (resolution outliers + SLA breaches),
        with optional filtering by category, severity, or anomaly type.
        """
        anomalies = self.detect_resolution_time_outliers() + self.detect_sla_breaches()

        if category:
            anomalies = [a for a in anomalies if a.category.lower() == category.lower()]
        if severity:
            anomalies = [a for a in anomalies if a.severity.lower() == severity.lower()]
        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type.lower() == anomaly_type.lower()]

        return anomalies

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Generates aggregate anomaly summary statistics.
        """
        res_outliers = self.detect_resolution_time_outliers()
        sla_breaches = self.detect_sla_breaches()
        all_anomalies = res_outliers + sla_breaches

        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        category_counts = {"Billing": 0, "General": 0, "Technical": 0}

        for a in all_anomalies:
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
            category_counts[a.category] = category_counts.get(a.category, 0) + 1

        ref_now = get_reference_timestamp()

        return {
            "total_anomalies": len(all_anomalies),
            "resolution_time_outliers": len(res_outliers),
            "sla_breaches": len(sla_breaches),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "reference_now": ref_now.strftime("%Y-%m-%d %H:%M"),
            "category_thresholds": self.get_category_resolution_stats(),
        }
