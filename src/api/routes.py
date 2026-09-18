"""
FastAPI route definitions for support ticket querying, anomaly detection,
health checks, and dataset metrics.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from src.api.models import (
    AnomalyResponse,
    DatasetMetricsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.anomaly.detector import AnomalyDetector
from src.config import (
    get_reference_source,
    get_reference_timestamp_str,
    settings,
)
from src.data_layer.loader import get_cached_dataframe
from src.llm.client import LLMClient
from src.llm.query_engine import NLQueryEngine

logger = logging.getLogger(__name__)

router = APIRouter()
_query_engine: Optional[NLQueryEngine] = None
_anomaly_detector: Optional[AnomalyDetector] = None
_llm_client: Optional[LLMClient] = None


def get_query_engine() -> NLQueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = NLQueryEngine()
    return _query_engine


def get_anomaly_detector() -> AnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns system operational health, dataset status, dynamic reference timestamp,
    and LLM provider connectivity.
    """
    df = get_cached_dataframe()
    llm = get_llm_client()
    _, llm_status_msg = llm.check_availability()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        dataset_loaded=not df.empty,
        total_records=len(df),
        reference_now=get_reference_timestamp_str(),
        reference_now_source=get_reference_source(),
        llm_provider=settings.LLM_PROVIDER,
        llm_status=llm_status_msg,
    )


@router.post("/api/query", response_model=QueryResponse, tags=["Natural Language Query"])
def query_tickets(request: QueryRequest):
    """
    Translates a natural language question into safe SQL, executes it against
    the dataset, and returns structured data and an executive natural language answer.
    """
    engine = get_query_engine()
    result = engine.execute_nl_query(request.question)

    return QueryResponse(
        question=result["question"],
        sql=result["sql"],
        explanation=result.get("explanation"),
        results=result["results"],
        row_count=result["row_count"],
        answer=result["answer"],
        caveats=result.get("caveats"),
        execution_time_ms=result["execution_time_ms"],
        is_fallback=result["is_fallback"],
        reference_now=get_reference_timestamp_str(),
        error=result.get("error"),
    )


@router.get("/api/anomalies", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def get_anomalies(
    category: Optional[str] = Query(None, description="Filter by category: Billing, General, Technical"),
    severity: Optional[str] = Query(None, description="Filter by severity: CRITICAL, HIGH, MEDIUM"),
    anomaly_type: Optional[str] = Query(None, description="Filter by type: RESOLUTION_TIME_OUTLIER, SLA_BREACH"),
):
    """
    Detects and flags anomalies in customer support tickets:
    - Per-category statistical IQR outliers on resolution time.
    - SLA breaches (unresolved High/Critical tickets older than 24 hours).
    """
    detector = get_anomaly_detector()
    summary = detector.get_anomaly_summary()
    items = detector.detect_all_anomalies(
        category=category,
        severity=severity,
        anomaly_type=anomaly_type,
    )

    return AnomalyResponse(
        summary={
            "total_anomalies": summary["total_anomalies"],
            "resolution_time_outliers": summary["resolution_time_outliers"],
            "sla_breaches": summary["sla_breaches"],
            "by_severity": summary["by_severity"],
            "by_category": summary["by_category"],
        },
        category_thresholds=summary["category_thresholds"],
        total_count=len(items),
        anomalies=[i.to_dict() for i in items],
        reference_now=get_reference_timestamp_str(),
    )


@router.get("/api/metrics", response_model=DatasetMetricsResponse, tags=["Dataset Analytics"])
def get_dataset_metrics():
    """
    Returns high-level analytical KPIs, breakdowns, and averages across the dataset.
    """
    df = get_cached_dataframe()

    resolved_mask = df["resolution_time_hrs"].notna()
    resolved_df = df[resolved_mask]

    return DatasetMetricsResponse(
        total_tickets=len(df),
        resolved_tickets=int(resolved_mask.sum()),
        open_tickets=int((df["status"] == "Open").sum()),
        escalated_tickets=int((df["status"] == "Escalated").sum()),
        unresolved_tickets=int((~resolved_mask).sum()),
        priority_breakdown=df["priority"].value_counts().to_dict(),
        category_breakdown=df["category"].value_counts().to_dict(),
        avg_response_time_hrs=round(float(df["response_time_hrs"].mean()), 2),
        avg_resolution_time_hrs=round(float(resolved_df["resolution_time_hrs"].mean()), 2),
        avg_customer_rating=round(float(resolved_df["customer_rating"].mean()), 2),
        active_agents_count=int(df["agent_id"].nunique()),
        reference_now=get_reference_timestamp_str(),
    )
