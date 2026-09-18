"""
Pydantic schemas for API request and response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about the customer support ticket dataset.",
        examples=["How many tickets are currently open?", "Which agent has the lowest average customer rating?"]
    )


class QueryResponse(BaseModel):
    question: str
    sql: Optional[str]
    explanation: Optional[str]
    results: List[Dict[str, Any]]
    row_count: int
    answer: str
    caveats: Optional[str] = None
    execution_time_ms: float
    is_fallback: bool
    reference_now: str
    error: Optional[str] = None


class AnomalyItem(BaseModel):
    ticket_id: str
    category: str
    priority: str
    status: str
    anomaly_type: str
    severity: str
    metric_name: str
    metric_value: float
    threshold_value: float
    explanation: str
    created_at: str
    agent_id: str
    issue_summary: str
    resolved_at: Optional[str] = None


class CategoryThreshold(BaseModel):
    count: int
    q1: float
    median: float
    q3: float
    iqr: float
    upper_fence: float
    extreme_fence: float
    mean: float
    std: float


class AnomalyResponse(BaseModel):
    summary: Dict[str, Any]
    category_thresholds: Dict[str, CategoryThreshold]
    total_count: int
    anomalies: List[AnomalyItem]
    reference_now: str


class HealthResponse(BaseModel):
    status: str
    version: str
    dataset_loaded: bool
    total_records: int
    reference_now: str
    reference_now_source: str
    llm_provider: str
    llm_status: str


class DatasetMetricsResponse(BaseModel):
    total_tickets: int
    resolved_tickets: int
    open_tickets: int
    escalated_tickets: int
    unresolved_tickets: int
    priority_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    avg_response_time_hrs: float
    avg_resolution_time_hrs: float
    avg_customer_rating: float
    active_agents_count: int
    reference_now: str
