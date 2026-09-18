"""
Data layer module for customer support ticket ingestion, transformation,
in-memory SQLite loading, and concurrency-safe query execution.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import (
    DEFAULT_CSV_PATH,
    get_reference_timestamp,
    set_reference_timestamp,
    settings,
)

logger = logging.getLogger(__name__)

# Global state for in-memory database master connection & cached dataframe
_MASTER_DB_CONN: Optional[sqlite3.Connection] = None
_CACHED_DF: Optional[pd.DataFrame] = None
DB_URI = "file:tickets_db?mode=memory&cache=shared"


def _sqlite_authorizer(action_code: int, arg1: Any, arg2: Any, db_name: Any, trigger_name: Any) -> int:
    """
    Strict SQLite engine authorizer callback.
    Permits only read operations (SELECT, READ, FUNCTION).
    Completely blocks DDL, DML modifications, PRAGMA, ATTACH, DETACH, etc.
    """
    # Allowed action codes in SQLite
    # 21: SQLITE_SELECT
    # 20: SQLITE_READ
    # 31: SQLITE_FUNCTION
    ALLOWED_ACTIONS = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }

    if action_code in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK

    # Log and deny any unauthorized database operation
    logger.warning(
        f"SQLite authorizer blocked operation: action_code={action_code}, arg1={arg1}, arg2={arg2}"
    )
    return sqlite3.SQLITE_DENY


def load_and_prepare_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ingests and cleans the support tickets CSV.
    - Decouples resolution outcome from status string (keyed on resolution_time_hrs IS NOT NULL).
    - Computes derived resolved_at timestamp rounded to minute precision (YYYY-MM-DD HH:MM).
    - Dynamically computes and registers REFERENCE_NOW = MAX(created_at, resolved_at).
    """
    global _CACHED_DF
    path = csv_path or settings.CSV_FILE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Support tickets dataset not found at: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    # Column name hygiene
    df.columns = [c.strip() for c in df.columns]
    required_cols = [
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "agent_id",
        "customer_rating",
        "issue_summary",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Parse timestamps
    df["created_at_dt"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M")
    df["created_at"] = df["created_at_dt"].dt.strftime("%Y-%m-%d %H:%M")

    # Clean numeric fields
    df["response_time_hrs"] = pd.to_numeric(df["response_time_hrs"], errors="coerce")
    df["resolution_time_hrs"] = pd.to_numeric(df["resolution_time_hrs"], errors="coerce")
    df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

    # Derived resolved_at timestamp (minute precision, rounded to nearest 60s)
    # Decoupled from status: any ticket with a non-null resolution_time_hrs has resolved_at
    def compute_resolved_at(row: pd.Series) -> Optional[datetime]:
        if pd.isna(row["resolution_time_hrs"]):
            return None
        res_seconds = row["resolution_time_hrs"] * 3600.0
        rounded_seconds = round(res_seconds / 60.0) * 60
        return row["created_at_dt"] + timedelta(seconds=rounded_seconds)

    df["resolved_at_dt"] = df.apply(compute_resolved_at, axis=1)
    df["resolved_at"] = df["resolved_at_dt"].dt.strftime("%Y-%m-%d %H:%M")

    # Dynamic calculation of REFERENCE_NOW = MAX(created_at, resolved_at)
    max_created = df["created_at_dt"].max()
    max_resolved = df["resolved_at_dt"].max()
    latest_event = max(max_created, max_resolved)

    # Identify which ticket set the latest event
    latest_ticket_id = "N/A"
    if max_resolved >= max_created:
        matching = df[df["resolved_at_dt"] == max_resolved]
        if not matching.empty:
            latest_ticket_id = matching.iloc[0]["ticket_id"]
    else:
        matching = df[df["created_at_dt"] == max_created]
        if not matching.empty:
            latest_ticket_id = matching.iloc[0]["ticket_id"]

    source_info = (
        f"Derived from dataset MAX(created_at, resolved_at) "
        f"[Event by {latest_ticket_id} on {latest_event.strftime('%Y-%m-%d %H:%M')}]"
    )
    set_reference_timestamp(latest_event, source_info=source_info)
    logger.info(
        f"Dataset loaded: {len(df)} rows. REFERENCE_NOW set to {latest_event.strftime('%Y-%m-%d %H:%M')} ({source_info})"
    )

    _CACHED_DF = df
    return df


def initialize_database(csv_path: Optional[Path] = None) -> None:
    """
    Initializes the shared in-memory SQLite database from the prepared DataFrame.
    Retains a master connection to keep the in-memory database alive in this process.
    """
    global _MASTER_DB_CONN
    df = load_and_prepare_data(csv_path)

    # Master connection: writeable to create schema and populate
    master_conn = sqlite3.connect(DB_URI, uri=True, check_same_thread=False)
    cursor = master_conn.cursor()

    # Drop existing table if any
    cursor.execute("DROP TABLE IF EXISTS tickets")

    # Create table with typed schema and derived resolved_at
    cursor.execute("""
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            response_time_hrs REAL NOT NULL,
            resolution_time_hrs REAL,
            agent_id TEXT NOT NULL,
            customer_rating REAL,
            issue_summary TEXT NOT NULL,
            resolved_at TEXT
        )
    """)

    # Prepare records for insertion
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row["ticket_id"]),
            str(row["created_at"]),
            str(row["category"]),
            str(row["priority"]),
            str(row["status"]),
            float(row["response_time_hrs"]),
            float(row["resolution_time_hrs"]) if pd.notna(row["resolution_time_hrs"]) else None,
            str(row["agent_id"]),
            float(row["customer_rating"]) if pd.notna(row["customer_rating"]) else None,
            str(row["issue_summary"]),
            str(row["resolved_at"]) if pd.notna(row["resolved_at"]) else None,
        ))

    cursor.executemany(
        """
        INSERT INTO tickets (
            ticket_id, created_at, category, priority, status,
            response_time_hrs, resolution_time_hrs, agent_id,
            customer_rating, issue_summary, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    # Build performance indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_resolved_at ON tickets (resolved_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets (category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets (priority)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets (status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_agent_id ON tickets (agent_id)")

    master_conn.commit()
    _MASTER_DB_CONN = master_conn
    logger.info("In-memory SQLite database initialized successfully with %d rows and indexes.", len(records))


def get_db_connection(read_only: bool = True) -> sqlite3.Connection:
    """
    Factory creating a thread-safe connection to the shared in-memory database.
    If read_only=True, attaches the strict SQLite engine authorizer.
    """
    global _MASTER_DB_CONN
    if _MASTER_DB_CONN is None:
        initialize_database()

    conn = sqlite3.connect(DB_URI, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if read_only:
        conn.set_authorizer(_sqlite_authorizer)

    return conn


def execute_query(sql: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Executes a validated read-only SQL query against the shared in-memory SQLite store.
    Returns: (list_of_row_dicts, column_names)
    """
    conn = get_db_connection(read_only=True)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        results = [dict(zip(columns, row)) for row in rows]
        return results, columns
    finally:
        conn.close()


def get_cached_dataframe() -> pd.DataFrame:
    """Returns the cached pandas DataFrame of the ingested dataset."""
    global _CACHED_DF
    if _CACHED_DF is None:
        return load_and_prepare_data()
    return _CACHED_DF
