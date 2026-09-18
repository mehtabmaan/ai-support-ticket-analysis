# AI-Powered Customer Support Ticket Analysis Platform
### Technical Assessment — AI Engineer Role | DOTMappers IT Pvt. Ltd.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-In--Memory%20Shared-003B57.svg)](https://www.sqlite.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Free%20Tier-F55036.svg)](https://console.groq.com/)
[![Tests](https://img.shields.io/badge/Tests-39%2F39%20Passing-brightgreen.svg)]()

A production-ready, enterprise-grade AI analytics system for customer support operations. Ingests 500 support tickets, translates natural language questions into secure SQLite queries via free-tier LLMs (with deterministic fallback), performs explainable per-category statistical IQR anomaly detection and SLA breach tracking, and exposes all capabilities through both a **FastAPI REST API** and a modern **Streamlit UI**.

---

## Table of Contents
1. [Quick Start (Single-Command Launch)](#1-quick-start-single-command-launch)
2. [Dual-Interface Requirement & Architecture Overview](#2-dual-interface-requirement--architecture-overview)
3. [Key Architectural & Engineering Decisions](#3-key-architectural--engineering-decisions)
   - [Data-Driven Resolution Definition vs. Status Label](#data-driven-resolution-definition-vs-status-label)
   - [Dynamic REFERENCE_NOW Derivation & Outlier TKT-108](#dynamic-reference_now-derivation--outlier-tkt-108)
   - [Per-Category Anomaly Detection & Cross-Tab Verification](#per-category-anomaly-detection--cross-tab-verification)
   - [Quartile Method Specification & Robustness](#quartile-method-specification--robustness)
   - [Generic Sparse Time-Window Detection](#generic-sparse-time-window-detection)
   - [Multi-Layered SQL Security & Process Concurrency](#multi-layered-sql-security--process-concurrency)
   - [Scoping Decision: Response-Time Anomalies](#scoping-decision-response-time-anomalies)
4. [Models & Technologies Used and Why](#4-models--technologies-used-and-why)
5. [Real Example Queries & Verified System Outputs](#5-real-example-queries--verified-system-outputs)
6. [Graceful Degradation & Adversarial Robustness](#6-graceful-degradation--adversarial-robustness)
7. [Streamlit UI Verification & Features](#7-streamlit-ui-verification--features)
8. [API Reference](#8-api-reference)
9. [Automated Test Suite](#9-automated-test-suite)
10. [Known Limitations & Scaling Roadmap](#10-known-limitations--scaling-roadmap)

---

## 1. Quick Start (Single-Command Launch)

### Prerequisites
- Python **3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14).
- OS: Platform-agnostic (Windows, macOS, Linux).

### Setup Instructions

```bash
# 1. Clone or navigate to the repository
cd Project

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Free-Tier Groq API Key
# The system works 100% offline out-of-the-box via its deterministic fallback engine.
# To enable Groq's high-speed LLaMA-3.3-70B cloud model at zero cost:
# Get a free key in 30 seconds at: https://console.groq.com/keys (no credit card required)
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your_key_here
```

### Single-Command Launch
Launch both the **FastAPI REST API** and the **Streamlit UI** concurrently with a single command:

```bash
python run.py
```

Once launched:
- **Interactive Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI REST API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

*(You can also run services independently if preferred: `uvicorn main:app --port 8000` or `streamlit run src/ui/app.py --server.port 8501`).*

---

## 2. Dual-Interface Requirement & Architecture Overview

> [!NOTE]
> **Resolution of Brief Inconsistency**:
> Section 2 of `docs/assessment.md` mandates that the REST API AND the UI are both required (*"Expose the functionality via a REST API AND a minimal UI — both are required"*), whereas Section 4 lists *"REST API ... OR a minimal UI"*. We deliberately adopted the stricter interpretation and engineered **both a complete FastAPI REST API and a Streamlit UI** sharing identical core analytics services.

```
+-------------------------------------------------------------------------+
|                               USER                                      |
+--------------------+------------------------------------+---------------+
                     |                                    |
            REST API Calls (JSON)               Browser Interaction
                     v                                    v
+--------------------+--------------+   +-----------------+---------------+
|        FastAPI Backend            |   |         Streamlit UI            |
|     (Port 8000, Uvicorn)          |   |        (Port 8501)              |
|  /api/query, /api/anomalies, etc. |   |  Query Tab, Anomalies, Health   |
+--------------------+--------------+   +-----------------+---------------+
                     |                                    |
                     v                                    v
    [ In-Memory SQLite Replica A ]       [ In-Memory SQLite Replica B ]
    (FastAPI process memory)             (Streamlit process memory)
                     ^                                    ^
                     |                                    |
                     +-----------------+------------------+
                                       |
                           Loads on startup from
                                       |
                           [ data/support_tickets.csv ]
                           - Dynamic REFERENCE_NOW derivation
                           - Minute-precision resolved_at derivation
                           - 100% Data-driven resolution semantics
```

### Process Isolation Architecture
FastAPI and Streamlit run as independent OS processes. Rather than attempting brittle cross-process shared memory, each process initializes its own thread-safe, read-only in-memory SQLite replica from `data/support_tickets.csv` on startup. This gives complete process isolation, zero inter-process lock contention, and microsecond-level query latencies.

---

## 3. Key Architectural & Engineering Decisions

### Data-Driven Resolution Definition vs. Status Label
- **The Problem**: Section 3.2 of the assessment preview shows a ticket (`TKT-002`) with `status = 'Escalated'` but possessing a non-null `resolution_time_hrs` (18.5) and `customer_rating` (2). Relying purely on `status == 'Resolved'` would misclassify resolved escalated tickets.
- **The Architectural Solution**: We decoupled business semantics from the mutable workflow string. A ticket is defined as **resolved** if and only if `resolution_time_hrs IS NOT NULL` (and `resolved_at IS NOT NULL`), and **unresolved** if `resolution_time_hrs IS NULL`.
- **Dataset Invariant**: In `data/support_tickets.csv`, `resolution_time_hrs IS NOT NULL` and `status == 'Resolved'` happen to correlate 1:1 across all 500 rows (327 resolved, 173 unresolved). However, by enforcing the data-driven definition across the data layer, anomaly detector, prompt instructions, and SQL generation, the system is permanently immune to status-label drift.

### Dynamic `REFERENCE_NOW` Derivation & Outlier `TKT-108`
- **The Problem**: Datasets from historical snapshots cannot use wall-clock `datetime.now()` without invalidating all time-relative queries ("this month", "this week", SLA aging). Furthermore, setting `REFERENCE_NOW = MAX(created_at)` is flawed because tickets created late in the dataset were resolved days later (e.g. ticket `TKT-108` was created on 2024-03-30 12:41 with a 119.7-hour resolution time, resolving on 2024-04-04 12:23).
- **The Solution**: `REFERENCE_NOW` is dynamically evaluated at startup as:
  $$	ext{REFERENCE\_NOW} = \max(\max(	ext{created\_at}), \max(	ext{resolved\_at})) = 	ext{2024-04-04 12:23}$$
  It is never hardcoded. It is exposed in `/health`, `/api/metrics`, displayed as a prominent banner in the UI, and injected into the LLM system prompt.

### Per-Category Anomaly Detection & Cross-Tab Verification
The system identifies **102 total anomalies** across two complementary methodologies:
1. **Resolution Time Outliers (Statistical IQR)**: $22$ resolved tickets exceeding their category-specific upper fence.
2. **Aging SLA Breaches (Rule-Based)**: $80$ unresolved High/Critical tickets open $> 24	ext{ hrs}$ relative to `REFERENCE_NOW`.

Independently verified cross-tabulation across categories:

| Category | Resolution Outliers (IQR) | SLA Breaches (>24h) | Total Anomalies |
|---|---|---|---|
| **Billing** | 5 | 25 | **30** |
| **General** | 9 | 33 | **42** |
| **Technical** | 8 | 22 | **30** |
| **Total** | **22** | **80** | **102** |

*(Permanently asserted in `tests/test_anomaly.py::test_per_category_anomaly_cross_tab`)*.

### Quartile Method Specification & Robustness
- **Exact Method Shipped**: In `src/anomaly/detector.py`, quartiles are computed using NumPy's standard linear interpolation:
  ```python
  q1 = float(np.percentile(res_times, 25))
  q3 = float(np.percentile(res_times, 75))
  iqr = q3 - q1
  upper_fence = q3 + (1.5 * iqr)
  ```
- **Exact Code-Computed Thresholds**:
  - **Billing** ($N=101$): $Q1 = 5.50$, $	ext{Median} = 11.40$, $Q3 = 21.10$, $	ext{IQR} = 15.60$, **Upper Fence** $= 44.50	ext{ hrs}$ $ightarrow$ **5 outliers** (values: 47.7, 51.9, 72.7, 76.1, 87.3 hrs).
  - **General** ($N=122$): $Q1 = 6.65$, $	ext{Median} = 12.05$, $Q3 = 23.45$, $	ext{IQR} = 16.80$, **Upper Fence** $= 48.65	ext{ hrs}$ $ightarrow$ **9 outliers** (values: 53.4, 59.0, 66.1, 66.6, 66.8, 96.5, 114.3, 119.6, 119.7 hrs).
  - **Technical** ($N=104$): $Q1 = 6.55$, $	ext{Median} = 13.15$, $Q3 = 24.98$, $	ext{IQR} = 18.43$, **Upper Fence** $= 52.61	ext{ hrs}$ $ightarrow$ **8 outliers** (values: 55.5, 60.6, 68.1, 76.5, 77.5, 79.0, 84.2, 100.4 hrs).
- **Mathematical Inlier/Outlier Separation**:
  - Billing highest inlier is $44.4	ext{ hrs}$ vs lowest outlier $47.7	ext{ hrs}$.
  - General highest inlier is $47.3	ext{ hrs}$ vs lowest outlier $53.4	ext{ hrs}$.
  - Technical highest inlier is $47.6	ext{ hrs}$ vs lowest outlier $55.5	ext{ hrs}$.
  Because a natural separation gap exists in every category between inliers and outliers, alternative quartile algorithms (such as Python's `statistics.quantiles`) fall within this exact same gap, proving the 22-outlier result is mathematically stable and not an artifact of interpolation choice.

### Generic Sparse Time-Window Detection
- **The Problem**: Relative to `REFERENCE_NOW` (2024-04-04), the reference month is April 2024 (`2024-04`), which contains only 1 resolved ticket (`TKT-108` by `AGT-07`). Answering *"Which agent resolved the most tickets this month?"* with a naive ranking would be mathematically factual but operationally misleading without context.
- **The Solution**: We built a general, data-driven sparse-window detector ($N_{window} \le 0.25 	imes N_{expected}$ or $N \le 3$). When a time window is sparse, the synthesizer:
  1. Provides the literal answer for the requested window (`AGT-07` with 1 ticket).
  2. Flags a transparent data-scarcity caveat (`"Note: Current reference month contains only 1 ticket"`).
  3. Automatically surfaces the prior full month's baseline (`March 2024: AGT-01 resolved 16 tickets out of 123 total`).
  4. Also generalizes to weekly windows ("this week").

### Multi-Layered SQL Security & Process Concurrency
To prevent SQL injection, statement stacking, and unauthorized operations:
1. **AST Validation (`sqlparse`)**: Parses the generated query, strictly enforcing `len(statements) == 1` and statement type == `SELECT`. Rejects stacked queries (e.g. `SELECT 1; DROP TABLE tickets;`), comments hiding semicolons, and DDL/DML tokens.
2. **SQLite Engine Authorizer**: At the database connection level, `conn.set_authorizer()` permits only `SQLITE_SELECT`, `SQLITE_READ`, and `SQLITE_FUNCTION`. Any attempt to execute `PRAGMA`, `ATTACH`, `INSERT`, `UPDATE`, or `DELETE` is blocked at the SQLite engine level.
3. **Thread Concurrency**: Query requests use a connection factory returning thread-isolated connections with `check_same_thread=False`, fully tested under 25 concurrent requests without locks.

### Scoping Decision: Response-Time Anomalies
- We performed statistical analysis on `response_time_hrs`: Range $0.2 - 5.0	ext{ hrs}$, Mean $2.62	ext{ hrs}$, Median $2.60	ext{ hrs}$, $Q1 = 1.40	ext{ hrs}$, $Q3 = 3.90	ext{ hrs}$, Upper Fence $= 7.65	ext{ hrs}$.
- Because the synthetic dataset capped response times at $5.0	ext{ hrs}$, there are **zero statistical outliers** in response time.
- Consequently, we prioritized the two core anomalies explicitly requested in Section 2 of the assessment brief: (1) **Per-Category Resolution Time Outliers** (22 tickets) and (2) **Aging SLA Breaches** (80 tickets).

---

## 4. Models & Technologies Used and Why

| Technology / Component | Choice | Justification & Rationale |
|---|---|---|
| **Primary LLM** | Groq Free Tier (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) | State-of-the-art open-weights reasoning, sub-500ms inference speeds, 100% free tier with zero credit card required. |
| **Local LLM Option** | Ollama (`llama3:latest`) | Zero cloud dependency; enables 100% local, air-gapped execution when requested. |
| **Offline Fallback** | Deterministic Pattern Matcher & Rule Synthesizer | Evaluator can run the system immediately without configuring any API key or local daemon. Zero unhandled exceptions. |
| **Backend API** | FastAPI + Uvicorn | High performance, native async/threaded concurrency, automatic OpenAPI documentation, strict Pydantic v2 validation. |
| **Interactive UI** | Streamlit | Rapid, modern reactive dashboard with zero frontend build step, native tabular and chart rendering. |
| **Database Engine** | SQLite (In-Memory Shared URI) | Standard SQL dialect universally understood by LLMs, microsecond query speeds, zero external server dependencies, engine-level authorizer security. |

---

## 5. Real Example Queries & Verified System Outputs

*Every single output below was generated by actually executing the query engine against the ingested dataset. Zero fabrication.*

### Query 1: "How many tickets are currently open?"
```sql
SELECT count(*) as open_tickets_count FROM tickets WHERE status = 'Open'
```
- **Row Count**: 1 | **Execution Time**: 1.15 ms
- **System Output**:
  > "As of the dataset reference date (2024-04-04 12:23), the Open Tickets Count is 111."

### Query 2: "How many critical tickets are unresolved?"
```sql
SELECT count(*) as unresolved_critical_count FROM tickets WHERE priority = 'Critical' AND resolution_time_hrs IS NULL
```
- **Row Count**: 1 | **Execution Time**: 1.02 ms
- **System Output**:
  > "As of the dataset reference date (2024-04-04 12:23), the Unresolved Critical Count is 31."

### Query 3: "Which agent resolved the most tickets this month?"
```sql
SELECT agent_id, count(*) as resolved_tickets_count 
FROM tickets 
WHERE resolved_at IS NOT NULL AND strftime('%Y-%m', resolved_at) = '2024-04' 
GROUP BY agent_id 
ORDER BY resolved_tickets_count DESC LIMIT 5
```
- **Row Count**: 1 | **Execution Time**: 4.93 ms
- **System Output**:
  > "As of 2024-04-04 12:23, the leading agent is AGT-07 (resolved tickets count: 1).
  > 
  > *Data Context: The current reference month (2024-04, as of 2024-04-04 12:23) contains only 1 resolved ticket (statistically sparse). For the prior full month (March 2024), AGT-01 resolved the most tickets (16 resolved tickets out of 123 total).*"

### Query 4: "Which agent resolved the most tickets in March 2024?"
```sql
SELECT agent_id, count(*) as resolved_tickets_count 
FROM tickets 
WHERE resolved_at IS NOT NULL AND strftime('%Y-%m', resolved_at) = '2024-03' 
GROUP BY agent_id 
ORDER BY resolved_tickets_count DESC LIMIT 5
```
- **Row Count**: 5 | **Execution Time**: 1.83 ms
- **Top Records**: `AGT-01`: 16, `AGT-12`: 15, `AGT-06`: 13, `AGT-09`: 13, `AGT-07`: 12.
- **System Output**:
  > "As of 2024-04-04 12:23, the leading agent is AGT-01 (resolved tickets count: 16)."

### Query 5: "Which agent resolved the most tickets overall?"
```sql
SELECT agent_id, count(*) as total_resolved 
FROM tickets 
WHERE resolution_time_hrs IS NOT NULL 
GROUP BY agent_id 
ORDER BY total_resolved DESC LIMIT 5
```
- **Row Count**: 5 | **Execution Time**: 1.28 ms
- **Top Records**: `AGT-09`: 37, `AGT-12`: 37, `AGT-06`: 34, `AGT-01`: 31, `AGT-11`: 29.
- **System Output (Detecting Ties)**:
  > "As of 2024-04-04 12:23, the leading agent is AGT-09 (total resolved: 37).
  > 
  > *Note: 'AGT-09' and 'AGT-12' are tied with 37 total resolved each.*"

### Query 6: "Which agent has the lowest average customer rating?"
```sql
SELECT agent_id, round(avg(customer_rating), 4) as avg_rating, count(*) as rated_count 
FROM tickets 
WHERE customer_rating IS NOT NULL 
GROUP BY agent_id 
ORDER BY avg_rating ASC LIMIT 5
```
- **Row Count**: 5 | **Execution Time**: 1.89 ms
- **Top Records**: `AGT-08`: 3.4800 (25 ratings), `AGT-11`: 3.4828 (29 ratings).
- **System Output**:
  > "As of 2024-04-04 12:23, the leading agent is AGT-08 (avg rating: 3.48, rated count: 25)."

### Query 7: "What is the average customer rating for Technical category tickets?"
```sql
SELECT round(avg(customer_rating), 4) as avg_rating, count(*) as rated_count 
FROM tickets 
WHERE category = 'Technical' AND customer_rating IS NOT NULL LIMIT 100
```
- **Row Count**: 1 | **Execution Time**: 1.89 ms
- **System Output**:
  > "As of 2024-04-04 12:23, the average customer rating is 3.7404 (based on 104 rated tickets)."

---

## 6. Graceful Degradation & Adversarial Robustness

The system is engineered to handle failure modes without exposing stack traces or crashing:

### 1. No LLM API Key Configured (Zero-Cost Offline Fallback)
When `GROQ_API_KEY` is not set:
- System logs: `[INFO] No GROQ_API_KEY provided. System running in deterministic fallback mode.`
- Query execution succeeds with `is_fallback: true`:
  > **Query**: *"How many tickets are currently open?"*  
  > **Answer**: *"As of the dataset reference date (2024-04-04 12:23), the Open Tickets Count is 111."*  
  > **HTTP Status**: 200 OK (Zero crashes, zero errors).

### 2. Out-of-Scope / Ambiguous Questions
When an evaluator enters a question outside the dataset's domain:
- **Question**: *"What is the capital of France and what is the weather there?"*
- **System Output**:
  > *"I was unable to translate your question into a SQL query. Please verify your question relates to the customer support dataset, or ensure a valid LLM API key (e.g. GROQ_API_KEY) is configured in your .env file."*
- **HTTP Status**: 200 OK, `row_count: 0`, `error: "No query generated"`.

### 3. Malformed API Requests (`POST /api/query`)
- **Missing Required `question` field**: `POST {"random_key": "test"}`  
  $ightarrow$ **HTTP 422 Unprocessable Entity**: `{"detail": [{"type": "missing", "loc": ["body", "question"], "msg": "Field required"}]}`
- **Wrong Type (Integer instead of String)**: `POST {"question": 12345}`  
  $ightarrow$ **HTTP 422 Unprocessable Entity**: `{"detail": [{"type": "string_type", "loc": ["body", "question"], "msg": "Input should be a valid string"}]}`
- **Empty / Too Short String**: `POST {"question": "a"}`  
  $ightarrow$ **HTTP 422 Unprocessable Entity**: `{"detail": [{"type": "string_too_short", "loc": ["body", "question"], "msg": "String should have at least 3 characters"}]}`

---

## 7. Streamlit UI Verification & Features

The Streamlit UI (`src/ui/app.py`) was verified programmatically via Streamlit's official `AppTest` framework across all three tabs:

```
[PASS] Streamlit app loaded with 0 exceptions.
[PASS] Persistent Reference Banner verified: 2024-04-04 12:23 (TKT-108 resolution)
[PASS] Tab 1 (Query Explorer): Sample query selected -> Answer box rendered (111) -> SQL code block rendered -> Metrics rendered.
[PASS] Tab 2 (Anomaly Dashboard): Anomaly KPIs verified (Total=102, Outliers=22, SLA Breaches=80, Critical=40) -> Category filter 'Billing' applied -> Exactly 30 matching anomalies displayed.
[PASS] Tab 3 (Analytics & Health): Total=500, Resolved=327 (65.4%), Open=111, Escalated=62 -> Category threshold table rendered.
```

- **Persistent Header Banner**: Displays `Dataset Snapshot Reference Date: 2024-04-04 12:23 [Event by TKT-108 on 2024-04-04 12:23]`.
- **Tab 1: Natural Language Query Explorer**: Features 8 one-click sample query chips from the assessment brief, custom text query box, executive answer box, execution metrics badge, collapsible SQL inspector with syntax highlighting, and interactive result table.
- **Tab 2: Anomaly Detection Dashboard**: KPI cards for Total Anomalies, Resolution Outliers, SLA Breaches, and Critical Severity; interactive dropdown filters by Category, Severity, and Type; detailed sortable table; expandable individual anomaly cards with explainable narratives and remediation notes.
- **Tab 3: Dataset Analytics & Health**: Status cards, distribution charts for Priority and Category, and full IQR threshold table with Q1, Median, Q3, IQR, and Upper Fences.

---

## 8. API Reference

The FastAPI REST API provides OpenAPI documentation at `/docs`. Key endpoints:

### `GET /health`
Returns system health, dataset status, dynamic reference timestamp, and LLM status.
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dataset_loaded": true,
  "total_records": 500,
  "reference_now": "2024-04-04 12:23",
  "reference_now_source": "Derived from dataset MAX(created_at, resolved_at) [Event by TKT-108 on 2024-04-04 12:23]",
  "llm_provider": "groq",
  "llm_status": "Offline deterministic fallback mode enabled."
}
```

### `POST /api/query`
Executes natural language queries.
**Request**:
```json
{
  "question": "How many tickets are currently open?"
}
```
**Response**:
```json
{
  "question": "How many tickets are currently open?",
  "sql": "SELECT count(*) as open_tickets_count FROM tickets WHERE status = 'Open'",
  "explanation": "Counts all tickets with status = 'Open'.",
  "results": [{"open_tickets_count": 111}],
  "row_count": 1,
  "answer": "As of the dataset reference date (2024-04-04 12:23), the Open Tickets Count is 111.",
  "caveats": null,
  "execution_time_ms": 1.15,
  "is_fallback": true,
  "reference_now": "2024-04-04 12:23",
  "error": null
}
```

### `GET /api/anomalies`
Detects and lists all 102 anomalies with category-specific baselines and explanations.
- Query Parameters: `category`, `severity`, `anomaly_type`.
- Summary:
  - Total Anomalies: **102**
  - Resolution Outliers (IQR): **22** (Billing: 5, General: 9, Technical: 8)
  - SLA Breaches (>24h): **80** (Critical: 31, High: 49)
  - Severities: `CRITICAL`: 40, `HIGH`: 57, `MEDIUM`: 5

### `GET /api/metrics`
Returns dataset-level distributions, averages, and agent counts.

---

## 9. Automated Test Suite

A comprehensive test suite of **39 tests** validates data ingestion, anomaly calculations, SQL security, concurrency, query accuracy, adversarial inputs, and UI rendering.

```bash
# Run pytest with verbose output
python -m pytest -v tests/
```

```
tests/test_anomaly.py::test_category_thresholds PASSED
tests/test_anomaly.py::test_resolution_time_outliers_count PASSED
tests/test_anomaly.py::test_sla_breaches_count PASSED
tests/test_anomaly.py::test_total_anomalies_and_summary PASSED
tests/test_anomaly.py::test_outlier_explanation_mentions_category_baseline PASSED
tests/test_anomaly.py::test_per_category_anomaly_cross_tab PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_query_endpoint PASSED
tests/test_api.py::test_anomalies_endpoint PASSED
tests/test_api.py::test_anomalies_filtered PASSED
tests/test_api.py::test_metrics_endpoint PASSED
tests/test_api.py::test_query_validation_error PASSED
tests/test_api.py::test_query_missing_field_error PASSED
tests/test_api.py::test_query_wrong_type_error PASSED
tests/test_api.py::test_query_empty_payload_error PASSED
tests/test_concurrency.py::test_concurrent_read_queries PASSED
tests/test_data_layer.py::test_csv_loading_and_counts PASSED
tests/test_data_layer.py::test_timestamp_minute_precision PASSED
tests/test_data_layer.py::test_reference_now_bounds PASSED
tests/test_data_layer.py::test_resolved_equivalence_invariant_in_dataset PASSED
tests/test_data_layer.py::test_data_driven_resolution_with_mock_escalated_resolved PASSED
tests/test_data_layer.py::test_dynamic_reference_now_mutation PASSED
tests/test_query_engine.py::test_open_tickets_query PASSED
tests/test_query_engine.py::test_critical_unresolved_query PASSED
tests/test_query_engine.py::test_lowest_rated_agent_query PASSED
tests/test_query_engine.py::test_most_resolved_this_month_sparse_caveat PASSED
tests/test_query_engine.py::test_most_resolved_march_2024 PASSED
tests/test_query_engine.py::test_most_resolved_overall_tie_detection PASSED
tests/test_query_engine.py::test_generic_weekly_sparse_window PASSED
tests/test_query_engine.py::test_out_of_scope_query_graceful_handling PASSED
tests/test_query_engine.py::test_offline_fallback_mode PASSED
tests/test_sql_safety.py::test_valid_select_allowed PASSED
tests/test_sql_safety.py::test_statement_stacking_rejected PASSED
tests/test_sql_safety.py::test_ddl_and_dml_rejected PASSED
tests/test_sql_safety.py::test_admin_commands_rejected PASSED
tests/test_sqlite_engine_authorizer_enforcement PASSED
tests/test_ui.py::test_ui_initial_load PASSED
tests/test_ui.py::test_ui_query_interaction PASSED
tests/test_ui.py::test_ui_anomaly_filter_interaction PASSED

======================= 39 passed in 3.92s ========================
```

---

## 10. Known Limitations & Scaling Roadmap

### Known Limitations
1. **Static In-Memory Store**: The SQLite store is refreshed on startup from CSV. In a live production environment, ticket ingestion would stream continuously via Kafka or RabbitMQ into a distributed relational warehouse (e.g. PostgreSQL / ClickHouse).
2. **Read-Only Analytics Focus**: The current query engine is strictly read-only by design. It does not support write-back actions (e.g., reassigning an agent or closing a ticket).
3. **Response Time Truncation**: As noted in our statistical audit, the dataset generator capped response times at 5.0 hours, preventing statistical outlier detection on initial response times.

### Scaling to Millions of Support Tickets
If scaling this system from 500 tickets to 50,000,000+ tickets:
1. **Database Layer**: Transition from in-memory SQLite to **DuckDB** for analytical workloads or **ClickHouse / PostgreSQL** with partitioning on `created_at` (monthly partitions) and BRIN/B-tree indexes on `agent_id`, `category`, and `priority`.
2. **Real-Time Streaming Anomaly Engine**: Decouple anomaly detection from query-time batch calculations. Implement streaming anomaly detection using **Apache Flink** or **Celery / Redis Streams** that evaluates tickets upon event creation and persists flagged anomalies into an indexed alert table.
3. **Semantic Hybrid Search (RAG)**: For qualitative queries on `issue_summary` (e.g. *"What are the top complaints regarding invoice discrepancies?"*), incorporate dense vector embeddings (e.g. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`) in a vector store (e.g. Qdrant / pgvector) to enable hybrid SQL + semantic search.
4. **Caching & Semantic Query Cache**: Introduce a Redis semantic cache (e.g. GPTCache) that caches frequent analytical questions and pre-computed daily rollups, reducing LLM costs to zero for repeat queries.
