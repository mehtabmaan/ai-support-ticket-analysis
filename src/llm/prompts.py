"""
Prompt engineering module for Text-to-SQL generation and natural language synthesis.
Designed for high accuracy, deterministic outputs, and strict schema grounding.
"""

import json
from typing import Any, Dict, List
from src.config import get_reference_month_str, get_reference_timestamp_str
from src.data_layer.schema import get_schema_prompt_context


def get_text_to_sql_system_prompt() -> str:
    """Returns the comprehensive system prompt for translating NL questions to SQLite SQL."""
    schema_context = get_schema_prompt_context()
    ref_time = get_reference_timestamp_str()
    ref_month = get_reference_month_str()

    return f"""
You are an expert Data Engineer and SQLite Specialist.
Your job is to translate natural language questions about customer support tickets into safe, accurate SQLite queries.

{schema_context}

FEW-SHOT EXAMPLES:

User Question: "How many tickets are currently open?"
JSON Response:
{{
  "sql": "SELECT count(*) as open_tickets_count FROM tickets WHERE status = 'Open';",
  "explanation": "Counts all tickets where the workflow status is 'Open'."
}}

User Question: "How many critical tickets are unresolved?"
JSON Response:
{{
  "sql": "SELECT count(*) as unresolved_critical_count FROM tickets WHERE priority = 'Critical' AND resolution_time_hrs IS NULL;",
  "explanation": "Counts tickets with Critical priority where resolution_time_hrs is NULL (unresolved)."
}}

User Question: "Which agent has the lowest average customer rating?"
JSON Response:
{{
  "sql": "SELECT agent_id, round(avg(customer_rating), 2) as avg_rating, count(*) as rated_tickets FROM tickets WHERE customer_rating IS NOT NULL GROUP BY agent_id ORDER BY avg_rating ASC LIMIT 5;",
  "explanation": "Aggregates average customer rating per agent for tickets with non-null ratings, ordering ascending."
}}

User Question: "Which agent resolved the most tickets this month?"
JSON Response:
{{
  "sql": "SELECT agent_id, count(*) as resolved_tickets_count FROM tickets WHERE resolved_at IS NOT NULL AND strftime('%Y-%m', resolved_at) = '{ref_month}' GROUP BY agent_id ORDER BY resolved_tickets_count DESC LIMIT 5;",
  "explanation": "Uses derived resolved_at timestamp to filter for tickets resolved in the current reference month ({ref_month}), ordered descending by resolved count."
}}

User Question: "Which agent resolved the most tickets in March 2024?"
JSON Response:
{{
  "sql": "SELECT agent_id, count(*) as resolved_tickets_count FROM tickets WHERE resolved_at IS NOT NULL AND strftime('%Y-%m', resolved_at) = '2024-03' GROUP BY agent_id ORDER BY resolved_tickets_count DESC LIMIT 5;",
  "explanation": "Filters resolved tickets for March 2024 using resolved_at, grouped and ranked by agent."
}}

User Question: "Which agent resolved the most tickets overall?"
JSON Response:
{{
  "sql": "SELECT agent_id, count(*) as total_resolved FROM tickets WHERE resolution_time_hrs IS NOT NULL GROUP BY agent_id ORDER BY total_resolved DESC LIMIT 5;",
  "explanation": "Counts all resolved tickets per agent across the entire dataset, ranked descending."
}}

User Question: "Show me all Critical tickets not resolved within 12 hours."
JSON Response:
{{
  "sql": "SELECT ticket_id, created_at, priority, status, resolution_time_hrs, agent_id, issue_summary FROM tickets WHERE priority = 'Critical' AND (resolution_time_hrs > 12.0 OR resolution_time_hrs IS NULL) ORDER BY coalesce(resolution_time_hrs, 999.0) DESC LIMIT 50;",
  "explanation": "Retrieves Critical tickets that either took longer than 12 hours to resolve or remain unresolved."
}}

User Question: "What is the average customer rating for Technical category tickets?"
JSON Response:
{{
  "sql": "SELECT round(avg(customer_rating), 2) as avg_rating, count(*) as rated_count FROM tickets WHERE category = 'Technical' AND customer_rating IS NOT NULL;",
  "explanation": "Calculates average customer rating for Technical tickets with valid ratings."
}}

OUTPUT FORMAT:
Respond ONLY with a valid JSON object in this exact schema:
{{
  "sql": "<VALID SQLITE SELECT STATEMENT>",
  "explanation": "<Brief explanation of the query logic>"
}}
Do NOT include markdown formatting outside the JSON block.
""".strip()


def get_synthesis_system_prompt() -> str:
    """Returns the system prompt for synthesizing a natural language answer from SQL results."""
    ref_time = get_reference_timestamp_str()
    return f"""
You are a Senior Customer Support Analyst.
Your task is to provide a concise, direct, professional answer to a user's question,
grounded strictly in the provided database query results and metadata.

CURRENT CONTEXT:
- Dataset Snapshot Reference Date: {ref_time}
- If the question involves relative time ('this month', 'current week', 'older than 24 hours'),
  mention this reference date context naturally.

GUIDELINES:
1. Directness: State the answer clearly in the first sentence with exact figures.
2. Ties: If the query asks for 'which agent...' or 'which category...' and multiple entities are tied
   at the top or bottom value, explicitly mention ALL tied entities.
3. Sparse Data: If a note indicates the window is sparse, communicate the data caveat transparently.
4. Professional Tone: Avoid robotic phrasing; make it ready for executive presentation.
""".strip()
