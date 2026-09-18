"""
Schema documentation module for support tickets.
Provides structured schema metadata and comprehensive prompt-grounding text
for Text-to-SQL generation.
"""

from src.config import get_reference_timestamp_str, get_reference_month_str

SCHEMA_METADATA = {
    "table_name": "tickets",
    "columns": {
        "ticket_id": {"type": "TEXT", "description": "Unique ticket identifier (e.g. 'TKT-001')"},
        "created_at": {"type": "TEXT", "description": "Timestamp when ticket was created (format 'YYYY-MM-DD HH:MM')"},
        "category": {"type": "TEXT", "description": "Issue category: 'Billing', 'Technical', 'General'"},
        "priority": {"type": "TEXT", "description": "Urgency level: 'Low', 'Medium', 'High', 'Critical'"},
        "status": {"type": "TEXT", "description": "Workflow status: 'Open', 'Escalated', 'Resolved'"},
        "response_time_hrs": {"type": "REAL", "description": "Hours from creation to first agent response"},
        "resolution_time_hrs": {"type": "REAL", "description": "Hours from creation to resolution (NULL for unresolved tickets)"},
        "agent_id": {"type": "TEXT", "description": "Support agent identifier ('AGT-01' through 'AGT-12')"},
        "customer_rating": {"type": "REAL", "description": "Satisfaction rating 1 to 5 (NULL for unresolved tickets)"},
        "issue_summary": {"type": "TEXT", "description": "Brief free-text summary of customer issue"},
        "resolved_at": {"type": "TEXT", "description": "Derived timestamp when ticket was resolved ('YYYY-MM-DD HH:MM'). NULL if unresolved."}
    }
}


def get_schema_prompt_context() -> str:
    """
    Builds the rich schema grounding text injected into the LLM system prompt.
    Includes rules for verb disambiguation, reference time grounding, and SQLite dialect.
    """
    ref_time = get_reference_timestamp_str()
    ref_month = get_reference_month_str()

    return f"""
DATABASE SCHEMA (SQLite dialect):
Table: tickets
Columns:
- ticket_id TEXT PRIMARY KEY: Unique ticket identifier (e.g. 'TKT-001')
- created_at TEXT: Creation timestamp (format 'YYYY-MM-DD HH:MM')
- category TEXT: Issue category ('Billing', 'Technical', 'General')
- priority TEXT: Ticket priority ('Low', 'Medium', 'High', 'Critical')
- status TEXT: Ticket status ('Open', 'Escalated', 'Resolved')
- response_time_hrs REAL: Hours to first response (always present)
- resolution_time_hrs REAL: Hours to resolution (NULL for unresolved tickets)
- agent_id TEXT: Assigned agent ('AGT-01' to 'AGT-12')
- customer_rating REAL: Customer rating 1 to 5 (NULL for unresolved tickets)
- issue_summary TEXT: Short summary of the reported issue
- resolved_at TEXT: Derived resolution timestamp ('YYYY-MM-DD HH:MM'). NULL for unresolved tickets.

CRITICAL BUSINESS & QUERY RULES:
1. DATASET REFERENCE TIME:
   The current reference timestamp of the dataset is '{ref_time}'.
   The current reference month is '{ref_month}'.
   For any relative-time question (e.g. 'this month', 'current month', 'this week', 'recent'),
   ALWAYS use this reference date ('{ref_time}'). NEVER assume today's real-world date.

2. VERB DISAMBIGUATION (OPENED vs RESOLVED):
   - For 'opened this month' / 'created this month':
     Use: WHERE strftime('%Y-%m', created_at) = '{ref_month}'
   - For 'resolved this month' / 'which agent resolved the most tickets this month':
     Use: WHERE resolved_at IS NOT NULL AND strftime('%Y-%m', resolved_at) = '{ref_month}'
   - For 'which agent resolved the most tickets overall':
     Use: WHERE resolution_time_hrs IS NOT NULL GROUP BY agent_id ORDER BY count(*) DESC

3. DATA-DRIVEN RESOLUTION DEFINITION:
   - A ticket has a resolution outcome if and only if 'resolution_time_hrs IS NOT NULL' (or 'resolved_at IS NOT NULL').
   - A ticket is unresolved if 'resolution_time_hrs IS NULL'.

4. TIME CALCULATIONS (SQLite):
   - Grouping by month: strftime('%Y-%m', created_at) or strftime('%Y-%m', resolved_at)
   - Duration / age relative to reference date: (julianday('{ref_time}') - julianday(created_at)) * 24.0

5. SAFETY:
   - ONLY generate read-only SELECT queries.
   - Do NOT use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, ATTACH, or multi-statement queries.
""".strip()
