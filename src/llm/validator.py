"""
SQL validation and hardening module.
Enforces multi-layer defense against SQL injection, stacked queries,
and unauthorized DDL/DML operations before execution.
"""

import re
from typing import Set
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML, DDL


class SQLSecurityError(ValueError):
    """Raised when generated SQL violates safety constraints."""
    pass


FORBIDDEN_KEYWORDS: Set[str] = {
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "EXEC",
    "EXECUTE",
    "VACUUM",
    "REINDEX",
}


def validate_and_sanitize_sql(sql: str) -> str:
    """
    Validates that the SQL statement is strictly a single, read-only SELECT query.
    Enforces AST inspection via sqlparse.
    """
    if not sql or not sql.strip():
        raise SQLSecurityError("Empty SQL statement provided.")

    clean_sql = sql.strip().rstrip(";")

    # 1. Parse AST with sqlparse
    parsed = sqlparse.parse(clean_sql)
    if len(parsed) != 1:
        raise SQLSecurityError(
            f"Expected exactly 1 SQL statement, but found {len(parsed)}. "
            "Stacked statements are strictly prohibited."
        )

    stmt: Statement = parsed[0]

    # 2. Check statement type
    stmt_type = stmt.get_type().upper()
    if stmt_type != "SELECT":
        raise SQLSecurityError(
            f"Unauthorized statement type '{stmt_type}'. Only 'SELECT' queries are allowed."
        )

    # 3. Check for forbidden keywords across all tokens
    # Token extraction handles comments, literals, and subqueries
    for token in stmt.flatten():
        val = token.value.strip().upper()
        if val in FORBIDDEN_KEYWORDS:
            raise SQLSecurityError(f"Forbidden keyword '{val}' detected in query.")

    # 4. Regex safety check for PRAGMA or ATTACH even if disguised
    forbidden_pattern = re.compile(r"\b(PRAGMA|ATTACH|DETACH|INTO\s+OUTFILE)\b", re.IGNORECASE)
    if forbidden_pattern.search(clean_sql):
        raise SQLSecurityError("Forbidden database administrative directive detected.")

    # 5. Append LIMIT if query selects raw rows without aggregation or LIMIT
    has_limit = bool(re.search(r"\bLIMIT\s+\d+", clean_sql, re.IGNORECASE))
    is_count_only = bool(re.search(r"SELECT\s+COUNT\s*\(", clean_sql, re.IGNORECASE))

    if not has_limit and not is_count_only:
        clean_sql = f"{clean_sql} LIMIT 100"

    return clean_sql
