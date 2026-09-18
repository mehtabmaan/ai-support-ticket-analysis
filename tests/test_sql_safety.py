"""
Tests for SQL security validation, AST parsing, and SQLite authorizer protection.
"""

import sqlite3
import pytest
from src.data_layer.loader import get_db_connection, initialize_database
from src.llm.validator import SQLSecurityError, validate_and_sanitize_sql


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    initialize_database()


def test_valid_select_allowed():
    sql = "SELECT ticket_id, category, priority FROM tickets WHERE status = 'Open' LIMIT 10;"
    sanitized = validate_and_sanitize_sql(sql)
    assert "SELECT" in sanitized
    assert "LIMIT 10" in sanitized


def test_statement_stacking_rejected():
    stacked_sql = "SELECT 1; DROP TABLE tickets;"
    with pytest.raises(SQLSecurityError) as exc:
        validate_and_sanitize_sql(stacked_sql)
    assert "Stacked statements are strictly prohibited" in str(exc.value)


def test_ddl_and_dml_rejected():
    for bad in [
        "INSERT INTO tickets (ticket_id) VALUES ('HACK')",
        "UPDATE tickets SET status = 'Resolved'",
        "DELETE FROM tickets WHERE ticket_id = 'TKT-001'",
        "DROP TABLE tickets",
        "ALTER TABLE tickets ADD COLUMN evil TEXT",
        "CREATE TABLE evil (id INT)",
    ]:
        with pytest.raises(SQLSecurityError):
            validate_and_sanitize_sql(bad)


def test_admin_commands_rejected():
    for admin in [
        "PRAGMA table_info(tickets)",
        "ATTACH DATABASE ':memory:' AS evil",
        "DETACH DATABASE evil",
        "VACUUM",
    ]:
        with pytest.raises(SQLSecurityError):
            validate_and_sanitize_sql(admin)


def test_sqlite_engine_authorizer_enforcement():
    conn = get_db_connection(read_only=True)
    cursor = conn.cursor()

    # Reading is permitted
    cursor.execute("SELECT count(*) FROM tickets")
    assert cursor.fetchone()[0] == 500

    # Write operations are blocked at SQLite C engine level
    with pytest.raises(sqlite3.DatabaseError):
        cursor.execute("DELETE FROM tickets")

    with pytest.raises(sqlite3.DatabaseError):
        cursor.execute("DROP TABLE tickets")

    conn.close()
