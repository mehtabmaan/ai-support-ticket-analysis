"""
Tests for SQLite concurrency safety under concurrent requests.
"""

from concurrent.futures import ThreadPoolExecutor
import pytest
from src.data_layer.loader import execute_query, initialize_database


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    initialize_database()


def test_concurrent_read_queries():
    def run_query(i):
        sql = f"SELECT count(*) as cnt, agent_id FROM tickets WHERE rowid % 12 = {i % 12} GROUP BY agent_id"
        res, _ = execute_query(sql)
        return len(res)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_query, i) for i in range(25)]
        results = [f.result() for f in futures]

    assert len(results) == 25
    assert all(r >= 0 for r in results)
