"""
SQLite-based job queue for distributing search queries across parallel agent workers.

Uses WAL journal mode and BEGIN IMMEDIATE transactions to guarantee that each
query is claimed by exactly one worker, even when N workers run concurrently.
Stale in_progress jobs (worker crashed) are automatically reset on startup.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Jobs stuck in in_progress longer than this are considered stale and reset
STALE_TIMEOUT_MINUTES = 120


class JobQueue:
    def __init__(self, db_path: str | Path = "data/jobs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._reset_stale()

    # ── internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    query       TEXT PRIMARY KEY,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    created_at  TEXT NOT NULL,
                    started_at  TEXT,
                    finished_at TEXT,
                    error       TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
            conn.commit()

    def _reset_stale(self) -> None:
        """Reset in_progress jobs that have been running too long (crashed workers)."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=STALE_TIMEOUT_MINUTES)
        ).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET status='pending', started_at=NULL
                WHERE status='in_progress' AND started_at < ?
                """,
                (cutoff,),
            )
            if cursor.rowcount:
                logger.warning("Reset %d stale jobs", cursor.rowcount)
            conn.commit()

    # ── public API ────────────────────────────────────────────────────────────

    def initialize(self, queries: list[str], reset: bool = False) -> None:
        """
        Populate the queue with queries. Idempotent by default — already
        existing queries are not touched. Pass reset=True to wipe and reload.
        """
        with self._connect() as conn:
            if reset:
                conn.execute("DELETE FROM jobs")
            now = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                "INSERT OR IGNORE INTO jobs (query, status, created_at) VALUES (?, 'pending', ?)",
                [(q, now) for q in queries],
            )
            conn.commit()
        logger.info("Queue initialized with %d queries (reset=%s)", len(queries), reset)

    def claim_next(self) -> str | None:
        """
        Atomically claim the next pending query and return it.
        Returns None if the queue is empty.

        Uses BEGIN IMMEDIATE to prevent two workers from claiming the same query.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT query FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            query = row["query"]
            conn.execute(
                "UPDATE jobs SET status='in_progress', started_at=? WHERE query=?",
                (datetime.now(timezone.utc).isoformat(), query),
            )
            conn.execute("COMMIT")
            return query

    def mark_done(self, query: str) -> None:
        """Mark a claimed query as successfully completed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=? WHERE query=?",
                (datetime.now(timezone.utc).isoformat(), query),
            )
            conn.commit()

    def mark_failed(self, query: str, error: str) -> None:
        """Mark a claimed query as failed, storing the error message."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE query=?",
                (datetime.now(timezone.utc).isoformat(), error[:2000], query),
            )
            conn.commit()

    def stats(self) -> dict:
        """Return counts per status and a list of failed queries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
            ).fetchall()
            counts = {row["status"]: row["n"] for row in rows}
            failed = conn.execute(
                "SELECT query, error FROM jobs WHERE status='failed'"
            ).fetchall()
        return {
            "pending": counts.get("pending", 0),
            "in_progress": counts.get("in_progress", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "total": sum(counts.values()),
            "failed_queries": [{"query": r["query"], "error": r["error"]} for r in failed],
        }

    def retry_failed(self) -> int:
        """Reset all failed jobs back to pending. Returns the number of reset jobs."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status='pending', started_at=NULL, finished_at=NULL, error=NULL "
                "WHERE status='failed'"
            )
            conn.commit()
            return cursor.rowcount
