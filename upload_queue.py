"""
SQLite-backed upload queue for store-and-forward S3 uploads.

Persists across process restarts — any clips queued during a crashed or
interrupted session are retried automatically on the next run.
"""
import sqlite3
import os
from datetime import datetime, timedelta

MAX_ATTEMPTS = 10
_BASE_DELAY  = 5   # seconds; doubles each attempt, capped at 300s


def _backoff(attempts: int) -> int:
    return min(_BASE_DELAY * (2 ** attempts), 300)


class UploadQueue:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    type         TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    clip_id      TEXT,
                    tier         TEXT,
                    mp4_path     TEXT,
                    ts_path      TEXT,
                    klv_path     TEXT,
                    local_path   TEXT,
                    status       TEXT DEFAULT 'pending',
                    attempts     INTEGER DEFAULT 0,
                    next_retry_at TEXT,
                    error        TEXT,
                    created_at   TEXT NOT NULL
                )
            ''')
            # Migrate DBs created before the transcode stage moved to the worker
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(uploads)')]
            if 'mp4_path' not in cols:
                conn.execute('ALTER TABLE uploads ADD COLUMN mp4_path TEXT')

    def enqueue_clip(self, session_id: str, clip_id: str, tier: str,
                     ts_path: str, klv_path: str, mp4_path: str = None):
        with self._conn() as conn:
            conn.execute(
                '''INSERT INTO uploads
                   (type, session_id, clip_id, tier, ts_path, klv_path, mp4_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                ('clip', session_id, clip_id, tier, ts_path, klv_path, mp4_path,
                 datetime.now().isoformat()),
            )

    def enqueue_summary(self, session_id: str, local_path: str):
        with self._conn() as conn:
            conn.execute(
                '''INSERT INTO uploads
                   (type, session_id, local_path, created_at)
                   VALUES (?, ?, ?, ?)''',
                ('summary', session_id, local_path, datetime.now().isoformat()),
            )

    def get_pending(self, limit: int = 10) -> list[dict]:
        """Return items ready for upload (respects exponential backoff)."""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                '''SELECT * FROM uploads
                   WHERE status = 'pending'
                     AND attempts < ?
                     AND (next_retry_at IS NULL OR next_retry_at <= ?)
                   ORDER BY created_at ASC
                   LIMIT ?''',
                (MAX_ATTEMPTS, now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_done(self, row_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE uploads SET status = 'done', error = NULL WHERE id = ?",
                (row_id,),
            )

    def mark_failed(self, row_id: int, error: str):
        with self._conn() as conn:
            row = conn.execute(
                'SELECT attempts FROM uploads WHERE id = ?', (row_id,)
            ).fetchone()
            attempts     = row['attempts'] + 1
            next_retry   = (datetime.now() + timedelta(seconds=_backoff(attempts))).isoformat()
            conn.execute(
                '''UPDATE uploads
                   SET attempts = ?, next_retry_at = ?, error = ?
                   WHERE id = ?''',
                (attempts, next_retry, str(error), row_id),
            )

    def pending_count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM uploads WHERE status = 'pending' AND attempts < ?",
                (MAX_ATTEMPTS,),
            ).fetchone()[0]

    def failed_items(self) -> list[dict]:
        """Items that exhausted MAX_ATTEMPTS — never retried, only reported."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM uploads WHERE status = 'pending' AND attempts >= ?",
                (MAX_ATTEMPTS,),
            ).fetchall()
        return [dict(r) for r in rows]

    def purge_done(self, older_than_days: int = 7) -> int:
        """Delete done rows older than the cutoff. Returns rows removed."""
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM uploads WHERE status = 'done' AND created_at < ?",
                (cutoff,),
            )
        return cur.rowcount
