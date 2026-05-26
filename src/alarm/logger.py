"""
SQLite event logging for fall detection alarms.

Persists all alarm events with metadata to a local SQLite database
using WAL mode for non-blocking concurrent access.

Reference: research/9.2 - SQLite with WAL for edge-based logging.
Schema includes event_id, timestamps, severity, confidence, camera_id,
video_clip_path, review status, and false_positive flag.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    camera_id       TEXT NOT NULL DEFAULT 'camera_0',
    duration_seconds REAL,
    video_clip_path TEXT,
    metadata        TEXT,
    reviewed        INTEGER NOT NULL DEFAULT 0,
    is_false_positive INTEGER
)
"""


class EventLogger:
    """
    SQLite-based alarm event logger with WAL optimization.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for non-blocking reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

        logger.info(f"Event logger initialized: {self._db_path}")

    def log_event(
        self,
        severity: str,
        confidence: float,
        camera_id: str = "camera_0",
        duration_seconds: Optional[float] = None,
        video_clip_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Log a fall detection event to the database.

        Parameters
        ----------
        severity : str
            Event severity: 'WARNING', 'CRITICAL', or 'RECOVERED'.
        confidence : float
            Peak model confidence score during the event.
        camera_id : str
            Identifier for the video source.
        duration_seconds : float, optional
            Duration of the fall event sequence.
        video_clip_path : str, optional
            Path to the saved video evidence clip.
        metadata : dict, optional
            Additional event metadata (JSON-serializable).

        Returns
        -------
        int
            The event_id of the inserted record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        cursor = self._conn.execute(
            """INSERT INTO events
               (timestamp, severity, confidence, camera_id,
                duration_seconds, video_clip_path, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                severity,
                confidence,
                camera_id,
                duration_seconds,
                video_clip_path,
                meta_json,
            ),
        )
        self._conn.commit()

        event_id = cursor.lastrowid
        logger.info(
            f"Event logged: id={event_id} severity={severity} "
            f"confidence={confidence:.3f}"
        )
        return event_id

    def update_metadata(self, event_id: int, updates: Dict) -> None:
        """Merge new key/value pairs into an event's metadata JSON.

        Used for delayed enrichments such as VLM narrations that arrive
        a few seconds after the alarm event was originally logged.

        Parameters
        ----------
        event_id : int
            Row identifier of the event to enrich.
        updates : dict
            JSON-serializable values to merge into the existing metadata.
        """
        cursor = self._conn.execute(
            "SELECT metadata FROM events WHERE event_id = ?",
            (event_id,),
        )
        row = cursor.fetchone()
        if row is None:
            logger.warning(f"update_metadata: event_id {event_id} not found")
            return
        existing = json.loads(row["metadata"]) if row["metadata"] else {}
        existing.update(updates)
        self._conn.execute(
            "UPDATE events SET metadata = ? WHERE event_id = ?",
            (json.dumps(existing), event_id),
        )
        self._conn.commit()

    def update_severity(self, event_id: int, severity: str) -> None:
        """Overwrite the severity column for an existing event.

        Used after post-fall stillness analysis refines the initial
        CRITICAL label into SEVERE / MODERATE / MINOR.

        Parameters
        ----------
        event_id : int
            Row identifier of the event to update.
        severity : str
            New severity label.
        """
        self._conn.execute(
            "UPDATE events SET severity = ? WHERE event_id = ?",
            (severity, event_id),
        )
        self._conn.commit()

    def mark_reviewed(self, event_id: int, is_false_positive: bool) -> None:
        """
        Mark an event as reviewed by a human operator.

        Parameters
        ----------
        event_id : int
            The event to update.
        is_false_positive : bool
            Whether the event was a false alarm.
        """
        self._conn.execute(
            """UPDATE events
               SET reviewed = 1, is_false_positive = ?
               WHERE event_id = ?""",
            (int(is_false_positive), event_id),
        )
        self._conn.commit()

    def get_events(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query recent events from the database.

        Parameters
        ----------
        limit : int
            Maximum number of events to return.
        severity : str, optional
            Filter by severity level.

        Returns
        -------
        list of dict
            Event records as dictionaries.
        """
        if severity:
            cursor = self._conn.execute(
                """SELECT * FROM events
                   WHERE severity = ?
                   ORDER BY event_id DESC LIMIT ?""",
                (severity, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM events ORDER BY event_id DESC LIMIT ?",
                (limit,),
            )

        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Event logger closed")
