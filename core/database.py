"""
Hospital PPE Monitoring System - SQLite Database Layer.
Provides ACID-compliant, thread-safe persistence for violation incidents,
hospital dispatches, audit trails, and shift compliance analytics.
"""

import sqlite3
import os
import json
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

DEFAULT_DB_PATH = "data/incidents/hospital_ppe.db"


class HospitalDatabase:
    """
    Thread-safe SQLite database manager for hospital PPE incidents and dispatches.
    """
    _local = threading.local()

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns thread-local connection for safe multi-threaded concurrency."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging for high concurrency
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = conn
        return conn

    def _init_schema(self):
        """Initializes database schema with indexing on query-heavy columns."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. Incidents Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                track_id INTEGER,
                zone_id TEXT,
                zone_name TEXT,
                zone_code TEXT,
                severity TEXT,
                timestamp TEXT,
                state TEXT,
                missing_items TEXT,
                improper_items TEXT,
                dwell_seconds REAL,
                details TEXT,
                snapshot_filename TEXT,
                status TEXT DEFAULT 'NEW',
                episode_id TEXT,
                duration_seconds REAL DEFAULT 0.0,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                resolved_by TEXT,
                resolved_at TEXT,
                notes TEXT DEFAULT '',
                last_seen_timestamp TEXT
            );
            """)

            # 2. Indexes for fast dashboard & audit queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_zone_id ON incidents(zone_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);")

            # 3. Hospital Dispatches Table (HL7 events)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dispatches (
                event_id TEXT PRIMARY KEY,
                facility_id TEXT,
                timestamp TEXT,
                shift TEXT,
                department_name TEXT,
                department_code TEXT,
                nurse_station TEXT,
                safety_officer TEXT,
                severity TEXT,
                track_id INTEGER,
                missing_ppe TEXT,
                improper_ppe TEXT,
                dwell_duration_seconds REAL,
                evidence_snapshot TEXT,
                status TEXT
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dispatches_timestamp ON dispatches(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dispatches_shift ON dispatches(shift);")

            conn.commit()
            conn.close()

    def insert_or_update_incident(self, incident_dict: Dict[str, Any]):
        """Inserts or updates an incident episode in the database."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            missing_str = json.dumps(incident_dict.get("missing_items", []))
            improper_str = json.dumps(incident_dict.get("improper_items", []))

            cursor.execute("""
            INSERT INTO incidents (
                id, track_id, zone_id, zone_name, zone_code, severity,
                timestamp, state, missing_items, improper_items, dwell_seconds,
                details, snapshot_filename, status, episode_id, duration_seconds,
                acknowledged_by, acknowledged_at, resolved_by, resolved_at, notes, last_seen_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                dwell_seconds = excluded.dwell_seconds,
                duration_seconds = excluded.duration_seconds,
                missing_items = excluded.missing_items,
                improper_items = excluded.improper_items,
                status = excluded.status,
                acknowledged_by = excluded.acknowledged_by,
                acknowledged_at = excluded.acknowledged_at,
                resolved_by = excluded.resolved_by,
                resolved_at = excluded.resolved_at,
                notes = excluded.notes,
                last_seen_timestamp = excluded.last_seen_timestamp;
            """, (
                incident_dict.get("id"),
                incident_dict.get("track_id"),
                incident_dict.get("zone_id"),
                incident_dict.get("zone_name"),
                incident_dict.get("zone_code"),
                incident_dict.get("severity"),
                incident_dict.get("timestamp"),
                incident_dict.get("state"),
                missing_str,
                improper_str,
                incident_dict.get("dwell_seconds", 0.0),
                incident_dict.get("details", ""),
                incident_dict.get("snapshot_filename"),
                incident_dict.get("status", "NEW"),
                incident_dict.get("episode_id"),
                incident_dict.get("duration_seconds", 0.0),
                incident_dict.get("acknowledged_by"),
                incident_dict.get("acknowledged_at"),
                incident_dict.get("resolved_by"),
                incident_dict.get("resolved_at"),
                incident_dict.get("notes", ""),
                incident_dict.get("last_seen_timestamp")
            ))
            conn.commit()

    def get_recent_incidents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves most recent violation incidents."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["missing_items"] = json.loads(item["missing_items"])
                except Exception:
                    item["missing_items"] = []
                try:
                    item["improper_items"] = json.loads(item["improper_items"])
                except Exception:
                    item["improper_items"] = []
                results.append(item)
            return results

    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Finds a specific incident by its unique ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            try:
                item["missing_items"] = json.loads(item["missing_items"])
            except Exception:
                item["missing_items"] = []
            try:
                item["improper_items"] = json.loads(item["improper_items"])
            except Exception:
                item["improper_items"] = []
            return item

    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        user: str = "Admin",
        notes: Optional[str] = None
    ) -> bool:
        """Updates the lifecycle status of an incident episode."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if status in ["ACKNOWLEDGED", "UNDER_REVIEW"]:
                cursor.execute("""
                UPDATE incidents SET
                    status = ?,
                    acknowledged_by = ?,
                    acknowledged_at = ?,
                    notes = COALESCE(?, notes)
                WHERE id = ?;
                """, (status, user, now, notes, incident_id))
            elif status in ["RESOLVED", "FALSE_POSITIVE"]:
                cursor.execute("""
                UPDATE incidents SET
                    status = ?,
                    resolved_by = ?,
                    resolved_at = ?,
                    notes = COALESCE(?, notes)
                WHERE id = ?;
                """, (status, user, now, notes, incident_id))
            else:
                cursor.execute("UPDATE incidents SET status = ? WHERE id = ?;", (status, incident_id))
            conn.commit()
            return cursor.rowcount > 0

    def insert_dispatch(self, dispatch_payload: Dict[str, Any]):
        """Records a hospital alert dispatch (HL7 event)."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            dept = dispatch_payload.get("department", {})
            alert = dispatch_payload.get("clinical_alert", {})

            cursor.execute("""
            INSERT OR REPLACE INTO dispatches (
                event_id, facility_id, timestamp, shift,
                department_name, department_code, nurse_station, safety_officer,
                severity, track_id, missing_ppe, improper_ppe, dwell_duration_seconds,
                evidence_snapshot, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                dispatch_payload.get("event_id"),
                dispatch_payload.get("facility_id"),
                dispatch_payload.get("timestamp"),
                dispatch_payload.get("shift"),
                dept.get("name"),
                dept.get("code"),
                dept.get("nurse_station"),
                dept.get("safety_officer"),
                alert.get("severity"),
                alert.get("track_id"),
                json.dumps(alert.get("missing_ppe", [])),
                json.dumps(alert.get("improper_ppe", [])),
                alert.get("dwell_duration_seconds", 0.0),
                alert.get("evidence_snapshot"),
                dispatch_payload.get("status")
            ))
            conn.commit()

    def get_shift_summary(self) -> Dict[str, int]:
        """Calculates dispatch counts grouped by clinical hospital shift."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT shift, COUNT(*) as count FROM dispatches GROUP BY shift;")
            rows = cursor.fetchall()
            summary = {
                "MORNING_SHIFT (07:00 - 15:00)": 0,
                "EVENING_SHIFT (15:00 - 23:00)": 0,
                "NIGHT_SHIFT (23:00 - 07:00)": 0,
            }
            for row in rows:
                if row["shift"] in summary:
                    summary[row["shift"]] = row["count"]
            return summary
