"""
Incident and Alert Management Module.
Handles episode-based violation tracking, incident snapshot capture with HIPAA/privacy face blurring,
incident status lifecycle management (NEW -> ACKNOWLEDGED -> UNDER_REVIEW -> RESOLVED / FALSE_POSITIVE),
data retention cleanup, and compliance analytics.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
import json
import uuid
import cv2
import numpy as np

from core.compliance import ComplianceResult, ComplianceState
from core.tracker import TrackedPerson
from core.database import HospitalDatabase


class IncidentStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


@dataclass
class Incident:
    """Represents a logged PPE violation incident episode."""
    id: str
    track_id: int
    zone_id: str
    zone_name: str
    zone_code: str
    severity: str
    timestamp: str
    state: str
    missing_items: List[str]
    improper_items: List[str]
    dwell_seconds: float
    details: str
    snapshot_filename: Optional[str] = None
    status: str = "NEW"
    episode_id: Optional[str] = None
    duration_seconds: float = 0.0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    notes: str = ""
    last_seen_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertManager:
    """
    Manages violation alerting, episode deduplication, snapshot archiving with privacy blurring,
    incident lifecycle transitions, and compliance telemetry with SQLite backing.
    """

    def __init__(
        self,
        snapshot_dir: str = "data/incidents/snapshots",
        log_dir: str = "data/incidents",
        cooldown_seconds: float = 12.0,
        enable_privacy_blur: bool = True
    ):
        self.snapshot_dir = snapshot_dir
        self.log_dir = log_dir
        self.cooldown_seconds = cooldown_seconds
        self.enable_privacy_blur = enable_privacy_blur
        self.log_file = os.path.join(log_dir, "incidents.json")

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize SQLite database
        self.db = HospitalDatabase(os.path.join(log_dir, "hospital_ppe.db"))

        self.incidents: List[Incident] = []
        self._load_existing_incidents()

        # Episode tracking: (track_id, zone_id) -> (Incident, episode_start_time_sec)
        self._active_episodes: Dict[Tuple[int, str], Tuple[Incident, float]] = {}

        # Running statistics counters
        self.total_evaluations: int = 0
        self.compliant_evaluations: int = 0
        self.partial_evaluations: int = 0
        self.violation_evaluations: int = 0

    def _load_existing_incidents(self):
        # First attempt loading from SQLite database
        try:
            db_records = self.db.get_recent_incidents(limit=500)
            if db_records:
                for item in db_records:
                    self.incidents.append(Incident(**item))
                return
        except Exception as e:
            print(f"[AlertManager] Notice: SQLite load fallback to JSON: {e}")

        # Fallback to JSON file if DB is empty or fresh
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        for item in data:
                            if "status" not in item:
                                item["status"] = "RESOLVED"
                            if "episode_id" not in item:
                                item["episode_id"] = item.get("id")
                            if "duration_seconds" not in item:
                                item["duration_seconds"] = item.get("dwell_seconds", 0.0)
                            if "acknowledged_by" not in item:
                                item["acknowledged_by"] = None
                            if "acknowledged_at" not in item:
                                item["acknowledged_at"] = None
                            if "resolved_by" not in item:
                                item["resolved_by"] = None
                            if "resolved_at" not in item:
                                item["resolved_at"] = None
                            if "notes" not in item:
                                item["notes"] = ""
                            if "last_seen_timestamp" not in item:
                                item["last_seen_timestamp"] = item.get("timestamp")
                            inc = Incident(**item)
                            self.incidents.append(inc)
                            # Migrate to SQLite
                            try:
                                self.db.insert_or_update_incident(inc.to_dict())
                            except Exception:
                                pass
            except Exception as e:
                print(f"[AlertManager] Warning loading incidents log: {e}")

    def _save_incidents(self, specific_incident: Optional[Incident] = None):
        # 1. Update SQLite Database
        try:
            if specific_incident:
                self.db.insert_or_update_incident(specific_incident.to_dict())
            else:
                for inc in self.incidents[:50]:  # sync recent incidents
                    self.db.insert_or_update_incident(inc.to_dict())
        except Exception as e:
            print(f"[AlertManager] Warning saving to SQLite: {e}")

        # 2. Sync to JSON for legacy and audit export compatibility
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([inc.to_dict() for inc in self.incidents], f, indent=2)
        except Exception as e:
            print(f"[AlertManager] Error saving incidents log: {e}")

    @staticmethod
    def blur_head_privacy(
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        blur_kernel: Tuple[int, int] = (51, 51)
    ) -> np.ndarray:
        """
        Applies heavy Gaussian blur to head/face region before writing snapshot
        to disk, ensuring HIPAA and clinical privacy compliance.
        """
        if frame is None or frame.size == 0:
            return frame

        out_frame = frame.copy()
        h, w = out_frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        pw = x2 - x1
        ph = y2 - y1

        if pw <= 4 or ph <= 4:
            return out_frame

        # Upper 28% of person bounding box represents head/face
        hy2 = min(y2, int(y1 + ph * 0.28))
        head_crop = out_frame[y1:hy2, x1:x2]

        if head_crop.size > 0 and head_crop.shape[0] > 3 and head_crop.shape[1] > 3:
            kx = blur_kernel[0] if blur_kernel[0] % 2 == 1 else blur_kernel[0] + 1
            ky = blur_kernel[1] if blur_kernel[1] % 2 == 1 else blur_kernel[1] + 1
            blurred = cv2.GaussianBlur(head_crop, (kx, ky), 0)
            out_frame[y1:hy2, x1:x2] = blurred

        return out_frame

    def process_result(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        result: ComplianceResult,
        current_time_sec: float
    ) -> Optional[Incident]:
        """
        Evaluates person result under episode-based tracking:
        - Deduplicates continuous violations under single incident episode.
        - Automatically resolves episode when person complies or leaves.
        - Applies face blurring to saved snapshots.
        """
        self.total_evaluations += 1
        if result.state == ComplianceState.COMPLIANT:
            self.compliant_evaluations += 1
        elif result.state == ComplianceState.PARTIALLY_COMPLIANT:
            self.partial_evaluations += 1
        elif result.state == ComplianceState.NON_COMPLIANT:
            self.violation_evaluations += 1

        # Check if violation requires alerting
        if result.should_alert and result.zone:
            ep_key = (person.track_id, result.zone.id)

            # Ongoing episode: update metadata without duplicating alerts
            if ep_key in self._active_episodes:
                incident, start_time = self._active_episodes[ep_key]
                incident.dwell_seconds = result.dwell_seconds
                incident.duration_seconds = max(0.0, current_time_sec - start_time)
                incident.missing_items = result.missing_items
                incident.improper_items = result.improper_items
                incident.last_seen_timestamp = datetime.now().isoformat()
                return None

            # New violation episode begins
            inc_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
            timestamp_str = datetime.now().isoformat()
            snap_filename = f"{inc_id}.jpg"
            snap_path = os.path.join(self.snapshot_dir, snap_filename)

            # Apply privacy blurring to snapshot before writing to disk
            snap_frame = self.blur_head_privacy(frame, person.bbox) if self.enable_privacy_blur else frame
            try:
                cv2.imwrite(snap_path, snap_frame)
            except Exception as e:
                print(f"[AlertManager] Failed to save snapshot {snap_path}: {e}")
                snap_filename = None

            incident = Incident(
                id=inc_id,
                track_id=person.track_id,
                zone_id=result.zone.id,
                zone_name=result.zone.name,
                zone_code=result.zone.code,
                severity=result.zone.alert_severity,
                timestamp=timestamp_str,
                state=result.state.value,
                missing_items=result.missing_items,
                improper_items=result.improper_items,
                dwell_seconds=result.dwell_seconds,
                details=result.details,
                snapshot_filename=snap_filename,
                status=IncidentStatus.NEW.value,
                episode_id=inc_id,
                duration_seconds=0.0,
                last_seen_timestamp=timestamp_str
            )

            self.incidents.insert(0, incident)
            self._active_episodes[ep_key] = (incident, current_time_sec)
            self._save_incidents()
            return incident

        else:
            # Person is compliant, in grace period, or unmonitored
            # If person had an active violation episode in this or previous zone, auto-resolve it
            zone_id = result.zone.id if result.zone else ""
            keys_to_resolve = [k for k in self._active_episodes.keys() if k[0] == person.track_id]
            for ep_key in keys_to_resolve:
                incident, start_time = self._active_episodes.pop(ep_key)
                if incident.status in [IncidentStatus.NEW.value, IncidentStatus.ACKNOWLEDGED.value, IncidentStatus.UNDER_REVIEW.value]:
                    incident.status = IncidentStatus.RESOLVED.value
                    incident.resolved_at = datetime.now().isoformat()
                    incident.resolved_by = "SYSTEM (Auto-Compliance)"
                    incident.duration_seconds = max(0.0, current_time_sec - start_time)
            if keys_to_resolve:
                self._save_incidents()

            return None

    def close_inactive_tracks(self, active_track_ids: List[int], current_time_sec: float):
        """Auto-resolves episodes for personnel who have left the monitored camera area."""
        active_set = set(active_track_ids)
        stale_keys = [k for k in self._active_episodes.keys() if k[0] not in active_set]

        if not stale_keys:
            return

        for k in stale_keys:
            incident, start_time = self._active_episodes.pop(k)
            if incident.status in [IncidentStatus.NEW.value, IncidentStatus.ACKNOWLEDGED.value, IncidentStatus.UNDER_REVIEW.value]:
                incident.status = IncidentStatus.RESOLVED.value
                incident.resolved_at = datetime.now().isoformat()
                incident.resolved_by = "SYSTEM (Exited Area)"
                incident.duration_seconds = max(0.0, current_time_sec - start_time)

        self._save_incidents()

    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        user: str = "Staff",
        notes: str = ""
    ) -> Optional[Incident]:
        """
        Transitions incident status along the clinical lifecycle:
        NEW -> ACKNOWLEDGED -> UNDER_REVIEW -> RESOLVED / FALSE_POSITIVE.
        """
        incident = self.get_incident_by_id(incident_id)
        if not incident:
            return None

        status_upper = status.upper()
        incident.status = status_upper
        now_str = datetime.now().isoformat()

        if status_upper == IncidentStatus.ACKNOWLEDGED.value:
            incident.acknowledged_by = user
            incident.acknowledged_at = now_str
        elif status_upper in [IncidentStatus.RESOLVED.value, IncidentStatus.FALSE_POSITIVE.value]:
            incident.resolved_by = user
            incident.resolved_at = now_str
            # Remove from active episodes if still present
            keys_to_remove = [k for k, (inc, _) in self._active_episodes.items() if inc.id == incident_id]
            for k in keys_to_remove:
                del self._active_episodes[k]

        if notes:
            incident.notes = f"{incident.notes}; {notes}".strip("; ")

        self._save_incidents()
        return incident

    def cleanup_old_snapshots(self, max_age_days: int = 7) -> int:
        """Purges incident snapshot JPEG files older than max_age_days."""
        purged = 0
        cutoff = datetime.now() - timedelta(days=max_age_days)

        if not os.path.exists(self.snapshot_dir):
            return 0

        for f in os.listdir(self.snapshot_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(self.snapshot_dir, f)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    if mtime < cutoff:
                        os.remove(fpath)
                        purged += 1
                except Exception:
                    pass

        return purged

    def cleanup_old_incidents(self, max_age_days: int = 30) -> int:
        """Purges incident log records older than max_age_days."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        initial_len = len(self.incidents)
        retained = []

        for inc in self.incidents:
            try:
                inc_time = datetime.fromisoformat(inc.timestamp)
                if inc_time >= cutoff:
                    retained.append(inc)
            except Exception:
                retained.append(inc)

        self.incidents = retained
        self._save_incidents()
        return initial_len - len(retained)

    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [inc.to_dict() for inc in self.incidents[:limit]]

    def get_incident_by_id(self, incident_id: str) -> Optional[Incident]:
        for inc in self.incidents:
            if inc.id == incident_id:
                return inc
        return None

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Calculates compliance rate, violations by zone, and missing item frequencies."""
        total_evals = max(1, self.total_evaluations)
        compliance_rate = (self.compliant_evaluations / total_evals) * 100.0

        zone_counts: Dict[str, int] = {}
        item_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
        status_counts: Dict[str, int] = {
            "NEW": 0,
            "ACKNOWLEDGED": 0,
            "UNDER_REVIEW": 0,
            "RESOLVED": 0,
            "FALSE_POSITIVE": 0
        }

        for inc in self.incidents:
            zone_counts[inc.zone_name] = zone_counts.get(inc.zone_name, 0) + 1
            severity_counts[inc.severity] = severity_counts.get(inc.severity, 0) + 1
            status_counts[inc.status] = status_counts.get(inc.status, 0) + 1

            for m in inc.missing_items:
                item_counts[m] = item_counts.get(m, 0) + 1
            for imp in inc.improper_items:
                clean_imp = imp.split(" ")[0]
                item_counts[clean_imp] = item_counts.get(clean_imp, 0) + 1

        return {
            "total_incidents": len(self.incidents),
            "compliance_rate_percent": round(compliance_rate, 1),
            "total_evaluations": self.total_evaluations,
            "compliant_evaluations": self.compliant_evaluations,
            "partial_evaluations": self.partial_evaluations,
            "violation_evaluations": self.violation_evaluations,
            "zone_distribution": zone_counts,
            "missing_item_distribution": item_counts,
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "active_episodes_count": len(self._active_episodes)
        }
