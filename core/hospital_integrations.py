"""
Hospital Information System (HIS) & Infection Control Integration Module.
Handles HL7-compatible notification payloads, shift-based compliance auditing,
and automated nurse station dispatch.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
from core.alert_manager import Incident


class HospitalAlertDispatcher:
    """
    Dispatches infection control alerts to department nurse stations,
    safety officer pagers, and hospital audit systems.
    """

    def __init__(self, log_path: str = "data/incidents/hospital_dispatches.json"):
        self.log_path = log_path
        self.facility_id = "MGH-MAIN-01"
        self.dispatches: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._load_dispatches()

    def _load_dispatches(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.dispatches = json.load(f)
            except Exception:
                self.dispatches = []

    def _save_dispatches(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.dispatches, f, indent=2)
        except Exception as e:
            print(f"[HospitalAlertDispatcher] Error saving dispatches: {e}")

    @staticmethod
    def get_current_hospital_shift() -> str:
        """Determines the current clinical hospital shift."""
        hour = datetime.now().hour
        if 7 <= hour < 15:
            return "MORNING_SHIFT (07:00 - 15:00)"
        elif 15 <= hour < 23:
            return "EVENING_SHIFT (15:00 - 23:00)"
        else:
            return "NIGHT_SHIFT (23:00 - 07:00)"

    def dispatch_incident_alert(
        self,
        incident: Incident,
        department_name: str,
        department_code: str,
        nurse_station: str = "",
        safety_email: str = ""
    ) -> Dict[str, Any]:
        """
        Formats and logs an infection control dispatch payload.
        Compatible with hospital notification and HL7 event standards.
        """
        shift = self.get_current_hospital_shift()
        timestamp = datetime.now().isoformat()

        # HL7/FHIR style structured clinical safety event
        dispatch_payload = {
            "event_id": f"DISP-{incident.id}",
            "facility_id": self.facility_id,
            "timestamp": timestamp,
            "shift": shift,
            "department": {
                "name": department_name,
                "code": department_code,
                "nurse_station": nurse_station,
                "safety_officer": safety_email,
            },
            "clinical_alert": {
                "type": "INFECTION_CONTROL_PPE_BREACH",
                "severity": incident.severity,
                "track_id": incident.track_id,
                "missing_ppe": incident.missing_items,
                "improper_ppe": incident.improper_items,
                "dwell_duration_seconds": incident.dwell_seconds,
                "evidence_snapshot": incident.snapshot_filename
            },
            "status": "DISPATCHED_TO_NURSE_STATION"
        }

        self.dispatches.insert(0, dispatch_payload)
        self._save_dispatches()
        return dispatch_payload

    def get_shift_audit_summary(self) -> Dict[str, Any]:
        """Groups recorded violation dispatches by clinical shift."""
        shift_counts = {
            "MORNING_SHIFT (07:00 - 15:00)": 0,
            "EVENING_SHIFT (15:00 - 23:00)": 0,
            "NIGHT_SHIFT (23:00 - 07:00)": 0
        }

        for d in self.dispatches:
            s = d.get("shift", "")
            if s in shift_counts:
                shift_counts[s] += 1

        return {
            "current_shift": self.get_current_hospital_shift(),
            "total_dispatches": len(self.dispatches),
            "dispatches_by_shift": shift_counts
        }

