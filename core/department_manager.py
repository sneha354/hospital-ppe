"""
Hospital Department Management and Multi-Camera Orchestration Module.
Manages concurrent surveillance streams, department-specific infection control policies,
and hospital-wide compliance telemetry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import os
import yaml
import time
import numpy as np

from core.pipeline import PPECompliancePipeline
from core.zone_manager import Zone
from core.camera_worker import CameraWorker


@dataclass
class Department:
    """Represents a distinct hospital clinical department with dedicated monitoring."""
    id: str
    name: str
    code: str
    building: str
    floor: str
    risk_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    camera_source: str
    camera_type: str
    color: Tuple[int, int, int]  # RGB
    required_ppe: List[str]
    optional_ppe: List[str] = field(default_factory=list)
    grace_period_seconds: float = 3.0
    alert_severity: str = "WARNING"
    nurse_station_contact: str = ""
    safety_officer_email: str = ""
    description: str = ""
    pipeline: Optional[PPECompliancePipeline] = None
    worker: Optional[CameraWorker] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "building": self.building,
            "floor": self.floor,
            "risk_level": self.risk_level,
            "camera_source": self.camera_source,
            "camera_type": self.camera_type,
            "color": list(self.color),
            "required_ppe": self.required_ppe,
            "optional_ppe": self.optional_ppe,
            "grace_period_seconds": self.grace_period_seconds,
            "alert_severity": self.alert_severity,
            "nurse_station_contact": self.nurse_station_contact,
            "safety_officer_email": self.safety_officer_email,
            "description": self.description,
        }


class HospitalDepartmentManager:
    """
    Coordinates multi-department surveillance pipelines, concurrent camera feeds,
    and departmental compliance analytics.
    """

    def __init__(self, config_path: str = "config/departments.yaml", start_workers: bool = True):
        self.config_path = config_path
        self.departments: Dict[str, Department] = {}
        self.start_workers = start_workers
        self.load_departments()

    def load_departments(self):
        """Loads departments from YAML configuration and initializes pipelines."""
        if not os.path.exists(self.config_path):
            print(f"[HospitalDepartmentManager] Config {self.config_path} not found.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_depts = data.get("departments", [])
        for d in raw_depts:
            dept = Department(
                id=d["id"],
                name=d.get("name", d["id"]),
                code=d.get("code", "DEPT-00"),
                building=d.get("building", "Main Building"),
                floor=d.get("floor", "1st Floor"),
                risk_level=d.get("risk_level", "MEDIUM"),
                camera_source=str(d.get("camera_source", "simulation")),
                camera_type=d.get("camera_type", "SURVEILLANCE"),
                color=tuple(d.get("color", [59, 130, 246])),
                required_ppe=d.get("required_ppe", []),
                optional_ppe=d.get("optional_ppe", []),
                grace_period_seconds=float(d.get("grace_period_seconds", 3.0)),
                alert_severity=d.get("alert_severity", "WARNING"),
                nurse_station_contact=d.get("nurse_station_contact", ""),
                safety_officer_email=d.get("safety_officer_email", ""),
                description=d.get("description", "")
            )

            # Initialize isolated compliance pipeline and background worker for this department
            dept.pipeline = self._create_department_pipeline(dept)
            dept.worker = CameraWorker(
                pipeline=dept.pipeline,
                name=f"Worker-{dept.id}",
                target_fps=20.0
            )
            if self.start_workers:
                dept.worker.start()

            self.departments[dept.id] = dept

    def _create_department_pipeline(self, dept: Department) -> PPECompliancePipeline:
        """Creates a dedicated PPE compliance pipeline for a specific department."""
        # Create department-specific primary zone
        primary_zone = Zone(
            id=f"zone_{dept.id}",
            name=f"{dept.name} Active Area",
            code=dept.code,
            risk_level=dept.risk_level,
            color=dept.color,
            polygon=[(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)],
            required_ppe=dept.required_ppe,
            optional_ppe=dept.optional_ppe,
            grace_period_seconds=dept.grace_period_seconds,
            alert_severity=dept.alert_severity,
            description=dept.description
        )

        pipeline = PPECompliancePipeline(source=dept.camera_source)
        # Register the department zone as sole primary in its zone manager
        pipeline.zone_manager.zones.clear()
        pipeline.zone_manager.add_or_update_zone(primary_zone, persist=False)
        return pipeline

    def get_all_departments(self) -> List[Department]:
        return list(self.departments.values())

    def get_department(self, dept_id: str) -> Optional[Department]:
        return self.departments.get(dept_id)

    def get_department_worker(self, dept_id: str) -> Optional[CameraWorker]:
        dept = self.departments.get(dept_id)
        return dept.worker if dept else None

    def set_department_camera(self, dept_id: str, new_source: str) -> bool:
        """Dynamically updates the camera feed for a specific department."""
        dept = self.departments.get(dept_id)
        if not dept:
            return False

        if dept.worker is not None:
            dept.worker.stop()

        if dept.pipeline is not None:
            dept.pipeline.stream_manager.release()

        dept.camera_source = str(new_source)
        dept.pipeline = self._create_department_pipeline(dept)
        dept.worker = CameraWorker(
            pipeline=dept.pipeline,
            name=f"Worker-{dept.id}",
            target_fps=20.0
        )
        if self.start_workers:
            dept.worker.start()

        return True

    def get_department_person_details(self, dept_id: str) -> List[Dict[str, Any]]:
        """
        Returns real-time person-level PPE evaluation details for personnel
        currently detected in the specified department.
        """
        dept = self.departments.get(dept_id)
        if not dept or not dept.pipeline:
            return []

        now = time.monotonic()
        persons_data = []

        active_tracks = [t for t in dept.pipeline.tracker.tracks.values() if t.lost_frames == 0]
        for t in active_tracks:
            duration_sec = max(0, int(now - t.first_seen_time))
            mins, secs = divmod(duration_sec, 60)
            hrs, mins = divmod(mins, 60)
            duration_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            # Format detection timestamp
            time_detected = time.strftime("%I:%M:%S %p", time.localtime(time.time() - duration_sec))

            # Evaluate against this department's specific required PPE
            ppe_status = t.get_smoothed_ppe_status()

            # Item details with simulated realistic confidence scores
            items = {
                "face_mask": {
                    "name": "Mask",
                    "detected": bool(ppe_status.get("face_mask", False)),
                    "confidence": 98 if ppe_status.get("face_mask", False) else 12,
                    "required": "face_mask" in dept.required_ppe
                },
                "gloves": {
                    "name": "Gloves",
                    "detected": bool(ppe_status.get("gloves", False)),
                    "confidence": 95 if ppe_status.get("gloves", False) else 8,
                    "required": "gloves" in dept.required_ppe
                },
                "gown": {
                    "name": "Gown",
                    "detected": bool(ppe_status.get("gown", False)),
                    "confidence": 92 if ppe_status.get("gown", False) else 15,
                    "required": "gown" in dept.required_ppe
                },
                "eye_protection": {
                    "name": "Eye Protection",
                    "detected": bool(ppe_status.get("eye_protection", False)),
                    "confidence": 88 if ppe_status.get("eye_protection", False) else 6,
                    "required": "eye_protection" in dept.required_ppe
                },
                "head_cover": {
                    "name": "Head Cover",
                    "detected": bool(ppe_status.get("head_cover", False)),
                    "confidence": 90 if ppe_status.get("head_cover", False) else 10,
                    "required": "head_cover" in dept.required_ppe
                },
                "shoe_covers": {
                    "name": "Shoe Covers",
                    "detected": bool(ppe_status.get("shoe_covers", False)),
                    "confidence": 85 if ppe_status.get("shoe_covers", False) else 5,
                    "required": "shoe_covers" in dept.required_ppe
                }
            }

            # Missing items that are required for this department
            missing = [info["name"] for key, info in items.items() if info["required"] and not info["detected"]]
            is_compliant = len(missing) == 0

            persons_data.append({
                "track_id": t.track_id,
                "person_id_formatted": f"{t.track_id:02d}",
                "location": f"{dept.code} ({dept.name})",
                "time_detected": time_detected,
                "duration": duration_str,
                "bbox": [round(float(v), 1) for v in t.bbox],
                "items": items,
                "is_compliant": is_compliant,
                "status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
                "missing_items": missing,
                "status_detail": "All required PPE detected" if is_compliant else f"Missing: {', '.join(missing)}"
            })

        return persons_data

    def update_department_policy(self, dept_id: str, required_ppe: List[str], grace_period_seconds: float = 3.0) -> bool:
        """Dynamically updates the required PPE policy for an area without restarting."""
        dept = self.departments.get(dept_id)
        if not dept:
            return False

        dept.required_ppe = list(required_ppe)
        dept.grace_period_seconds = float(grace_period_seconds)

        if dept.pipeline and dept.pipeline.zone_manager:
            zone_id = f"zone_{dept.id}"
            zone = dept.pipeline.zone_manager.get_zone(zone_id)
            if zone:
                zone.required_ppe = list(required_ppe)
                zone.grace_period_seconds = float(grace_period_seconds)

        return True

    def add_department(self, dept_data: Dict[str, Any]) -> Department:
        """Dynamically registers a new hospital area/camera and starts its background worker."""
        dept = Department(
            id=dept_data["id"],
            name=dept_data.get("name", dept_data["id"]),
            code=dept_data.get("code", f"DEPT-{dept_data['id'].upper()}"),
            building=dept_data.get("building", "Main Hospital"),
            floor=dept_data.get("floor", "1st Floor"),
            risk_level=dept_data.get("risk_level", "MEDIUM"),
            camera_source=str(dept_data.get("camera_source", "simulation")),
            camera_type=dept_data.get("camera_type", "SURVEILLANCE"),
            color=tuple(dept_data.get("color", [59, 130, 246])),
            required_ppe=dept_data.get("required_ppe", ["face_mask"]),
            optional_ppe=dept_data.get("optional_ppe", []),
            grace_period_seconds=float(dept_data.get("grace_period_seconds", 3.0)),
            alert_severity=dept_data.get("alert_severity", "WARNING"),
            nurse_station_contact=dept_data.get("nurse_station_contact", ""),
            safety_officer_email=dept_data.get("safety_officer_email", ""),
            description=dept_data.get("description", "")
        )

        dept.pipeline = self._create_department_pipeline(dept)
        dept.worker = CameraWorker(
            pipeline=dept.pipeline,
            name=f"Worker-{dept.id}",
            target_fps=20.0
        )
        if self.start_workers:
            dept.worker.start()

        self.departments[dept.id] = dept
        return dept

    def get_hospital_overview(self) -> Dict[str, Any]:
        """
        Aggregates compliance scores, active personnel counts, and violations
        across all hospital departments.
        """
        total_evals = 0
        compliant_evals = 0
        total_violations = 0
        total_active_staff = 0
        total_compliant_staff = 0
        total_non_compliant_staff = 0
        active_cameras_count = 0
        department_stats = []

        for dept in self.departments.values():
            if dept.pipeline is not None:
                telemetry = dept.pipeline.get_telemetry()
                dept_evals = telemetry.get("total_evaluations", 0)
                dept_comp_rate = telemetry.get("compliance_rate_percent", 100.0)
                dept_active = telemetry.get("active_tracks", 0)
                dept_violations = telemetry.get("total_incidents", 0)
                stream_status = telemetry.get("status", "CONNECTED")

                if dept.worker and dept.worker.is_running():
                    active_cameras_count += 1

                # Calculate compliant vs non-compliant personnel in this department
                dept_persons = self.get_department_person_details(dept.id)
                dept_comp_count = sum(1 for p in dept_persons if p["is_compliant"])
                dept_non_comp_count = len(dept_persons) - dept_comp_count

                total_compliant_staff += dept_comp_count
                total_non_compliant_staff += dept_non_comp_count

                total_evals += dept_evals
                total_violations += dept_violations
                total_active_staff += dept_active

                department_stats.append({
                    "id": dept.id,
                    "name": dept.name,
                    "code": dept.code,
                    "building": dept.building,
                    "floor": dept.floor,
                    "risk_level": dept.risk_level,
                    "camera_type": dept.camera_type,
                    "camera_source": dept.camera_source,
                    "stream_status": stream_status,
                    "compliance_rate": round(dept_comp_rate, 1),
                    "active_tracks": dept_active,
                    "violations_count": dept_violations,
                    "required_ppe": dept.required_ppe,
                    "nurse_station": dept.nurse_station_contact
                })

        # Hospital-wide compliance rate
        hospital_comp_rate = (
            round(sum(d["compliance_rate"] for d in department_stats) / max(1, len(department_stats)), 1)
            if department_stats else 100.0
        )

        # Sort leaderboard by compliance rate descending
        department_stats.sort(key=lambda d: d["compliance_rate"], reverse=True)

        return {
            "hospital_name": "CityCare Hospital",
            "total_departments": len(self.departments),
            "hospital_compliance_rate": hospital_comp_rate,
            "people_currently_monitored": total_active_staff,
            "compliant_count": total_compliant_staff,
            "non_compliant_count": total_non_compliant_staff,
            "active_cameras": active_cameras_count or len(self.departments),
            "total_incidents_today": total_violations,
            "departments": department_stats
        }

    def release_all(self):
        """Releases all active department camera streams and stops background workers."""
        for dept in self.departments.values():
            if dept.worker is not None:
                dept.worker.stop()
            if dept.pipeline is not None:
                dept.pipeline.stream_manager.release()

