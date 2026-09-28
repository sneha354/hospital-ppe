"""
Unit tests for episode-based violation alerting, deduplication, auto-resolution, and incident lifecycle transitions.
"""

import os
import shutil
import numpy as np
import pytest

from core.alert_manager import AlertManager, IncidentStatus
from core.compliance import ComplianceResult, ComplianceState
from core.tracker import TrackedPerson
from core.detector import Detection
from core.zone_manager import Zone


@pytest.fixture
def temp_alert_manager(tmp_path):
    snap_dir = str(tmp_path / "snapshots")
    log_dir = str(tmp_path / "logs")
    manager = AlertManager(snapshot_dir=snap_dir, log_dir=log_dir, cooldown_seconds=5.0)
    yield manager
    if os.path.exists(str(tmp_path)):
        shutil.rmtree(str(tmp_path), ignore_errors=True)


def test_episode_deduplication(temp_alert_manager):
    """Verify that a continuous violation results in exactly ONE incident episode."""
    manager = temp_alert_manager
    zone = Zone(
        id="zone_or",
        name="Operating Room 1",
        code="OR-01",
        risk_level="CRITICAL",
        color=(255, 0, 0),
        polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        required_ppe=["face_mask"]
    )

    det = Detection(bbox=(50, 50, 150, 250), class_name="person")
    person = TrackedPerson(track_id=7, bbox=(50, 50, 150, 250), last_detection=det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = 100.0

    # Step 1: Violation begins at t0 -> creates first incident
    res1 = ComplianceResult(
        track_id=7,
        zone=zone,
        state=ComplianceState.NON_COMPLIANT,
        compliance_score=0.0,
        missing_items=["face_mask"],
        dwell_seconds=5.0,
        should_alert=True
    )
    inc1 = manager.process_result(frame, person, res1, current_time_sec=t0)
    assert inc1 is not None
    assert inc1.status == IncidentStatus.NEW.value
    assert len(manager.incidents) == 1

    # Step 2: Person remains in violation for the next 50 frames (over 20 seconds)
    # None of these should spawn a duplicate incident!
    for step in range(1, 50):
        t_current = t0 + step * 0.5
        res_ongoing = ComplianceResult(
            track_id=7,
            zone=zone,
            state=ComplianceState.NON_COMPLIANT,
            compliance_score=0.0,
            missing_items=["face_mask"],
            dwell_seconds=5.0 + step * 0.5,
            should_alert=True
        )
        inc_dup = manager.process_result(frame, person, res_ongoing, current_time_sec=t_current)
        assert inc_dup is None  # Deduplicated!
        assert len(manager.incidents) == 1  # Still exactly 1 incident recorded!

    # Step 3: Person complies at t0 + 25.0s -> episode is auto-resolved
    res_compliant = ComplianceResult(
        track_id=7,
        zone=zone,
        state=ComplianceState.COMPLIANT,
        compliance_score=1.0,
        missing_items=[],
        dwell_seconds=30.0,
        should_alert=False
    )
    manager.process_result(frame, person, res_compliant, current_time_sec=t0 + 25.0)

    # Episode should now be marked RESOLVED
    assert len(manager.incidents) == 1
    resolved_inc = manager.incidents[0]
    assert resolved_inc.status == IncidentStatus.RESOLVED.value
    assert resolved_inc.resolved_by == "SYSTEM (Auto-Compliance)"
    assert resolved_inc.duration_seconds >= 24.0


def test_incident_lifecycle_transitions(temp_alert_manager):
    """Verify manual transitions: NEW -> ACKNOWLEDGED -> RESOLVED / FALSE_POSITIVE."""
    manager = temp_alert_manager
    zone = Zone(
        id="zone_triage",
        name="Triage",
        code="ED",
        risk_level="MEDIUM",
        color=(0, 255, 0),
        polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        required_ppe=["face_mask"]
    )

    det = Detection(bbox=(50, 50, 150, 250), class_name="person")
    person = TrackedPerson(track_id=12, bbox=(50, 50, 150, 250), last_detection=det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = ComplianceResult(
        track_id=12,
        zone=zone,
        state=ComplianceState.NON_COMPLIANT,
        compliance_score=0.0,
        missing_items=["face_mask"],
        dwell_seconds=3.5,
        should_alert=True
    )
    inc = manager.process_result(frame, person, res, current_time_sec=50.0)
    assert inc is not None
    assert inc.status == IncidentStatus.NEW.value

    # 1. Nurse Acknowledges
    ack_inc = manager.update_incident_status(inc.id, "ACKNOWLEDGED", user="Nurse Smith", notes="Notified staff member")
    assert ack_inc.status == IncidentStatus.ACKNOWLEDGED.value
    assert ack_inc.acknowledged_by == "Nurse Smith"
    assert ack_inc.acknowledged_at is not None

    # 2. Safety Officer Resolves
    res_inc = manager.update_incident_status(inc.id, "RESOLVED", user="Officer Chen", notes="Mask donned correctly")
    assert res_inc.status == IncidentStatus.RESOLVED.value
    assert res_inc.resolved_by == "Officer Chen"
    assert res_inc.resolved_at is not None

