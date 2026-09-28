"""
Unit tests for wall-clock monotonic timing, dwell calculations, and FPS-invariant temporal smoothing.
"""

import time
import pytest
from core.tracker import TrackedPerson, PersonTracker
from core.detector import Detection
from core.associator import PersonPPEProfile
from core.compliance import ComplianceEngine, ComplianceState
from core.zone_manager import Zone


def test_wall_clock_temporal_smoothing():
    """Verify temporal smoothing uses wall-clock sliding window rather than frame count."""
    det = Detection(bbox=(100, 100, 200, 300), class_name="person")
    person = TrackedPerson(
        track_id=1,
        bbox=(100, 100, 200, 300),
        last_detection=det,
        history_window_seconds=1.5
    )

    t0 = 1000.0

    # Person starts with NO mask at t0
    prof_no_mask = PersonPPEProfile(
        person_det=det,
        mask_status="no_mask"
    )
    for i in range(10):
        person.update_profile(prof_no_mask, frame_idx=i, current_time=t0 + i * 0.05)

    status = person.get_smoothed_ppe_status(window_seconds=1.5, current_time=t0 + 0.5)
    assert status["no_mask"] is True
    assert status["face_mask"] is False

    # At t0 + 2.0s (after old history expired), person puts on a mask
    t1 = t0 + 2.0
    prof_mask = PersonPPEProfile(
        person_det=det,
        mask_status="face_mask"
    )
    for i in range(10):
        person.update_profile(prof_mask, frame_idx=10 + i, current_time=t1 + i * 0.05)

    # In window [t1, t1 + 0.5], mask should now be smoothed to True
    status_new = person.get_smoothed_ppe_status(window_seconds=1.5, current_time=t1 + 0.5)
    assert status_new["face_mask"] is True
    assert status_new["no_mask"] is False


def test_compliance_engine_wall_clock_dwell_and_grace():
    """Verify ComplianceEngine calculates dwell time and grace periods using monotonic wall-clock seconds."""
    engine = ComplianceEngine(fps=25.0)

    zone = Zone(
        id="zone_icu",
        name="ICU Isolation",
        code="ICU",
        risk_level="HIGH",
        color=(0, 0, 255),
        polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        required_ppe=["face_mask"],
        grace_period_seconds=4.0
    )

    det = Detection(bbox=(100, 100, 200, 300), class_name="person")
    person = TrackedPerson(
        track_id=42,
        bbox=(100, 100, 200, 300),
        last_detection=det
    )

    t_entry = 100.0

    # Frame 1: Person enters zone at t_entry (dwell = 0.0s, in grace period)
    res1 = engine.evaluate_person(person, zone, current_frame=1, current_time=t_entry)
    assert res1.state == ComplianceState.GRACE_PERIOD
    assert res1.dwell_seconds == 0.0
    assert res1.grace_remaining_seconds == 4.0
    assert not res1.should_alert

    # Frame 2: Person has been in zone for 2.5s (still in grace period)
    res2 = engine.evaluate_person(person, zone, current_frame=2, current_time=t_entry + 2.5)
    assert res2.state == ComplianceState.GRACE_PERIOD
    assert res2.dwell_seconds == 2.5
    assert res2.grace_remaining_seconds == 1.5
    assert not res2.should_alert

    # Frame 3: Person has been in zone for 4.5s (grace expired, non-compliant)
    res3 = engine.evaluate_person(person, zone, current_frame=3, current_time=t_entry + 4.5)
    assert res3.state == ComplianceState.NON_COMPLIANT
    assert res3.dwell_seconds == 4.5
    assert res3.grace_remaining_seconds == 0.0
    assert res3.should_alert

