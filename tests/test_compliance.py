"""
Unit tests for ComplianceEngine: policy evaluation, grace period, and states.
"""

import pytest
from core.compliance import ComplianceEngine, ComplianceState
from core.tracker import TrackedPerson
from core.zone_manager import Zone
from core.associator import PersonPPEProfile
from core.detector import Detection, CLASS_PERSON, CLASS_FACE_MASK, CLASS_MASK_INCORRECT


@pytest.fixture
def icu_zone():
    return Zone(
        id="zone_icu",
        name="ICU",
        code="ICU-01",
        risk_level="HIGH",
        color=(255, 0, 0),
        polygon=[],
        required_ppe=["face_mask", "gown", "gloves"],
        grace_period_seconds=2.0
    )


def test_compliant_person(icu_zone):
    engine = ComplianceEngine(fps=20.0)

    person_det = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    prof = PersonPPEProfile(
        person_det=person_det,
        mask_status=CLASS_FACE_MASK,
        has_gown=True,
        has_gloves=True
    )

    track = TrackedPerson(
        track_id=1,
        bbox=person_det.bbox,
        last_detection=person_det,
        current_profile=prof
    )
    # Populate history to satisfy temporal window
    for _ in range(15):
        track.update_profile(prof, 1)

    result = engine.evaluate_person(track, icu_zone, current_frame=50)
    assert result.state == ComplianceState.COMPLIANT
    assert result.compliance_score == 1.0
    assert len(result.missing_items) == 0
    assert result.should_alert is False


def test_grace_period_handling(icu_zone):
    engine = ComplianceEngine(fps=20.0)

    # Person with no PPE who just entered (dwell: 0.5s < 2.0s grace)
    person_det = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    prof = PersonPPEProfile(person_det=person_det, mask_status=None, has_gown=False, has_gloves=False)

    track = TrackedPerson(
        track_id=2,
        bbox=person_det.bbox,
        last_detection=person_det,
        current_profile=prof,
        current_zone_id=icu_zone.id,
        zone_entry_frame=10
    )
    for _ in range(5):
        track.update_profile(prof, 10)

    # Current frame 20 (dwell = (20-10)/20 = 0.5s < 2.0s grace)
    result = engine.evaluate_person(track, icu_zone, current_frame=20)
    assert result.state == ComplianceState.GRACE_PERIOD
    assert result.should_alert is False


def test_violation_after_grace_period(icu_zone):
    engine = ComplianceEngine(fps=20.0)

    person_det = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    prof = PersonPPEProfile(person_det=person_det, mask_status=None, has_gown=False, has_gloves=False)

    track = TrackedPerson(
        track_id=3,
        bbox=person_det.bbox,
        last_detection=person_det,
        current_profile=prof,
        current_zone_id=icu_zone.id,
        zone_entry_frame=10
    )
    for _ in range(15):
        track.update_profile(prof, 10)

    # Current frame 100 (dwell = (100-10)/20 = 4.5s > 2.0s grace)
    result = engine.evaluate_person(track, icu_zone, current_frame=100)
    assert result.state == ComplianceState.NON_COMPLIANT
    assert result.should_alert is True
    assert "face_mask" in result.missing_items
    assert "gown" in result.missing_items
    assert "gloves" in result.missing_items
