"""
Unit tests for ZoneManager: Point-in-polygon containment, zone loading, and person anchoring.
"""

import pytest
import numpy as np
from core.zone_manager import ZoneManager, Zone
from core.detector import Detection, CLASS_PERSON


@pytest.fixture
def zone_mgr():
    zm = ZoneManager("config/zones.yaml")
    return zm


def test_zones_loaded(zone_mgr):
    zones = zone_mgr.get_all_zones()
    assert len(zones) >= 3
    zone_ids = [z.id for z in zones]
    assert "zone_icu" in zone_ids
    assert "zone_or" in zone_ids
    assert "zone_triage" in zone_ids


def test_point_in_polygon_containment(zone_mgr):
    icu = zone_mgr.get_zone("zone_icu")
    assert icu is not None

    # Bottom center inside ICU: x=200, y=400 (normalized ~0.15, 0.55 in 1280x720)
    inside_person = Detection(
        bbox=(150, 300, 250, 500),  # bottom_center = (200, 500) -> 200/1280=0.156, 500/720=0.69
        class_name=CLASS_PERSON
    )
    assigned_zone = zone_mgr.find_zone_for_person(inside_person, 1280, 720)
    assert assigned_zone is not None
    assert assigned_zone.id == "zone_icu"

    # Bottom center outside all zones: x=10, y=10
    outside_person = Detection(
        bbox=(5, 5, 15, 25),
        class_name=CLASS_PERSON
    )
    unmonitored = zone_mgr.find_zone_for_person(outside_person, 1280, 720)
    # Either None or unmonitored public depending on polygon
    if unmonitored:
        assert unmonitored.id == "zone_public"


def test_add_and_delete_zone(zone_mgr):
    test_zone = Zone(
        id="zone_test_quarantine",
        name="Quarantine Tent",
        code="QT-99",
        risk_level="CRITICAL",
        color=(255, 0, 0),
        polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)],
        required_ppe=["face_mask", "gown", "gloves"]
    )
    zone_mgr.add_or_update_zone(test_zone)
    assert zone_mgr.get_zone("zone_test_quarantine") is not None

    deleted = zone_mgr.delete_zone("zone_test_quarantine")
    assert deleted is True
    assert zone_mgr.get_zone("zone_test_quarantine") is None

