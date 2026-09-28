"""
Unit and Integration tests for HospitalDepartmentManager, HospitalAlertDispatcher,
SafeCare area-specific PPE compliance policies, and dashboard summary APIs.
"""

import pytest
from fastapi.testclient import TestClient
from web.server import app, department_manager, alert_dispatcher
from core.alert_manager import Incident

client = TestClient(app)


def test_department_manager_loads_departments():
    depts = department_manager.get_all_departments()
    assert len(depts) >= 4

    dept_ids = [d.id for d in depts]
    # SafeCare departments: opd, gw, icu, or
    assert any(x in dept_ids for x in ["opd", "dept_triage"])
    assert any(x in dept_ids for x in ["gw", "dept_isolation"])
    assert any(x in dept_ids for x in ["icu", "dept_icu"])
    assert any(x in dept_ids for x in ["or", "dept_or"])


def test_department_specific_ppe_policies():
    or_dept = department_manager.get_department("or") or department_manager.get_department("dept_or")
    icu_dept = department_manager.get_department("icu") or department_manager.get_department("dept_icu")
    opd_dept = department_manager.get_department("opd") or department_manager.get_department("dept_triage")
    gw_dept = department_manager.get_department("gw") or department_manager.get_department("dept_isolation")

    assert or_dept is not None
    assert icu_dept is not None
    assert opd_dept is not None
    assert gw_dept is not None

    # OR requires sterile suite PPE (head cover, sterile gown, shoe covers, etc.)
    assert "head_cover" in or_dept.required_ppe
    assert "shoe_covers" in or_dept.required_ppe
    assert "gown" in or_dept.required_ppe

    # ICU requires airborne isolation PPE (mask, gloves, gown, eye protection, head cover)
    assert "face_mask" in icu_dept.required_ppe
    assert "eye_protection" in icu_dept.required_ppe
    assert "gloves" in icu_dept.required_ppe

    # OPD requires face mask only
    assert "face_mask" in opd_dept.required_ppe
    assert "shoe_covers" not in opd_dept.required_ppe

    # General Ward requires face mask and gloves
    assert "face_mask" in gw_dept.required_ppe
    assert "gloves" in gw_dept.required_ppe


def test_hospital_overview_aggregation():
    overview = department_manager.get_hospital_overview()
    assert "hospital_name" in overview
    assert "total_departments" in overview
    assert "hospital_compliance_rate" in overview
    assert "departments" in overview
    assert "people_currently_monitored" in overview
    assert "compliant_count" in overview
    assert "non_compliant_count" in overview
    assert len(overview["departments"]) >= 4


def test_dashboard_summary_endpoint():
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "people_monitored" in data
    assert "compliant" in data
    assert "non_compliant" in data
    assert "active_cameras" in data
    assert "total_incidents_today" in data


def test_department_person_details_endpoint():
    dept_id = "icu" if department_manager.get_department("icu") else "dept_icu"
    res = client.get(f"/api/departments/{dept_id}/person_details")
    assert res.status_code == 200
    data = res.json()
    assert data["department_id"] == dept_id
    assert "persons" in data
    assert "required_ppe" in data


def test_update_department_policy_endpoint():
    dept_id = "opd" if department_manager.get_department("opd") else "dept_triage"
    res = client.post(
        f"/api/departments/{dept_id}/policy",
        json={"required_ppe": ["face_mask", "gloves"], "grace_period_seconds": 4.5}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "gloves" in data["required_ppe"]


def test_compliance_trend_endpoint():
    res = client.get("/api/compliance/trend?area=icu")
    assert res.status_code == 200
    data = res.json()
    assert "labels" in data
    assert "compliant" in data
    assert "non_compliant" in data
    assert len(data["labels"]) == 6


def test_hospital_alert_dispatcher():
    test_incident = Incident(
        id="INC-TEST-01",
        track_id=42,
        zone_id="zone_or",
        zone_name="Operating Room",
        zone_code="OR-01",
        severity="CRITICAL",
        timestamp="2026-09-21T12:00:00",
        state="NON_COMPLIANT",
        missing_items=["face_mask", "head_cover"],
        improper_items=[],
        dwell_seconds=5.0,
        details="Missing sterile cap and mask in OR"
    )

    dispatch = alert_dispatcher.dispatch_incident_alert(
        incident=test_incident,
        department_name="Operating Room",
        department_code="OR-01",
        nurse_station="Ext. 4301",
        safety_email="or_safety@hospital.org"
    )

    assert dispatch["clinical_alert"]["type"] == "INFECTION_CONTROL_PPE_BREACH"
    assert dispatch["department"]["code"] == "OR-01"
    assert "SHIFT" in dispatch["shift"]
