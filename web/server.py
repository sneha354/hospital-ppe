"""
FastAPI Server for Hospital PPE Compliance Monitoring.
Provides real-time MJPEG video feeds, REST endpoints for zone configuration,
incident inspection, analytics, and live camera device management.
"""

from typing import List, Dict, Any, Optional
import os
import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.pipeline import PPECompliancePipeline
from core.camera_worker import CameraWorker
from core.video_stream import CameraDeviceScanner
from core.zone_manager import Zone
from core.department_manager import HospitalDepartmentManager
from core.hospital_integrations import HospitalAlertDispatcher

app = FastAPI(
    title="Hospital PPE & Face Mask Compliance Monitor",
    description="Real-time Computer Vision system for monitoring medical PPE requirements in designated hospital areas.",
    version="1.3.0"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "web", "templates")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "incidents", "snapshots")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize global pipeline, hospital department manager, and alert dispatcher
pipeline = PPECompliancePipeline()
global_camera_worker = CameraWorker(pipeline=pipeline, name="GlobalWorker", target_fps=25.0)
global_camera_worker.start()

department_manager = HospitalDepartmentManager(os.path.join(BASE_DIR, "config", "departments.yaml"))
alert_dispatcher = HospitalAlertDispatcher()

# Connect department incident notifications to alert dispatcher
for dept in department_manager.get_all_departments():
    if dept.worker:
        def make_dept_callback(d):
            return lambda incs: [
                alert_dispatcher.dispatch_incident_alert(
                    incident=inc,
                    department_name=d.name,
                    department_code=d.code,
                    nurse_station=d.nurse_station_contact,
                    safety_email=d.safety_officer_email
                ) for inc in incs
            ]
        dept.worker.on_incidents = make_dept_callback(dept)


# --- Pydantic Models ---
class ZonePayload(BaseModel):
    id: str
    name: str
    code: str
    risk_level: str
    color: List[int]
    polygon: List[List[float]]
    required_ppe: List[str]
    optional_ppe: Optional[List[str]] = []
    grace_period_seconds: Optional[float] = 3.0
    alert_severity: Optional[str] = "WARNING"
    description: Optional[str] = ""


class SourcePayload(BaseModel):
    source: str
    backend: Optional[str] = "auto"


class CameraTestPayload(BaseModel):
    source: str


class AreaPolicyPayload(BaseModel):
    required_ppe: List[str]
    grace_period_seconds: Optional[float] = 3.0


class AddDepartmentPayload(BaseModel):
    id: str
    name: str
    code: str
    building: Optional[str] = "Main Hospital"
    floor: Optional[str] = "1st Floor"
    risk_level: Optional[str] = "MEDIUM"
    camera_source: str
    required_ppe: List[str]
    grace_period_seconds: Optional[float] = 3.0


# --- HTML Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    depts = [d.to_dict() for d in department_manager.get_all_departments()]
    # Default to ICU (icu-01) as in reference mockup, or first department
    default_dept = next((d for d in depts if d["id"] == "icu"), depts[0] if depts else None)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "hospital_name": "CityCare Hospital",
            "departments": depts,
            "default_dept": default_dept,
            "overview": department_manager.get_hospital_overview(),
            "recent_incidents": pipeline.alert_manager.get_recent_incidents(10)
        }
    )


@app.get("/zones", response_class=HTMLResponse)
async def zones_view(request: Request):
    return templates.TemplateResponse(
        "zones.html",
        {
            "request": request,
            "zones": pipeline.zone_manager.get_all_zones()
        }
    )


@app.get("/incidents", response_class=HTMLResponse)
async def incidents_view(request: Request):
    return templates.TemplateResponse(
        "incidents.html",
        {
            "request": request,
            "incidents": pipeline.alert_manager.get_recent_incidents(100)
        }
    )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_view(request: Request):
    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "stats": pipeline.alert_manager.get_analytics_stats()
        }
    )


# --- Multi-Department Clinical Routes ---
@app.get("/departments", response_class=HTMLResponse)
async def departments_wall_view(request: Request):
    return templates.TemplateResponse(
        "departments.html",
        {
            "request": request,
            "overview": department_manager.get_hospital_overview()
        }
    )


@app.get("/departments/{dept_id}", response_class=HTMLResponse)
async def department_detail_view(request: Request, dept_id: str):
    dept = department_manager.get_department(dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    return templates.TemplateResponse(
        "department_detail.html",
        {
            "request": request,
            "department": dept.to_dict()
        }
    )


def generate_department_stream(dept_id: str):
    worker = department_manager.get_department_worker(dept_id)
    if not worker:
        return iter([])
    return worker.stream()


@app.get("/departments/{dept_id}/video_feed")
async def department_video_feed(dept_id: str):
    dept = department_manager.get_department(dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    return StreamingResponse(
        generate_department_stream(dept_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/departments")
async def list_departments():
    return JSONResponse([d.to_dict() for d in department_manager.get_all_departments()])


@app.post("/api/departments/{dept_id}/camera")
async def update_department_camera(dept_id: str, payload: CameraTestPayload):
    success = department_manager.set_department_camera(dept_id, payload.source)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update department camera")
    return {"status": "success", "department_id": dept_id, "camera_source": payload.source}


@app.get("/api/hospital/overview")
async def hospital_overview():
    return JSONResponse(department_manager.get_hospital_overview())


@app.get("/api/dashboard/summary")
async def dashboard_summary():
    """Aggregates metrics for the top 5 KPI cards on the SafeCare dashboard."""
    overview = department_manager.get_hospital_overview()
    return JSONResponse({
        "people_monitored": overview.get("people_currently_monitored", 0),
        "compliant": overview.get("compliant_count", 0),
        "non_compliant": overview.get("non_compliant_count", 0),
        "active_cameras": overview.get("active_cameras", 4),
        "total_incidents_today": overview.get("total_incidents_today", 0),
        "hospital_compliance_rate": overview.get("hospital_compliance_rate", 100.0)
    })


@app.get("/api/departments/{dept_id}/person_details")
async def get_department_person_details_api(dept_id: str):
    """Returns real-time person-level PPE evaluation details for the active camera."""
    dept = department_manager.get_department(dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    persons = department_manager.get_department_person_details(dept_id)
    return JSONResponse({
        "department_id": dept.id,
        "department_name": dept.name,
        "department_code": dept.code,
        "required_ppe": dept.required_ppe,
        "persons": persons,
        "total_persons": len(persons)
    })


@app.post("/api/departments/{dept_id}/policy")
async def update_department_policy_api(dept_id: str, payload: AreaPolicyPayload):
    """Updates an area's required PPE policy dynamically."""
    success = department_manager.update_department_policy(
        dept_id=dept_id,
        required_ppe=payload.required_ppe,
        grace_period_seconds=payload.grace_period_seconds or 3.0
    )
    if not success:
        raise HTTPException(status_code=404, detail="Department not found")
    dept = department_manager.get_department(dept_id)
    return {
        "status": "success",
        "department_id": dept_id,
        "required_ppe": dept.required_ppe,
        "grace_period_seconds": dept.grace_period_seconds
    }


@app.post("/api/departments/add")
async def add_department_api(payload: AddDepartmentPayload):
    """Adds a new hospital camera/area dynamically."""
    dept = department_manager.add_department(payload.dict())
    return {"status": "success", "department": dept.to_dict()}


@app.get("/api/compliance/trend")
async def compliance_trend(area: Optional[str] = "all"):
    """Returns hourly compliance trend data for the line chart."""
    return JSONResponse({
        "labels": ["8AM", "10AM", "12PM", "2PM", "4PM", "6PM"],
        "compliant": [24, 30, 28, 32, 26, 30],
        "non_compliant": [10, 8, 12, 7, 9, 8]
    })


@app.get("/api/sample_videos")
async def list_sample_videos():
    """Returns list of pre-packaged hospital scenario demonstration videos."""
    samples_dir = os.path.join(BASE_DIR, "data", "samples")
    if not os.path.exists(samples_dir):
        return JSONResponse([])

    friendly_metadata = {
        "clinical_person_compliant_scrubs_ppe.mp4": {
            "title": "Clinical Worker (Fully Compliant: Scrubs, Mask, Gloves, Bouffant Cap)",
            "category": "COMPLIANT DEMO",
            "badge_color": "green",
            "description": "Healthcare worker wearing medical scrubs, surgical blue mask, nitrile gloves, bouffant cap, and shoe covers."
        },
        "clinical_person_non_compliant_casual.mp4": {
            "title": "Civilian in Casual Clothes (Non-Compliant: No Scrubs, No Mask, Bare Hands)",
            "category": "VIOLATION DEMO",
            "badge_color": "red",
            "description": "Visitor in red street jacket and jeans with bare face (no mask), bare hands (no gloves), and no cap."
        },
        "icu_isolation_demo.mp4": {
            "title": "ICU Isolation Suite (Nurse in Yellow Gown/N95 + Entering Unmasked Visitor)",
            "category": "ICU AIIR",
            "badge_color": "amber",
            "description": "Active negative pressure ICU isolation room with compliant nurse and non-compliant visitor."
        },
        "triage_corridor_demo.mp4": {
            "title": "Triage Corridor (Compliant Staff + Non-compliant Patients)",
            "category": "OPD / TRIAGE",
            "badge_color": "blue",
            "description": "Emergency triage intake corridor monitoring respiratory mask adherence."
        },
        "or_surgery_demo.mp4": {
            "title": "Operating Room Sterile Suite (Surgical Team in Full Barrier Gear)",
            "category": "SURGICAL OR",
            "badge_color": "purple",
            "description": "Sterile surgical suite requiring complete 6-item barrier isolation."
        },
        "cssd_cleanroom_demo.mp4": {
            "title": "CSSD Sterile Cleanroom Compounding Facility",
            "category": "CLEANROOM",
            "badge_color": "cyan",
            "description": "Cleanroom particulate-controlled environment with sterile compounding techs."
        }
    }

    videos = []
    for f in os.listdir(samples_dir):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            meta = friendly_metadata.get(f, {
                "title": f.replace("_", " ").replace(".mp4", "").title(),
                "category": "CUSTOM UPLOAD",
                "badge_color": "gray",
                "description": "Uploaded surveillance video stream."
            })
            videos.append({
                "filename": f,
                "path": os.path.join("data", "samples", f).replace("\\", "/"),
                "name": meta["title"],
                "category": meta["category"],
                "badge_color": meta["badge_color"],
                "description": meta["description"]
            })
    return JSONResponse(videos)


@app.post("/api/upload_video")
async def upload_video_file(file: UploadFile = File(...), department_id: Optional[str] = None):
    """Allows uploading custom hospital surveillance or CCTV video files with security validation."""
    samples_dir = os.path.join(BASE_DIR, "data", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Read content and enforce max upload size (250 MB)
    MAX_SIZE_BYTES = 250 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of 250 MB ({len(content)/(1024*1024):.1f} MB provided)."
        )

    # 3. Sanitize filename with UUID to prevent directory traversal
    safe_filename = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    dest_path = os.path.join(samples_dir, safe_filename)
    with open(dest_path, "wb") as buffer:
        buffer.write(content)

    rel_path = os.path.join("data", "samples", safe_filename).replace("\\", "/")

    if department_id:
        department_manager.set_department_camera(department_id, rel_path)
    else:
        # Update main pipeline and worker to stream this uploaded video
        global pipeline, global_camera_worker
        global_camera_worker.stop()
        pipeline.stream_manager.release()
        pipeline = PPECompliancePipeline(source=rel_path)
        global_camera_worker = CameraWorker(pipeline=pipeline, name="GlobalWorker", target_fps=25.0)
        global_camera_worker.start()

    return {
        "status": "success",
        "original_filename": file.filename,
        "filename": safe_filename,
        "path": rel_path,
        "message": f"Uploaded and connected video {safe_filename} successfully."
    }


# --- Video Streaming Endpoints ---
def generate_mjpeg_stream():
    """Generates continuous MJPEG multipart stream directly from background CameraWorker."""
    return global_camera_worker.stream()


@app.get("/video_feed")
async def video_feed():
    """MJPEG live video stream."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/snapshot")
async def stream_snapshot():
    """Captures a single raw frame without overlays for the zone editor canvas."""
    success, frame, _ = pipeline.stream_manager.read_frame()
    if not success or frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ret, buffer = cv2.imencode(".jpg", frame)
    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")


# --- Production Live Camera Endpoints ---
@app.get("/api/camera/devices")
async def scan_camera_devices():
    """Scans physical hardware for connected USB webcams and DirectShow video devices."""
    devices = CameraDeviceScanner.scan_devices(max_devices=4)
    return JSONResponse(devices)


@app.post("/api/camera/test")
async def test_camera_connection(payload: CameraTestPayload):
    """Tests connection to an RTSP URL or USB camera index without switching global pipeline."""
    src = int(payload.source) if payload.source.isdigit() else payload.source
    cap = None
    try:
        if isinstance(src, str) and src.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024|stimeout;3000000"
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        elif isinstance(src, int) and os.name == 'nt':
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)

        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return {
                    "status": "success",
                    "connected": True,
                    "resolution": f"{w}x{h}",
                    "message": "Camera successfully connected and frame verified."
                }
            cap.release()

        return {
            "status": "error",
            "connected": False,
            "message": "Unable to read frame from camera. Check URL or device index."
        }
    except Exception as e:
        if cap is not None:
            cap.release()
        return {
            "status": "error",
            "connected": False,
            "message": f"Connection exception: {str(e)}"
        }


@app.get("/api/camera/status")
async def get_camera_status():
    """Returns live camera status, capture FPS, and latency."""
    return JSONResponse(pipeline.stream_manager.get_status())


# --- Telemetry & Stats API Endpoints ---
@app.get("/api/stats")
async def get_stats():
    """Returns live telemetry, compliance rates, and stream performance."""
    telemetry = pipeline.get_telemetry()
    analytics = pipeline.alert_manager.get_analytics_stats()
    telemetry["current_source"] = telemetry["source"]
    telemetry["zone_distribution"] = analytics.get("zone_distribution", {})
    telemetry["missing_item_distribution"] = analytics.get("missing_item_distribution", {})
    telemetry["severity_distribution"] = analytics.get("severity_distribution", {})
    telemetry["zones_count"] = len(pipeline.zone_manager.zones)
    return JSONResponse(telemetry)


@app.get("/api/zones")
async def list_zones():
    return JSONResponse([z.to_dict() for z in pipeline.zone_manager.get_all_zones()])


@app.post("/api/zones")
async def save_zone(payload: ZonePayload):
    new_zone = Zone(
        id=payload.id,
        name=payload.name,
        code=payload.code,
        risk_level=payload.risk_level,
        color=tuple(payload.color),
        polygon=[(p[0], p[1]) for p in payload.polygon],
        required_ppe=payload.required_ppe,
        optional_ppe=payload.optional_ppe or [],
        grace_period_seconds=payload.grace_period_seconds or 3.0,
        alert_severity=payload.alert_severity or "WARNING",
        description=payload.description or ""
    )
    pipeline.zone_manager.add_or_update_zone(new_zone)
    return {"status": "success", "zone": new_zone.to_dict()}


@app.delete("/api/zones/{zone_id}")
async def remove_zone(zone_id: str):
    deleted = pipeline.zone_manager.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"status": "deleted", "zone_id": zone_id}


@app.get("/api/incidents")
async def list_incidents(limit: int = 50):
    return JSONResponse(pipeline.alert_manager.get_recent_incidents(limit))


@app.get("/api/incidents/{incident_id}/snapshot")
async def get_incident_snapshot(incident_id: str):
    incident = pipeline.alert_manager.get_incident_by_id(incident_id)
    if not incident or not incident.snapshot_filename:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    snap_path = os.path.join(SNAPSHOTS_DIR, incident.snapshot_filename)
    if not os.path.exists(snap_path):
        raise HTTPException(status_code=404, detail="Image file missing")

    return FileResponse(snap_path, media_type="image/jpeg")


class IncidentStatusPayload(BaseModel):
    status: str
    user: Optional[str] = "Staff"
    notes: Optional[str] = ""


@app.post("/api/incidents/{incident_id}/status")
async def update_incident_status(incident_id: str, payload: IncidentStatusPayload):
    """Transitions incident status (NEW, ACKNOWLEDGED, UNDER_REVIEW, RESOLVED, FALSE_POSITIVE)."""
    updated = pipeline.alert_manager.update_incident_status(
        incident_id=incident_id,
        status=payload.status,
        user=payload.user or "Staff",
        notes=payload.notes or ""
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "success", "incident": updated.to_dict()}


@app.post("/api/incidents/cleanup")
async def cleanup_incidents(snapshot_days: int = 7, incident_days: int = 30):
    """Enforces data retention by purging snapshots and records older than configured days."""
    purged_snaps = pipeline.alert_manager.cleanup_old_snapshots(max_age_days=snapshot_days)
    purged_incs = pipeline.alert_manager.cleanup_old_incidents(max_age_days=incident_days)
    return {
        "status": "success",
        "purged_snapshots": purged_snaps,
        "purged_incidents": purged_incs
    }


@app.post("/api/pipeline/source")
async def switch_source(payload: SourcePayload):
    global pipeline, global_camera_worker
    # Gracefully stop worker and release stream
    global_camera_worker.stop()
    pipeline.stream_manager.release()
    pipeline = PPECompliancePipeline(source=payload.source, backend=payload.backend)
    global_camera_worker = CameraWorker(pipeline=pipeline, name="GlobalWorker", target_fps=25.0)
    global_camera_worker.start()
    return {
        "status": "success",
        "source": payload.source,
        "backend": payload.backend,
        "message": f"Switched to source: {payload.source}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
