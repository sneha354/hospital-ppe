# Hospital Medical PPE & Face Mask Compliance Monitoring System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8.svg?logo=opencv)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg?logo=sqlite)](https://sqlite.org)

An enterprise-grade, real-time Computer Vision solution engineered to monitor and enforce compliance with face masks and medical Personal Protective Equipment (PPE) requirements across designated hospital areas (ICU, Operating Rooms, Triage, Isolation Wards, Clean Rooms).

---

## 🌟 Key Features

1. **Medical PPE Item Detection**:
   - **Face Masks**: Surgical masks, N95 respirators, and face shields. Distinguishes between **Compliant** (covering nose and mouth), **Improperly Worn** (chin-strap/exposed nose), and **No Mask**.
   - **Eye Protection**: Protective goggles and full face shields.
   - **Body Protection**: Isolation gowns, surgical gowns, and clean lab coats.
   - **Hand Protection**: Nitrile (blue/purple) and latex (yellow/white) medical gloves.
   - **Head & Foot Protection**: Bouffant/surgical caps and disposable shoe covers.

2. **SafeCare Unified Multi-Page Clinical UI**:
   - **Persistent Navigation Sidebar**: Seamless left navigation across all pages (Dashboard, Live Monitoring, Areas & Policies, Incidents, Reports).
   - **Main Control Room**: Centralized live feed with instant ward camera switcher (`OPD-01`, `GW-01`, `ICU-01`, `OR-01`, and `ALL WARDS 4-Grid`).
   - **Multi-Department Surveillance Wall**: Matrix view displaying concurrent live video streams across ICU, General Ward, OPD, and Operating Room.
   - **Real-Time Video Ingestion**: Insert and evaluate clinical demo scenarios, upload CCTV recordings (`.mp4`, `.avi`), or bind RTSP / USB webcam sources on the fly.
   - **Interactive Polygon ROI Editor**: Draw arbitrary polygonal surveillance zones directly over live camera streams using HTML5 Canvas.

3. **Hospital Zone Policy Engine**:
   - Multi-zone surveillance with configurable risk tiers (**CRITICAL**, **HIGH**, **MEDIUM**, **LOW**).
   - Per-zone compliance requirements, alert severities, and entry grace periods.

4. **Anatomical PPE-to-Person Association**:
   - Maps detected PPE items directly to the wearing individual using normalized anatomical landmarks (Head $0-30\%$, Torso $20-80\%$, Hands $35-85\%$, Feet $80-100\%$).
   - Prevents false compliance from discarded PPE items lying on counters, tables, or floors.

5. **Multi-Object Tracking & Temporal Consistency**:
   - IoU-based tracking with trajectory smoothing and persistent Person Track IDs.
   - Sliding temporal window buffer (15-30 frames) eliminates false alarms caused by momentary head turns or brief equipment occlusions.
   - Entry grace period countdowns for healthcare workers donning equipment in airlocks.

6. **Incident Audit Log & Evidence Retention**:
   - High-performance **SQLite Database** (`hospital_ppe.db`) with WAL (Write-Ahead Logging) and thread-safe persistence.
   - **HIPAA Privacy Protection**: Personnel facial identities are automatically Gaussian-blurred on all violation evidence snapshots.
   - Episode lifecycle transitions (`NEW` &rarr; `ACKNOWLEDGED` &rarr; `RESOLVED` &rarr; `FALSE_POSITIVE`).
   - One-click CSV audit export and automated retention policy cleanup (7-day snapshots, 30-day logs).

---

## 🏛️ System Architecture

```
                               ┌────────────────────────┐
                               │  Hospital CCTV / RTSP  │
                               │  Webcam / Video File   │
                               └───────────┬────────────┘
                                           │ Video Frames
                                           ▼
                               ┌────────────────────────┐
                               │   BaseDetector Engine  │
                               │ (OpenCV / ONNX / Sim)  │
                               └───────────┬────────────┘
                                           │ Raw Detections (Person, Mask, Gown, Gloves, etc.)
                                           ▼
                               ┌────────────────────────┐
                               │  Anatomical Associator │
                               │ (Head/Torso/Hands/Feet)│
                               └───────────┬────────────┘
                                           │ Person PPE Profiles
                                           ▼
                               ┌────────────────────────┐
                               │     Person Tracker     │
                               │ (IoU + Temporal Buffer)│
                               └───────────┬────────────┘
                                           │ Tracked Persons + Dwell Times
                                           ▼
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌────────────────────────┐             ┌────────────────────────┐
│      Zone Manager      │             │   Compliance Engine    │
│(Point-in-Polygon ROIs) │             │ (Policy Match & Grace) │
└───────────┬────────────┘             └───────────┬────────────┘
            │                                     │
            └──────────────────┬──────────────────┘
                               │ Evaluated Results
                               ▼
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌────────────────────────┐             ┌────────────────────────┐
│     Alert Manager      │             │ Visual HUD & Annotator │
│ (Snapshots & SQLite)   │             │  (MJPEG Multi-Stream)  │
└───────────┬────────────┘             └───────────┬────────────┘
            │                                     │
            ▼                                     ▼
   ┌─────────────────┐                  ┌───────────────────┐
   │ SQLite Database │                  │ SafeCare Web App  │
   │ HIPAA Snapshots │                  │  & Streamlit App  │
   └─────────────────┘                  └───────────────────┘
```

---

## 🏥 Pre-Configured Hospital Zones & Departments

| Department Code | Department Name | Risk Level | Mandatory PPE | Grace Period | Alert Severity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ICU-01` | **Intensive Care Unit (ICU)** | `HIGH` | Face Mask, Eye Protection, Gown, Gloves | 2.5s | `CRITICAL` |
| `OR-01` | **Operating Room (OR)** | `CRITICAL` | Mask, Cap, Gown, Gloves, Eye Protection, Shoe Covers | 1.5s | `CRITICAL` |
| `GW-01` | **General Inpatient Ward** | `MEDIUM` | Face Mask | 3.5s | `WARNING` |
| `OPD-01` | **Outpatient & Triage** | `LOW` | Face Mask | 4.0s | `WARNING` |

---

## 🚀 Quickstart Guide

### 1. Installation

Clone repository and install dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Launch FastAPI Live Web Dashboard

```powershell
python main.py run-server --port 8000
```

Open your browser and navigate to:
- **Main Control Room**: `http://localhost:8000/`
- **Department Video Wall**: `http://localhost:8000/departments`
- **Zone Polygon Editor**: `http://localhost:8000/zones`
- **Incident Audit Log**: `http://localhost:8000/incidents`
- **Infection Control Analytics**: `http://localhost:8000/analytics`

### 3. Launch Streamlit Control Center

```powershell
python main.py run-streamlit --port 8501
```

Navigate to `http://localhost:8501` to upload test images/videos or run live webcam monitoring.

### 4. Run Automated Simulation Verification

```powershell
python main.py demo-simulation --frames 100
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | SafeCare Main Control Room dashboard |
| `/departments` | `GET` | Multi-department surveillance wall matrix |
| `/departments/{dept_id}` | `GET` | Single clinical department deep dive & camera ingestion |
| `/departments/{dept_id}/video_feed`| `GET` | Low-latency MJPEG video stream for a specific department |
| `/zones` | `GET` | Interactive HTML5 Canvas ROI polygon editor |
| `/incidents` | `GET` | HIPAA-aware incident audit log and evidence viewer |
| `/analytics` | `GET` | Chart.js infection control and compliance analytics |
| `/api/dashboard/summary` | `GET` | Live telemetry KPI metrics (People, Compliant, Violations, Cameras) |
| `/api/departments/{dept_id}/person_details` | `GET` | Tracked individuals, bounding boxes, and PPE checklist breakdown |
| `/api/departments/{dept_id}/camera` | `POST` | Dynamically update camera source for a department |
| `/api/departments/{dept_id}/policy` | `POST` | Update mandatory PPE requirements and grace period |
| `/api/departments/add` | `POST` | Dynamically register a new hospital ward or camera |
| `/api/upload_video` | `POST` | Upload recorded hospital CCTV footage (`.mp4`, `.avi`, `.mov`) |
| `/api/sample_videos` | `GET` | List available pre-configured clinical demonstration video scenarios |
| `/api/zones` | `GET`, `POST` | Retrieve or create/update hospital surveillance polygon zones |
| `/api/zones/{id}` | `DELETE` | Delete a hospital zone |
| `/api/incidents` | `GET` | Query recorded violation episodes from SQLite |
| `/api/incidents/{id}/snapshot` | `GET` | Fetch face-blurred JPEG evidence snapshot |
| `/api/incidents/{id}/status` | `POST` | Update incident status (`ACKNOWLEDGED`, `RESOLVED`, `FALSE_POSITIVE`) |
| `/api/incidents/cleanup` | `POST` | Enforce HIPAA data retention (purge snapshots > 7 days, logs > 30 days) |

---

## 📹 Hospital RTSP / IP Camera Integration

To connect live hospital CCTV cameras (Axis, Hikvision, Dahua, Milestone):

1. **Via Web Interface**:
   - Click **Insert / Change Video** on the main dashboard, select **Live Webcam / RTSP**, enter your RTSP stream URL (`rtsp://admin:pass@192.168.1.100:554/live`), and click **Connect RTSP**.
   - Or click **+ Add Camera** to add a new ward stream.

2. **Via Configuration File (`config/settings.yaml`)**:
   ```yaml
   system:
     camera_source: "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main"
     camera_width: 1280
     camera_height: 720
     target_fps: 25
   ```

---

## 🧪 Testing

Run unit and integration tests:

```powershell
python -m unittest discover tests/
```
