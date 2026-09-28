# Hospital Medical PPE & Face Mask Compliance Monitoring System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8.svg?logo=opencv)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)

An enterprise-grade, real-time Computer Vision solution engineered to monitor and enforce compliance with face masks and medical Personal Protective Equipment (PPE) requirements across designated hospital areas (ICU, Operating Rooms, Triage, Isolation Wards, Clean Rooms).

---

## Key Features

1. **Medical PPE Item Detection**:
   - **Face Masks**: Surgical masks, N95 respirators, and face shields. Distinguishes between **Compliant** (covering nose and mouth), **Improperly Worn** (chin-strap/exposed nose), and **No Mask**.
   - **Eye Protection**: Protective goggles and full face shields.
   - **Body Protection**: Isolation gowns, surgical gowns, and clean lab coats.
   - **Hand Protection**: Nitrile (blue/purple) and latex (yellow/white) medical gloves.
   - **Head & Foot Protection**: Bouffant/surgical caps and disposable shoe covers.

2. **Hospital Zone Policy Engine**:
   - Multi-zone surveillance with configurable risk tiers (**CRITICAL**, **HIGH**, **MEDIUM**, **LOW**).
   - Interactive Polygon ROI Editor: Draw arbitrary polygonal surveillance zones directly over live camera streams.
   - Per-zone compliance requirements, alert severities, and entry grace periods.

3. **Anatomical PPE-to-Person Association**:
   - Maps detected PPE items directly to the wearing individual using normalized anatomical landmarks (Head $0-30\%$, Torso $20-80\%$, Hands $35-85\%$, Feet $80-100\%$).
   - Prevents false compliance from discarded PPE items lying on counters, tables, or floors.

4. **Multi-Object Tracking & Temporal Consistency**:
   - IoU-based tracking with Kalman/linear trajectory smoothing and persistent Person Track IDs.
   - Sliding temporal window buffer (15-30 frames) eliminates false alarms caused by momentary head turns or brief equipment occlusions.
   - Entry grace period countdowns for healthcare workers donning equipment in airlocks.

5. **Incident Recording & Evidence Capture**:
   - Automatic capture of high-resolution violation snapshots with bounding boxes, timestamps, and missing items.
   - Alert deduplication and cooldown intervals.
   - Audit log with CSV/JSON export for hospital infection control committees.

6. **Dual Control Interfaces**:
   - **Web Control Room (FastAPI + HTML5 Canvas)**: Low-latency MJPEG streaming, real-time alert ticker with audio alarm, interactive zone drawing canvas, and Chart.js analytics.
   - **Streamlit Control Center**: Single image / video file inspection, live webcam analysis, and policy inspection.

7. **Flexible Multi-Backend Inference**:
   - **Heuristic OpenCV (Default Pure-CPU)**: Runs at 60+ FPS immediately on standard CPUs without requiring external weights or GPUs.
   - **ONNX YOLO DNN**: Supports loading custom-trained YOLOv8/YOLO11 ONNX models.
   - **Synthetic Hospital Simulation**: Built-in realistic multi-zone simulation generator for instant verification, testing, and demos without physical cameras.

---

## System Architecture

```
                               ┌────────────────────────┐
                               │  Hospital CCTV / RTSP  │
                               │  Webcam / Simulation   │
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
                               │  (IoU + Temporal Buffer│
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
│ (Snapshots & Auditing) │             │  (MJPEG / WebSockets)  │
└───────────┬────────────┘             └───────────┬────────────┘
            │                                     │
            ▼                                     ▼
   ┌─────────────────┐                  ┌───────────────────┐
   │ Incidents JSON  │                  │ FastAPI Dashboard │
   │ Snapshots (JPG) │                  │  & Streamlit App  │
   └─────────────────┘                  └───────────────────┘
```

---

## Pre-Configured Hospital Zones

| Zone ID | Zone Name | Risk Level | Mandatory PPE | Grace Period | Alert Severity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `zone_icu` | **ICU / Airborne Infection Isolation** | `HIGH` | Face Mask, Eye Protection, Gown, Gloves | 3.0s | `CRITICAL` |
| `zone_or` | **Operating Room / Sterile Suite** | `CRITICAL` | Mask, Cap, Gown, Gloves, Eye Protection, Shoe Covers | 1.5s | `CRITICAL` |
| `zone_triage` | **General Ward & Triage** | `MEDIUM` | Face Mask | 5.0s | `WARNING` |
| `zone_public` | **Corridors / Reception** | `LOW` | None (Optional Mask) | 10.0s | `INFO` |

---

## Quickstart Guide

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
- **Live Control Room**: `http://localhost:8000/`
- **Zone Polygon Editor**: `http://localhost:8000/zones`
- **Incident Audit Log**: `http://localhost:8000/incidents`
- **Infection Control Analytics**: `http://localhost:8000/analytics`

### 3. Launch Streamlit Control Center

```powershell
python main.py run-streamlit --port 8501
```

Navigate to `http://localhost:8501` to upload images/videos or run live webcam monitoring.

### 4. Run Automated Simulation Verification

```powershell
python main.py demo-simulation --frames 100
```

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video_feed` | `GET` | MJPEG video stream with bounding boxes and compliance overlays |
| `/api/stream/snapshot` | `GET` | Single snapshot frame for canvas ROI calibration |
| `/api/stats` | `GET` | Live telemetry, compliance rate, and incident counts |
| `/api/zones` | `GET` | List all configured hospital zones and policies |
| `/api/zones` | `POST` | Create or update a zone with custom polygon coordinates |
| `/api/zones/{id}` | `DELETE` | Remove a hospital zone |
| `/api/incidents` | `GET` | Retrieve recent violation incident records |
| `/api/incidents/{id}/snapshot` | `GET` | Download/view evidence snapshot image |
| `/api/pipeline/source` | `POST` | Dynamically switch video source (`simulation`, `0`, `rtsp://...`) |

---

## Hospital RTSP Camera Integration

To connect live hospital CCTV cameras (Axis, Hikvision, Dahua, Milestone):

1. Edit `config/settings.yaml`:
   ```yaml
   system:
     camera_source: "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main"
     camera_width: 1920
     camera_height: 1080
     target_fps: 25
   ```
2. Or switch dynamically via the web dashboard dropdown!

---

## Testing

Run the automated pytest test suite:

```powershell
pytest tests/ -v
```

