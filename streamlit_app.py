"""
Streamlit Control Center for Hospital PPE and Face Mask Compliance Monitoring.
Supports real-time webcam/simulation, batch image/video upload inspection,
interactive zone policy tuning, and incident history review.
"""

import time
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.pipeline import PPECompliancePipeline
from core.detector import DetectorFactory, CLASS_PERSON
from core.zone_manager import ZoneManager
from core.compliance import ComplianceState

st.set_page_config(
    page_title="Hospital PPE Compliance Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #06b6d4;
    }
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(source: str, backend: str):
    return PPECompliancePipeline(source=source, backend=backend)


# Sidebar Configuration
st.sidebar.title("🏥 Hospital PPE Monitor")
st.sidebar.caption("Infection Control & Compliance AI")

mode = st.sidebar.radio(
    "Select Operating Mode",
    ["📹 Live Surveillance Stream", "🖼️ Image File Inspection", "🎬 Video File Inspection", "🛡️ Zone Policies & Rules", "📋 Incident Log"]
)

backend_choice = st.sidebar.selectbox(
    "Detection Engine Backend",
    ["auto", "heuristic", "simulation", "onnx"],
    index=0,
    help="Heuristic runs on pure CPU using OpenCV; Simulation uses ground-truth scenario generator."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Hospital Facility:** Metropolitan General")
st.sidebar.markdown("**System Version:** v1.0.0 (Production Ready)")


# ==========================================
# 1. LIVE SURVEILLANCE STREAM
# ==========================================
if mode == "📹 Live Surveillance Stream":
    st.title("📹 Real-Time Hospital PPE Compliance Surveillance")

    source_choice = st.selectbox(
        "Surveillance Feed Input",
        ["simulation", "0 (Webcam)", "1 (External Camera)"],
        index=0
    )
    clean_source = "simulation" if "simulation" in source_choice else source_choice.split(" ")[0]

    col1, col2, col3, col4 = st.columns(4)
    metric_comp = col1.empty()
    metric_active = col2.empty()
    metric_viol = col3.empty()
    metric_evals = col4.empty()

    st.markdown("---")
    video_placeholder = st.empty()

    run_stream = st.checkbox("Start Live Stream", value=True)

    if run_stream:
        pipeline = get_pipeline(source=clean_source, backend=backend_choice)

        stop_btn = st.button("Stop Stream")
        frame_counter = 0

        while run_stream and not stop_btn:
            success, frame, results, incidents = pipeline.process_next_frame()
            if not success or frame is None:
                st.warning("Unable to fetch video frame.")
                break

            frame_counter += 1
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update live metrics every 5 frames
            if frame_counter % 5 == 0:
                stats = pipeline.alert_manager.get_analytics_stats()
                metric_comp.metric("Overall Compliance", f"{stats['compliance_rate_percent']}%")
                metric_active.metric("Active Personnel", len(pipeline.tracker.tracks))
                metric_viol.metric("Violations Logged", stats['total_incidents'])
                metric_evals.metric("Checks Completed", stats['total_evaluations'])

            time.sleep(0.03)


# ==========================================
# 2. IMAGE FILE INSPECTION
# ==========================================
elif mode == "🖼️ Image File Inspection":
    st.title("🖼️ Single Image PPE Compliance Audit")
    st.write("Upload a hospital surveillance image or staff photo to inspect PPE compliance against hospital zones.")

    uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run detection & association
        detector = DetectorFactory.create_detector(backend="heuristic")
        zone_mgr = ZoneManager("config/zones.yaml")

        with st.spinner("Analyzing image with Computer Vision PPE pipeline..."):
            pipeline = PPECompliancePipeline(source="simulation", backend="heuristic")
            # Replace frame with uploaded image
            raw_dets = pipeline.detector.detect(img_bgr)
            profiles = pipeline.associator.associate(raw_dets)
            active_tracks = pipeline.tracker.update(profiles)

            annotated = zone_mgr.draw_zones_overlay(img_bgr.copy(), alpha=0.2)
            results = []

            for p in active_tracks:
                z = zone_mgr.find_zone_for_person(p.last_detection, img_bgr.shape[1], img_bgr.shape[0])
                res = pipeline.compliance_engine.evaluate_person(p, z, 1)
                results.append(res)
                annotated = pipeline.compliance_engine.annotate_frame(annotated, p, res)

        col_img, col_data = st.columns([1.5, 1])
        with col_img:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated PPE Assessment", use_container_width=True)

        with col_data:
            st.subheader("Compliance Findings")
            if not results:
                st.info("No persons detected in the image.")
            else:
                for idx, res in enumerate(results):
                    zone_str = res.zone.name if res.zone else "Unmonitored Area"
                    st.markdown(f"### Person #{res.track_id} ({zone_str})")
                    if res.state == ComplianceState.COMPLIANT:
                        st.success(f"State: {res.state.value} (Score: {res.compliance_score*100:.0f}%)")
                    elif res.state == ComplianceState.NON_COMPLIANT:
                        st.error(f"State: {res.state.value} (Score: {res.compliance_score*100:.0f}%)")
                    else:
                        st.warning(f"State: {res.state.value} (Score: {res.compliance_score*100:.0f}%)")

                    st.markdown(f"**Present Items:** {', '.join(res.present_items) if res.present_items else 'None'}")
                    if res.missing_items:
                        st.markdown(f"**Missing Items:** :red[{', '.join(res.missing_items)}]")
                    if res.improper_items:
                        st.markdown(f"**Improper Items:** :orange[{', '.join(res.improper_items)}]")
                    st.markdown("---")


# ==========================================
# 3. VIDEO FILE INSPECTION
# ==========================================
elif mode == "🎬 Video File Inspection":
    st.title("🎬 Surveillance Video Inspection & Audit")
    st.write("Upload pre-recorded MP4/AVI surveillance footage to process compliance across all frames.")

    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = f"data/samples/temp_{video_file.name}"
        os.makedirs("data/samples", exist_ok=True)
        with open(tfile, "wb") as f:
            f.write(video_file.read())

        st.success(f"Video saved. Initializing pipeline on `{tfile}`...")
        pipeline = PPECompliancePipeline(source=tfile, backend="heuristic")

        progress_bar = st.progress(0)
        frame_box = st.empty()

        cap = cv2.VideoCapture(tfile)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        cap.release()

        processed_frames = 0
        while True:
            success, frame, results, incidents = pipeline.process_next_frame()
            if not success or frame is None:
                break

            processed_frames += 1
            if processed_frames % 2 == 0:
                frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                progress_bar.progress(min(1.0, processed_frames / total_frames))

        st.success(f"Completed video processing. Total frames analyzed: {processed_frames}")


# ==========================================
# 4. ZONE POLICIES & RULES
# ==========================================
elif mode == "🛡️ Zone Policies & Rules":
    st.title("🛡️ Designated Hospital Zones & Policy Matrix")
    st.write("Configure required medical PPE equipment for each department.")

    zone_mgr = ZoneManager("config/zones.yaml")
    zones = zone_mgr.get_all_zones()

    for z in zones:
        with st.expander(f"[{z.code}] {z.name} (Risk: {z.risk_level})", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Description:** {z.description}")
                st.markdown(f"**Grace Period:** {z.grace_period_seconds} seconds")
                st.markdown(f"**Alert Severity:** {z.alert_severity}")
            with col_b:
                st.markdown("**Mandatory Required PPE:**")
                for req in z.required_ppe:
                    st.markdown(f"- ✅ `{req}`")
                if z.optional_ppe:
                    st.markdown("**Optional / Recommended PPE:**")
                    for opt in z.optional_ppe:
                        st.markdown(f"- ℹ️ `{opt}`")


# ==========================================
# 5. INCIDENT LOG
# ==========================================
elif mode == "📋 Incident Log":
    st.title("📋 PPE Non-Compliance Incident Log")

    from core.alert_manager import AlertManager
    alert_mgr = AlertManager()
    incidents = alert_mgr.get_recent_incidents(100)

    if not incidents:
        st.info("No recorded incidents. All hospital zones are compliant.")
    else:
        df = pd.DataFrame(incidents)
        st.dataframe(df[["id", "timestamp", "zone_name", "severity", "track_id", "missing_items", "dwell_seconds"]], use_container_width=True)

        st.subheader("Violation Evidence Gallery")
        cols = st.columns(3)
        for idx, inc in enumerate(incidents[:9]):
            with cols[idx % 3]:
                st.markdown(f"**{inc['id']}** ({inc['zone_name']})")
                st.caption(f"Track #{inc['track_id']} | Missing: {', '.join(inc['missing_items'])}")
                snap_path = os.path.join("data/incidents/snapshots", inc.get("snapshot_filename", ""))
                if os.path.exists(snap_path):
                    st.image(snap_path, use_container_width=True)
                else:
                    st.text("Snapshot not available")

