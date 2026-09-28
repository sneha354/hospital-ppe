"""
End-to-End Hospital PPE Monitoring Pipeline.
Orchestrates threaded zero-latency frame capture, detection, anatomical association,
tracking, zone containment, compliance evaluation, alert triggering, and visual HUD annotation.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import cv2
import numpy as np

from core.detector import DetectorFactory, BaseDetector, SyntheticSimulatorDetector
from core.associator import AnatomicalAssociator
from core.tracker import PersonTracker, TrackedPerson
from core.zone_manager import ZoneManager
from core.compliance import ComplianceEngine, ComplianceResult
from core.alert_manager import AlertManager, Incident
from core.video_stream import VideoStreamManager, CameraStatus


class PPECompliancePipeline:
    """
    Unified Pipeline coordinating vision, tracking, compliance, and live camera health.
    """

    def __init__(
        self,
        zones_config_path: str = "config/zones.yaml",
        settings_config_path: str = "config/settings.yaml",
        source: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        # 1. Initialize Zone Manager
        self.zone_manager = ZoneManager(zones_config_path)

        # 2. Load Settings
        import yaml
        import os
        settings = {}
        if os.path.exists(settings_config_path):
            with open(settings_config_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}

        sys_conf = settings.get("system", {})
        det_conf = settings.get("detection", {})
        trk_conf = settings.get("tracking", {})
        alt_conf = settings.get("alerts", {})
        live_conf = settings.get("live_camera", {})

        self.source = source or sys_conf.get("camera_source", "simulation")
        self.backend = backend or det_conf.get("backend", "auto")
        self.width = sys_conf.get("camera_width", 1280)
        self.height = sys_conf.get("camera_height", 720)
        self.target_fps = sys_conf.get("target_fps", 25.0)

        # 3. Video Stream Manager (Threaded Zero-Latency)
        self.stream_manager = VideoStreamManager(
            source=self.source,
            width=self.width,
            height=self.height,
            rtsp_transport=live_conf.get("rtsp_transport", "tcp"),
            buffer_size=live_conf.get("buffer_size", 1),
            reconnect_interval_sec=live_conf.get("reconnect_interval_sec", 2.0)
        )

        # 4. Detector Engine
        model_path = det_conf.get("model_path")
        conf_thresh = det_conf.get("confidence_threshold", 0.5)
        inf_size = (
            live_conf.get("inference_width", 640),
            live_conf.get("inference_height", 360)
        )

        effective_backend = "simulation" if self.source == "simulation" else self.backend
        self.detector = DetectorFactory.create_detector(
            backend=effective_backend,
            model_path=model_path,
            conf_threshold=conf_thresh,
            inference_size=inf_size
        )

        # 5. Associator, Tracker, Compliance, Alerting
        self.associator = AnatomicalAssociator()
        self.tracker = PersonTracker(
            iou_threshold=trk_conf.get("iou_threshold", 0.35),
            max_lost_frames=trk_conf.get("max_lost_frames", 30),
            history_window=trk_conf.get("temporal_window_frames", 20)
        )
        self.compliance_engine = ComplianceEngine(fps=self.target_fps)
        self.alert_manager = AlertManager(
            snapshot_dir=alt_conf.get("snapshot_dir", "data/incidents/snapshots"),
            log_dir=alt_conf.get("log_dir", "data/incidents"),
            cooldown_seconds=alt_conf.get("cooldown_seconds", 10.0)
        )

        self.frame_count = 0
        self.start_time = time.time()

        # Telemetry & Performance Tracking
        self.inference_fps: float = 0.0
        self._perf_history: List[float] = []
        self._last_cycle_time = time.time()

    def process_next_frame(self) -> Tuple[bool, np.ndarray, List[ComplianceResult], List[Incident]]:
        """
        Executes one full cycle of the compliance monitoring pipeline.
        Returns: (success, annotated_frame, compliance_results, new_incidents)
        """
        cycle_start = time.time()
        self.frame_count += 1

        # Step 1: Ingest frame from threaded camera or simulation
        success, frame, sim_detections = self.stream_manager.read_frame()
        if not success or frame is None:
            # Generate offline/reconnecting placeholder frame
            placeholder = self._render_offline_frame()
            return False, placeholder, [], []

        # Step 2: Object Detection
        if isinstance(self.detector, SyntheticSimulatorDetector):
            self.detector.set_current_simulated_detections(sim_detections)
            raw_detections = self.detector.detect(frame)
        else:
            raw_detections = self.detector.detect(frame)

        # Step 3: Anatomical Association (PPE -> Person)
        profiles = self.associator.associate(raw_detections)

        # Step 4: Multi-Person Tracking (using wall-clock monotonic time)
        monotonic_now = time.monotonic()
        active_tracks = self.tracker.update(profiles, current_time=monotonic_now)

        # Step 5: Render Zone Overlays onto frame
        annotated_frame = self.zone_manager.draw_zones_overlay(frame, alpha=0.18)

        compliance_results: List[ComplianceResult] = []
        new_incidents: List[Incident] = []

        # Step 6: Zone Containment, Compliance Evaluation, and Alerting
        for person in active_tracks:
            zone = self.zone_manager.find_zone_for_person(person.last_detection, self.width, self.height)
            result = self.compliance_engine.evaluate_person(person, zone, self.frame_count, current_time=monotonic_now)
            compliance_results.append(result)
            annotated_frame = self.compliance_engine.annotate_frame(annotated_frame, person, result)

            incident = self.alert_manager.process_result(annotated_frame, person, result, monotonic_now)
            if incident:
                new_incidents.append(incident)

        # Auto-resolve episodes for personnel who departed
        self.alert_manager.close_inactive_tracks([p.track_id for p in active_tracks], monotonic_now)

        # Compute Inference FPS
        cycle_duration = time.time() - cycle_start
        if cycle_duration > 0:
            self._perf_history.append(1.0 / cycle_duration)
            if len(self._perf_history) > 30:
                self._perf_history.pop(0)
            self.inference_fps = round(sum(self._perf_history) / len(self._perf_history), 1)

        # Step 7: Render Global Status HUD at top of frame
        self._render_hud(annotated_frame, active_tracks, compliance_results)

        return True, annotated_frame, compliance_results, new_incidents

    def _render_offline_frame(self) -> np.ndarray:
        """Renders an animated reconnection/offline screen when camera drops."""
        frame = np.full((self.height, self.width, 3), 25, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        status_info = self.stream_manager.get_status()

        status_text = f"CAMERA STATUS: {status_info.get('status', 'RECONNECTING')}"
        source_text = f"Source: {self.source}"
        msg_text = "Attempting automatic zero-loss camera reconnection..."

        cv2.putText(frame, status_text, (self.width // 2 - 220, self.height // 2 - 30), font, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, source_text, (self.width // 2 - 180, self.height // 2 + 15), font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, msg_text, (self.width // 2 - 280, self.height // 2 + 55), font, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

        # Draw pulsating dot
        dot_radius = int(8 + 4 * np.sin(time.time() * 4))
        cv2.circle(frame, (self.width // 2, self.height // 2 + 100), dot_radius, (0, 165, 255), -1)

        return frame

    def _render_hud(self, frame: np.ndarray, tracks: List[TrackedPerson], results: List[ComplianceResult]):
        """Renders top telemetry bar with stream status, latency, and compliance statistics."""
        h, w = frame.shape[:2]
        hud_h = 36

        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 20, 28), -1)
        cv2.addWeighted(hud_overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (40, 50, 65), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Status badge (Green if connected)
        status_info = self.stream_manager.get_status()
        status = status_info.get("status", "CONNECTED")
        capture_fps = status_info.get("capture_fps", 0.0)
        latency_ms = status_info.get("latency_ms", 0.0)

        if status == "CONNECTED":
            status_color = (34, 197, 94)  # Green
            status_tag = f"LIVE | {capture_fps} FPS | {latency_ms:.0f}ms"
        elif status == "RECONNECTING":
            status_color = (0, 165, 255)  # Orange
            status_tag = "RECONNECTING..."
        else:
            status_color = (50, 50, 239)  # Red
            status_tag = "OFFLINE"

        # Left: Hospital branding + Stream Status
        cv2.putText(frame, "[MGH-AI]", (12, 23), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (85, 19), 4, status_color, -1)
        cv2.putText(frame, status_tag, (96, 23), font, 0.38, status_color, 1, cv2.LINE_AA)

        # Center: Active Personnel & Compliance Stats
        active_count = len(tracks)
        comp_count = sum(1 for r in results if r.state.value == "COMPLIANT")
        viol_count = sum(1 for r in results if r.state.value == "NON_COMPLIANT")

        cx = 360
        cv2.putText(frame, f"STAFF/VISITORS: {active_count}", (cx, 23), font, 0.40, (200, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"COMPLIANT: {comp_count}", (cx + 170, 23), font, 0.40, (34, 197, 94), 1, cv2.LINE_AA)
        cv2.putText(frame, f"VIOLATIONS: {viol_count}", (cx + 310, 23), font, 0.40, (50, 50, 239), 1, cv2.LINE_AA)

        # Right: Inference Performance & Source Name
        src_label = str(self.source)
        if len(src_label) > 20:
            src_label = src_label[:17] + "..."
        right_text = f"AI: {self.inference_fps} FPS | {src_label.upper()}"
        cv2.putText(frame, right_text, (w - 240, 23), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry dictionary for web dashboard."""
        status_info = self.stream_manager.get_status()
        analytics = self.alert_manager.get_analytics_stats()
        return {
            "source": self.source,
            "status": status_info.get("status", "CONNECTED"),
            "capture_fps": status_info.get("capture_fps", 0.0),
            "inference_fps": self.inference_fps,
            "latency_ms": status_info.get("latency_ms", 0.0),
            "active_tracks": len(self.tracker.tracks),
            "frame_count": self.frame_count,
            "compliance_rate_percent": analytics.get("compliance_rate_percent", 100.0),
            "total_incidents": analytics.get("total_incidents", 0),
            "total_evaluations": analytics.get("total_evaluations", 0),
            "violation_evaluations": analytics.get("violation_evaluations", 0),
            "partial_evaluations": analytics.get("partial_evaluations", 0),
        }
