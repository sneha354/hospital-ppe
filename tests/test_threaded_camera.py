"""
Unit tests for ThreadedLiveCamera, CameraDeviceScanner, and VideoStreamManager.
"""

import time
import pytest
import numpy as np
from core.video_stream import (
    ThreadedLiveCamera,
    CameraStatus,
    CameraDeviceScanner,
    VideoStreamManager
)


def test_camera_status_enum():
    assert CameraStatus.CONNECTED == "CONNECTED"
    assert CameraStatus.RECONNECTING == "RECONNECTING"
    assert CameraStatus.OFFLINE == "OFFLINE"


def test_camera_device_scanner():
    devices = CameraDeviceScanner.scan_devices(max_devices=2)
    assert isinstance(devices, list)
    # Even if no physical webcam is plugged in, it must return a valid list without crashing
    for d in devices:
        assert "index" in d
        assert "name" in d
        assert "resolution" in d
        assert "fps" in d


def test_video_stream_manager_simulation_mode():
    manager = VideoStreamManager(source="simulation", width=640, height=360)
    assert manager._is_simulation is True
    assert manager.sim_generator is not None

    success, frame, dets = manager.read_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    assert len(dets) > 0  # Simulation agents

    status = manager.get_status()
    assert status["status"] == "CONNECTED"
    assert status["capture_fps"] == 25.0
    manager.release()


def test_threaded_live_camera_lifecycle():
    # Test initialization with a simulated or invalid source to verify safe startup and shutdown
    camera = ThreadedLiveCamera(
        source="999",  # Non-existent index
        width=320,
        height=240,
        reconnect_interval_sec=0.5,
        max_reconnect_attempts=2
    )
    assert camera._is_running is True

    # Check status transitions gracefully without hanging or throwing exceptions
    time.sleep(0.6)
    assert camera.status in [CameraStatus.RECONNECTING, CameraStatus.OFFLINE]

    success, frame, latency = camera.get_latest_frame()
    # Since device 999 doesn't exist, success should be False
    assert success is False
    assert frame is None

    # Safe stop
    camera.stop()
    assert camera._is_running is False
    assert camera.status == CameraStatus.OFFLINE

