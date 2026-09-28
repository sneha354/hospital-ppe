"""
Unit tests for CameraWorker background execution, frame generation, and streaming.
"""

import time
import pytest
import numpy as np

from core.pipeline import PPECompliancePipeline
from core.camera_worker import CameraWorker


def test_camera_worker_lifecycle():
    """Verify CameraWorker starts, generates frames in background, and stops cleanly."""
    pipeline = PPECompliancePipeline(source="simulation")
    worker = CameraWorker(pipeline=pipeline, target_fps=30.0, name="TestWorker")

    assert not worker.is_running()
    assert worker.get_latest_jpeg() is None

    worker.start()
    assert worker.is_running()

    # Poll up to 2.0s for first frame to be generated
    latest_jpeg = None
    for _ in range(20):
        latest_jpeg = worker.get_latest_jpeg()
        if latest_jpeg is not None:
            break
        time.sleep(0.1)

    assert latest_jpeg is not None
    assert isinstance(latest_jpeg, bytes)
    assert len(latest_jpeg) > 1000

    status = worker.get_status()
    assert status["running"] is True
    assert status["frame_idx"] >= 1

    worker.stop(timeout=2.0)
    assert not worker.is_running()


def test_camera_worker_streaming_generator():
    """Verify streaming generator yields valid MJPEG chunks without busy-waiting."""
    pipeline = PPECompliancePipeline(source="simulation")
    worker = CameraWorker(pipeline=pipeline, target_fps=30.0, name="TestStreamWorker")
    worker.start()

    time.sleep(0.2)
    stream_gen = worker.stream()

    # Read first 3 frames from generator
    frames_received = 0
    for chunk in stream_gen:
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert chunk.endswith(b"\r\n")
        frames_received += 1
        if frames_received >= 3:
            break

    assert frames_received == 3
    worker.stop(timeout=2.0)
