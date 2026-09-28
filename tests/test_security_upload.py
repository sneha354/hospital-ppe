"""
Unit tests for video upload security, file extension whitelisting, and UUID sanitization.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient

from web.server import app

client = TestClient(app)


def test_upload_invalid_extension_rejected():
    """Verify that dangerous file formats like .exe, .py, .sh are rejected with HTTP 400."""
    fake_payload = io.BytesIO(b"malicious script content")
    response = client.post(
        "/api/upload_video",
        files={"file": ("exploit.py", fake_payload, "text/x-python")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]


def test_upload_valid_video_sanitized(tmp_path):
    """Verify that valid video formats are accepted and saved with a sanitized UUID filename."""
    dummy_video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00" + b"\x00" * 100
    file_obj = io.BytesIO(dummy_video_bytes)

    response = client.post(
        "/api/upload_video",
        files={"file": ("../../dangerous_hospital_cam.mp4", file_obj, "video/mp4")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Filename must be sanitized with UUID and not contain path traversal
    assert ".." not in data["filename"]
    assert data["filename"].startswith("upload_")
    assert data["filename"].endswith(".mp4")

