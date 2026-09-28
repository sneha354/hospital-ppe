"""
Unit tests for HIPAA / clinical privacy protection via face/head Gaussian blurring.
"""

import numpy as np
import pytest
from core.alert_manager import AlertManager


def test_face_blur_privacy():
    """Verify that blur_head_privacy blurs the head region while keeping the lower body sharp."""
    # Create test image with distinct checkerboard pattern
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    for y in range(0, 400, 10):
        for x in range(0, 400, 10):
            if (x // 10 + y // 10) % 2 == 0:
                frame[y:y+10, x:x+10] = 255

    person_bbox = (100.0, 50.0, 300.0, 350.0)  # w=200, h=300

    blurred = AlertManager.blur_head_privacy(frame, person_bbox, blur_kernel=(31, 31))

    # Head region (y: 50 to 50 + 300*0.28 = 134, x: 100 to 300) should be modified/blurred
    head_orig = frame[55:125, 110:290]
    head_blurred = blurred[55:125, 110:290]
    assert not np.array_equal(head_orig, head_blurred)

    # Torso/legs region (y: 200 to 340, x: 100 to 300) should NOT be blurred
    torso_orig = frame[200:340, 110:290]
    torso_blurred = blurred[200:340, 110:290]
    assert np.array_equal(torso_orig, torso_blurred)

    # Outside person bbox (y: 0 to 40, x: 0 to 400) should be completely untouched
    outside_orig = frame[0:40, 0:400]
    outside_blurred = blurred[0:40, 0:400]
    assert np.array_equal(outside_orig, outside_blurred)

