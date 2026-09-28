"""
Unit tests for PersonTracker: IoU tracking, ID persistence, and temporal buffering.
"""

import pytest
from core.tracker import PersonTracker
from core.associator import PersonPPEProfile
from core.detector import Detection, CLASS_PERSON, CLASS_FACE_MASK


def test_tracker_id_persistence():
    tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=10)

    # Frame 1: Person at (100, 100, 200, 300)
    det1 = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    prof1 = PersonPPEProfile(person_det=det1, mask_status=CLASS_FACE_MASK)
    tracks_f1 = tracker.update([prof1])

    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id

    # Frame 2: Person moves slightly to (105, 102, 205, 302)
    det2 = Detection(bbox=(105, 102, 205, 302), class_name=CLASS_PERSON)
    prof2 = PersonPPEProfile(person_det=det2, mask_status=CLASS_FACE_MASK)
    tracks_f2 = tracker.update([prof2])

    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t1_id  # Track ID must be preserved!


def test_tracker_prunes_lost_tracks():
    tracker = PersonTracker(max_lost_frames=3)

    det = Detection(bbox=(100, 100, 200, 300), class_name=CLASS_PERSON)
    prof = PersonPPEProfile(person_det=det)
    tracker.update([prof])

    # 4 empty frames
    for _ in range(4):
        tracker.update([])

    # Track should have been pruned
    assert len(tracker.tracks) == 0

