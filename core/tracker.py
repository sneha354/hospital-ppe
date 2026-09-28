"""
Multi-Person Tracking Module with Temporal Consistency.
Maintains persistent person identities, trajectory smoothing, dwell time,
and temporal window buffering for robust PPE compliance evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import numpy as np
from core.detector import Detection
from core.associator import PersonPPEProfile


import time


@dataclass
class TrackedPerson:
    """Represents an actively tracked person across frames."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # Current smoothed (x1, y1, x2, y2)
    last_detection: Detection
    current_profile: Optional[PersonPPEProfile] = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    lost_frames: int = 0
    current_zone_id: Optional[str] = None
    zone_entry_frame: int = 0
    total_dwell_frames: int = 0

    # Wall-clock timing (immune to variable inference FPS)
    first_seen_time: float = field(default_factory=time.monotonic)
    last_seen_time: float = field(default_factory=time.monotonic)
    zone_entry_time: float = field(default_factory=time.monotonic)
    history_window_seconds: float = 1.5

    # Temporal history buffer for PPE status smoothing
    history_window: int = 20
    mask_history: deque = field(default_factory=lambda: deque(maxlen=20))
    eye_prot_history: deque = field(default_factory=lambda: deque(maxlen=20))
    gown_history: deque = field(default_factory=lambda: deque(maxlen=20))
    gloves_history: deque = field(default_factory=lambda: deque(maxlen=20))
    cap_history: deque = field(default_factory=lambda: deque(maxlen=20))
    shoe_history: deque = field(default_factory=lambda: deque(maxlen=20))
    ppe_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Incident management
    last_alert_frame: int = -9999
    is_in_violation: bool = False

    def update_profile(self, profile: PersonPPEProfile, frame_idx: int, current_time: Optional[float] = None):
        if current_time is None:
            current_time = time.monotonic()

        self.current_profile = profile
        self.last_detection = profile.person_det
        self.last_seen_frame = frame_idx
        self.last_seen_time = current_time
        self.lost_frames = 0
        self.total_dwell_frames += 1

        # Smooth bbox with moving average
        alpha = 0.7  # Weight of new detection
        curr_box = profile.person_det.bbox
        self.bbox = (
            alpha * curr_box[0] + (1 - alpha) * self.bbox[0],
            alpha * curr_box[1] + (1 - alpha) * self.bbox[1],
            alpha * curr_box[2] + (1 - alpha) * self.bbox[2],
            alpha * curr_box[3] + (1 - alpha) * self.bbox[3],
        )

        # Append to legacy frame-based history buffers
        self.mask_history.append(profile.mask_status)
        self.eye_prot_history.append(profile.has_eye_protection)
        self.gown_history.append(profile.has_gown)
        self.gloves_history.append(profile.has_gloves)
        self.cap_history.append(profile.has_head_cover)
        self.shoe_history.append(profile.has_shoe_covers)

        # Timestamped status for wall-clock temporal smoothing
        status_entry = {
            "mask_status": profile.mask_status,
            "has_eye_protection": profile.has_eye_protection,
            "has_gown": profile.has_gown,
            "has_gloves": profile.has_gloves,
            "has_head_cover": profile.has_head_cover,
            "has_shoe_covers": profile.has_shoe_covers,
        }
        self.ppe_history.append((current_time, status_entry))

    def get_smoothed_ppe_status(
        self,
        threshold_ratio: float = 0.50,
        window_seconds: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes temporally smoothed PPE presence using wall-clock sliding window.
        Immune to variable inference FPS.
        """
        if current_time is None:
            current_time = time.monotonic()
        win_sec = window_seconds if window_seconds is not None else self.history_window_seconds

        # Filter recent entries within window_seconds
        recent_statuses = [
            s for ts, s in self.ppe_history
            if (current_time - ts) <= win_sec
        ]

        if not recent_statuses:
            # Fallback to legacy buffers if ppe_history has no recent items
            if not self.mask_history:
                return {
                    "face_mask": False,
                    "mask_incorrect": False,
                    "no_mask": True,
                    "eye_protection": False,
                    "gown": False,
                    "gloves": False,
                    "head_cover": False,
                    "shoe_covers": False,
                }
            n = len(self.mask_history)
            mask_counts = {"face_mask": 0, "mask_incorrect": 0, "no_mask": 0, None: 0}
            for s in self.mask_history:
                mask_counts[s] = mask_counts.get(s, 0) + 1

            has_mask = (mask_counts.get("face_mask", 0) / n) >= threshold_ratio
            mask_inc = (mask_counts.get("mask_incorrect", 0) / n) >= threshold_ratio
            no_mask = not has_mask and not mask_inc

            return {
                "face_mask": has_mask,
                "mask_incorrect": mask_inc,
                "no_mask": no_mask,
                "eye_protection": (sum(self.eye_prot_history) / len(self.eye_prot_history)) >= threshold_ratio if self.eye_prot_history else False,
                "gown": (sum(self.gown_history) / len(self.gown_history)) >= threshold_ratio if self.gown_history else False,
                "gloves": (sum(self.gloves_history) / len(self.gloves_history)) >= threshold_ratio if self.gloves_history else False,
                "head_cover": (sum(self.cap_history) / len(self.cap_history)) >= threshold_ratio if self.cap_history else False,
                "shoe_covers": (sum(self.shoe_history) / len(self.shoe_history)) >= threshold_ratio if self.shoe_history else False,
            }

        n = len(recent_statuses)
        mask_counts = {"face_mask": 0, "mask_incorrect": 0, "no_mask": 0, None: 0}
        eye_count = 0
        gown_count = 0
        gloves_count = 0
        cap_count = 0
        shoe_count = 0

        for st in recent_statuses:
            m = st.get("mask_status")
            mask_counts[m] = mask_counts.get(m, 0) + 1
            if st.get("has_eye_protection"):
                eye_count += 1
            if st.get("has_gown"):
                gown_count += 1
            if st.get("has_gloves"):
                gloves_count += 1
            if st.get("has_head_cover"):
                cap_count += 1
            if st.get("has_shoe_covers"):
                shoe_count += 1

        has_mask = (mask_counts.get("face_mask", 0) / n) >= threshold_ratio
        mask_inc = (mask_counts.get("mask_incorrect", 0) / n) >= threshold_ratio
        no_mask = not has_mask and not mask_inc

        return {
            "face_mask": has_mask,
            "mask_incorrect": mask_inc,
            "no_mask": no_mask,
            "eye_protection": (eye_count / n) >= threshold_ratio,
            "gown": (gown_count / n) >= threshold_ratio,
            "gloves": (gloves_count / n) >= threshold_ratio,
            "head_cover": (cap_count / n) >= threshold_ratio,
            "shoe_covers": (shoe_count / n) >= threshold_ratio,
        }


class PersonTracker:
    """
    IoU-based Multi-Object Tracker specialized for hospital personnel monitoring.
    Handles track birth, death, ID assignment, and temporal consistency.
    """

    def __init__(self, iou_threshold: float = 0.30, max_lost_frames: int = 30, history_window: int = 20):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.history_window = history_window
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.current_frame_idx = 0

    def update(self, profiles: List[PersonPPEProfile], current_time: Optional[float] = None) -> List[TrackedPerson]:
        """Updates tracks with current frame's person profiles."""
        if current_time is None:
            current_time = time.monotonic()

        self.current_frame_idx += 1
        active_tracks = list(self.tracks.values())

        if not profiles:
            # Mark all tracks as lost
            for track in active_tracks:
                track.lost_frames += 1
            self._prune_lost_tracks()
            return [t for t in self.tracks.values() if t.lost_frames == 0]

        if not active_tracks:
            # First detections: initialize all as new tracks
            for p in profiles:
                self._create_new_track(p, current_time)
            return list(self.tracks.values())

        # Compute IoU matrix between existing tracks and incoming detections
        iou_matrix = np.zeros((len(active_tracks), len(profiles)), dtype=np.float32)
        for i, track in enumerate(active_tracks):
            for j, p in enumerate(profiles):
                iou_matrix[i, j] = self._compute_iou(track.bbox, p.person_det.bbox)

        # Match tracks to detections greedily
        matched_tracks = set()
        matched_dets = set()

        while True:
            max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
            if max_val < self.iou_threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track = active_tracks[i]
            profile = profiles[j]

            track.update_profile(profile, self.current_frame_idx, current_time)
            profile.person_det.track_id = track.track_id

            matched_tracks.add(track.track_id)
            matched_dets.add(j)

            # Invalidate row and column
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Create new tracks for unmatched detections
        for j, p in enumerate(profiles):
            if j not in matched_dets:
                self._create_new_track(p, current_time)

        # Increment lost frames for unmatched tracks
        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.lost_frames += 1

        self._prune_lost_tracks()
        return [t for t in self.tracks.values() if t.lost_frames == 0]

    def _create_new_track(self, profile: PersonPPEProfile, current_time: Optional[float] = None) -> TrackedPerson:
        if current_time is None:
            current_time = time.monotonic()

        tid = self.next_track_id
        self.next_track_id += 1
        profile.person_det.track_id = tid

        track = TrackedPerson(
            track_id=tid,
            bbox=profile.person_det.bbox,
            last_detection=profile.person_det,
            current_profile=profile,
            first_seen_frame=self.current_frame_idx,
            last_seen_frame=self.current_frame_idx,
            first_seen_time=current_time,
            last_seen_time=current_time,
            zone_entry_time=current_time,
            lost_frames=0,
            history_window=self.history_window
        )
        track.update_profile(profile, self.current_frame_idx, current_time)
        self.tracks[tid] = track
        return track

    def _prune_lost_tracks(self):
        """Removes tracks that have been lost for more than max_lost_frames."""
        to_delete = [tid for tid, t in self.tracks.items() if t.lost_frames > self.max_lost_frames]
        for tid in to_delete:
            del self.tracks[tid]

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        if unionArea <= 0:
            return 0.0
        return interArea / unionArea
