"""
Hospital PPE Compliance Evaluation Engine.
Evaluates tracked personnel against zone-specific PPE policies,
computes compliance states, grace periods, and generates visual overlays.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import cv2
import numpy as np

from core.tracker import TrackedPerson
from core.zone_manager import Zone
from core.detector import CLASS_FACE_MASK, CLASS_MASK_INCORRECT, CLASS_NO_MASK


class ComplianceState(str, Enum):
    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    GRACE_PERIOD = "GRACE_PERIOD"
    UNMONITORED = "UNMONITORED"


@dataclass
class ComplianceResult:
    """Evaluation result for a single person in a specific zone."""
    track_id: int
    zone: Optional[Zone]
    state: ComplianceState
    compliance_score: float  # 0.0 to 1.0
    missing_items: List[str] = field(default_factory=list)
    improper_items: List[str] = field(default_factory=list)
    present_items: List[str] = field(default_factory=list)
    dwell_seconds: float = 0.0
    grace_remaining_seconds: float = 0.0
    should_alert: bool = False
    details: str = ""


import time


class ComplianceEngine:
    """
    Evaluates personnel compliance against zone policies.
    Applies grace periods, determines violation severities, and annotates frames.
    """

    def __init__(self, fps: float = 25.0):
        self.fps = max(1.0, fps)

    def evaluate_person(
        self,
        person: TrackedPerson,
        zone: Optional[Zone],
        current_frame: int = 0,
        current_time: Optional[float] = None
    ) -> ComplianceResult:
        """Evaluates a tracked person's PPE against their assigned zone policy."""
        wall_time = current_time if current_time is not None else time.monotonic()

        # If person is outside any defined zone
        if zone is None:
            return ComplianceResult(
                track_id=person.track_id,
                zone=None,
                state=ComplianceState.UNMONITORED,
                compliance_score=1.0,
                details="Outside designated hospital zones."
            )

        # Check if person changed zones
        if person.current_zone_id != zone.id:
            person.current_zone_id = zone.id
            person.zone_entry_frame = current_frame
            person.zone_entry_time = wall_time

        # Calculate dwell time in current zone
        if current_time is not None:
            # Explicit wall-clock time provided
            dwell_seconds = max(0.0, current_time - person.zone_entry_time)
        elif current_frame > person.zone_entry_frame:
            # Frame-based advance provided
            dwell_seconds = (current_frame - person.zone_entry_frame) / self.fps
        elif hasattr(person, "zone_entry_time") and person.zone_entry_time > 0:
            dwell_seconds = max(0.0, wall_time - person.zone_entry_time)
        else:
            dwell_seconds = 0.0

        # Get smoothed PPE presence using wall-clock window
        ppe_status = person.get_smoothed_ppe_status(current_time=wall_time)

        required = zone.required_ppe
        if not required:
            # Zone requires no PPE (e.g. public corridor)
            return ComplianceResult(
                track_id=person.track_id,
                zone=zone,
                state=ComplianceState.COMPLIANT,
                compliance_score=1.0,
                dwell_seconds=dwell_seconds,
                details="No mandatory PPE for this zone."
            )

        missing = []
        improper = []
        present = []

        for req in required:
            if req == "face_mask":
                if ppe_status.get("face_mask", False):
                    present.append("face_mask")
                elif ppe_status.get("mask_incorrect", False):
                    improper.append("face_mask (chin/exposed)")
                else:
                    missing.append("face_mask")
            else:
                if ppe_status.get(req, False):
                    present.append(req)
                else:
                    missing.append(req)

        # Compute compliance score
        total_req = len(required)
        num_satisfied = len(present)
        score = num_satisfied / float(total_req) if total_req > 0 else 1.0

        # Check grace period
        in_grace = dwell_seconds < zone.grace_period_seconds
        grace_remaining = max(0.0, zone.grace_period_seconds - dwell_seconds)

        # Determine Compliance State
        if not missing and not improper:
            state = ComplianceState.COMPLIANT
            should_alert = False
            details = "Fully compliant with zone requirements."
            person.is_in_violation = False

        elif in_grace:
            state = ComplianceState.GRACE_PERIOD
            should_alert = False
            details = f"Grace period active ({grace_remaining:.1f}s remaining)."
            person.is_in_violation = False

        elif improper and not missing:
            state = ComplianceState.PARTIALLY_COMPLIANT
            should_alert = True
            details = f"Improper PPE: {', '.join(improper)}"
            person.is_in_violation = True

        else:
            state = ComplianceState.NON_COMPLIANT
            should_alert = True
            details = f"Missing: {', '.join(missing)}"
            if improper:
                details += f" | Improper: {', '.join(improper)}"
            person.is_in_violation = True

        return ComplianceResult(
            track_id=person.track_id,
            zone=zone,
            state=state,
            compliance_score=score,
            missing_items=missing,
            improper_items=improper,
            present_items=present,
            dwell_seconds=dwell_seconds,
            grace_remaining_seconds=grace_remaining,
            should_alert=should_alert,
            details=details
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        result: ComplianceResult
    ) -> np.ndarray:
        """Renders bounding box, track ID, status badge, and item checklist on the frame."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        w, h = x2 - x1, y2 - y1

        # Determine color scheme based on compliance state
        if result.state == ComplianceState.COMPLIANT:
            color = (34, 197, 94)      # Green (BGR)
            badge_text = "COMPLIANT"
        elif result.state == ComplianceState.GRACE_PERIOD:
            color = (234, 179, 8)      # Cyan / Light Blue (BGR: 255, 200, 0)
            color = (245, 158, 11)     # Amber/Cyan
            badge_text = f"ENTRY GRACE ({result.grace_remaining_seconds:.1f}s)"
        elif result.state == ComplianceState.PARTIALLY_COMPLIANT:
            color = (0, 165, 255)      # Orange (BGR)
            badge_text = "PARTIALLY COMPLIANT"
        elif result.state == ComplianceState.NON_COMPLIANT:
            color = (50, 50, 239)      # Red (BGR)
            badge_text = "VIOLATION - NON COMPLIANT"
        else:
            color = (156, 163, 175)    # Gray
            badge_text = "UNMONITORED"

        # Draw Person Bounding Box (thick high-contrast)
        box_thickness = 3 if result.state in (ComplianceState.NON_COMPLIANT, ComplianceState.COMPLIANT) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

        # Draw Corner Accents for high-tech surveillance look
        corner_len = min(24, max(12, w // 4), max(12, h // 4))
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 5)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 5)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 5)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 5)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 5)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 5)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 5)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 5)

        # Draw Header Badge above person
        font = cv2.FONT_HERSHEY_SIMPLEX
        zone_code = result.zone.code if result.zone else "NONE"
        header_text = f"ID #{person.track_id:02d} | {zone_code} | {badge_text}"

        (tw, th), _ = cv2.getTextSize(header_text, font, 0.46, 1)
        badge_y1 = max(0, y1 - th - 14)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = max(x2, x1 + tw + 16)

        # Draw filled badge background with border
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 20, 28), -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 2)
        cv2.putText(frame, header_text, (badge_x1 + 8, badge_y2 - 6), font, 0.46, color, 1, cv2.LINE_AA)

        # Draw Item Checklist on the right or below badge if in a monitored zone
        if result.zone and result.zone.required_ppe:
            checklist_y = badge_y1 - 6
            items_to_show = []
            for req in result.zone.required_ppe:
                req_label = req.replace('_', ' ').title()
                if req in result.present_items:
                    items_to_show.append((f"[+] {req_label}", (34, 197, 94)))
                elif any(req in s for s in result.improper_items):
                    items_to_show.append((f"[!] {req_label} (wrong)", (0, 165, 255)))
                else:
                    items_to_show.append((f"[-] {req_label} (MISSING)", (50, 50, 239)))

            # Render compact vertical list
            for text, item_color in reversed(items_to_show):
                (iw, ih), _ = cv2.getTextSize(text, font, 0.38, 1)
                item_y1 = max(0, checklist_y - ih - 6)
                item_y2 = checklist_y
                cv2.rectangle(frame, (badge_x1, item_y1), (badge_x1 + iw + 12, item_y2), (15, 20, 28), -1)
                cv2.rectangle(frame, (badge_x1, item_y1), (badge_x1 + iw + 12, item_y2), item_color, 1)
                cv2.putText(frame, text, (badge_x1 + 6, item_y2 - 3), font, 0.38, item_color, 1, cv2.LINE_AA)
                checklist_y = item_y1 - 3

        return frame

