"""
Anatomical PPE-to-Person Association Module.
Maps detected masks, gowns, gloves, eye protection, and shoe covers to individual persons
based on spatial containment and anatomical landmarks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from core.detector import (
    Detection,
    CLASS_PERSON,
    CLASS_FACE_MASK,
    CLASS_MASK_INCORRECT,
    CLASS_NO_MASK,
    CLASS_EYE_PROTECTION,
    CLASS_GOWN,
    CLASS_GLOVES,
    CLASS_HEAD_COVER,
    CLASS_SHOE_COVERS
)


@dataclass
class PersonPPEProfile:
    """Consolidated PPE profile for an individual person."""
    person_det: Detection
    mask_status: Optional[str] = None  # "face_mask", "mask_incorrect", "no_mask", or None
    mask_confidence: float = 0.0
    has_eye_protection: bool = False
    has_gown: bool = False
    has_gloves: bool = False
    has_head_cover: bool = False
    has_shoe_covers: bool = False
    associated_items: List[Detection] = field(default_factory=list)

    @property
    def person_box(self) -> Tuple[float, float, float, float]:
        return self.person_det.bbox

    def get_detected_ppe_set(self) -> set:
        """Returns set of valid, properly worn PPE items."""
        items = set()
        if self.mask_status == CLASS_FACE_MASK:
            items.add("face_mask")
        if self.has_eye_protection:
            items.add("eye_protection")
        if self.has_gown:
            items.add("gown")
        if self.has_gloves:
            items.add("gloves")
        if self.has_head_cover:
            items.add("head_cover")
        if self.has_shoe_covers:
            items.add("shoe_covers")
        return items


class AnatomicalAssociator:
    """
    Associates PPE detections to human bounding boxes using anatomical height ratios:
    - Head (0.00 - 0.32): Face Mask, Eye Protection, Head Cover
    - Torso (0.20 - 0.80): Gown
    - Hands (0.35 - 0.88): Gloves
    - Feet (0.80 - 1.05): Shoe Covers
    """

    def __init__(self, horizontal_margin: float = 0.15, vertical_margin: float = 0.10):
        self.h_margin = horizontal_margin
        self.v_margin = vertical_margin

    def associate(self, detections: List[Detection]) -> List[PersonPPEProfile]:
        """Groups detected PPE items with the corresponding person detection."""
        # 1. Separate person detections from PPE detections
        persons = [d for d in detections if d.class_name == CLASS_PERSON]
        ppe_items = [d for d in detections if d.class_name != CLASS_PERSON]

        # Initialize profile for each detected person
        profiles = [PersonPPEProfile(person_det=p) for p in persons]

        if not profiles or not ppe_items:
            return profiles

        # 2. For each PPE item, find the best matching person based on anatomical score
        for item in ppe_items:
            best_person_idx = -1
            best_score = -1.0

            for idx, prof in enumerate(profiles):
                score = self._compute_anatomical_score(prof.person_det, item)
                if score > best_score and score > 0.30:  # Threshold for valid association
                    best_score = score
                    best_person_idx = idx

            if best_person_idx >= 0:
                self._assign_item_to_profile(profiles[best_person_idx], item)

        return profiles

    def _compute_anatomical_score(self, person: Detection, item: Detection) -> float:
        """
        Computes compatibility score [0.0 - 1.0] between a person bounding box and a PPE item.
        Checks horizontal overlap and whether the item is in the anatomically expected vertical slice.
        """
        px1, py1, px2, py2 = person.bbox
        pw = px2 - px1
        ph = py2 - py1

        if pw <= 0 or ph <= 0:
            return 0.0

        ix1, iy1, ix2, iy2 = item.bbox
        icx, icy = item.center

        # Normalized coordinates of item center relative to person box
        rel_x = (icx - px1) / pw
        rel_y = (icy - py1) / ph

        # Check horizontal bounds with margin (hands/arms can extend outside torso)
        if rel_x < -self.h_margin or rel_x > (1.0 + self.h_margin):
            return 0.0

        # Define expected anatomical vertical ranges [min_y, max_y]
        cname = item.class_name
        if cname in (CLASS_FACE_MASK, CLASS_MASK_INCORRECT, CLASS_NO_MASK, CLASS_EYE_PROTECTION, CLASS_HEAD_COVER):
            expected_range = (-0.05, 0.35)
        elif cname == CLASS_GOWN:
            expected_range = (0.15, 0.85)
        elif cname == CLASS_GLOVES:
            expected_range = (0.30, 0.90)
        elif cname == CLASS_SHOE_COVERS:
            expected_range = (0.75, 1.10)
        else:
            expected_range = (0.0, 1.0)

        min_y, max_y = expected_range
        if rel_y < min_y or rel_y > max_y:
            return 0.0

        # Compute overlap score: 1.0 if centered in range, decaying towards edges
        target_center_y = (min_y + max_y) / 2.0
        range_half_width = (max_y - min_y) / 2.0
        y_dist = abs(rel_y - target_center_y) / max(0.01, range_half_width)
        vertical_score = max(0.0, 1.0 - y_dist * 0.5)

        # Horizontal alignment score
        x_dist = abs(rel_x - 0.5) * 2.0
        horizontal_score = max(0.0, 1.0 - x_dist * 0.3)

        return vertical_score * 0.7 + horizontal_score * 0.3

    def _assign_item_to_profile(self, profile: PersonPPEProfile, item: Detection):
        """Updates the person profile with the detected PPE item."""
        profile.associated_items.append(item)
        cname = item.class_name

        if cname in (CLASS_FACE_MASK, CLASS_MASK_INCORRECT, CLASS_NO_MASK):
            # If multiple mask detections occur, retain the one with highest confidence
            if profile.mask_status is None or item.confidence > profile.mask_confidence:
                profile.mask_status = cname
                profile.mask_confidence = item.confidence

        elif cname == CLASS_EYE_PROTECTION:
            profile.has_eye_protection = True

        elif cname == CLASS_GOWN:
            profile.has_gown = True

        elif cname == CLASS_GLOVES:
            profile.has_gloves = True

        elif cname == CLASS_HEAD_COVER:
            profile.has_head_cover = True

        elif cname == CLASS_SHOE_COVERS:
            profile.has_shoe_covers = True

