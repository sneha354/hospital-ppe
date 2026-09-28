"""
Hospital Zone Management Module.
Handles polygonal Regions of Interest (ROIs), point-in-polygon containment,
zone policy assignment, and visual overlay rendering.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os
import yaml
import cv2
import numpy as np
from core.detector import Detection


@dataclass
class Zone:
    """Represents a designated hospital area with specific PPE compliance requirements."""
    id: str
    name: str
    code: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    color: Tuple[int, int, int]  # RGB
    polygon: List[Tuple[float, float]]  # List of (x, y) normalized [0.0 - 1.0] or pixel coords
    required_ppe: List[str] = field(default_factory=list)
    optional_ppe: List[str] = field(default_factory=list)
    grace_period_seconds: float = 3.0
    alert_severity: str = "WARNING"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "risk_level": self.risk_level,
            "color": list(self.color),
            "polygon": [list(p) for p in self.polygon],
            "required_ppe": list(self.required_ppe),
            "optional_ppe": list(self.optional_ppe),
            "grace_period_seconds": self.grace_period_seconds,
            "alert_severity": self.alert_severity,
            "description": self.description
        }


class ZoneManager:
    """Manages hospital zones, coordinate transformations, and spatial assignment."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/zones.yaml"
        self.zones: Dict[str, Zone] = {}
        self.ppe_definitions: Dict[str, Any] = {}
        self.load_zones(self.config_path)

    def load_zones(self, config_path: str):
        """Loads zones and PPE definitions from YAML configuration file."""
        if not os.path.exists(config_path):
            print(f"[ZoneManager] Config file {config_path} not found. Initializing empty.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.zones.clear()
        raw_zones = data.get("zones", [])
        for z in raw_zones:
            zone = Zone(
                id=z["id"],
                name=z.get("name", z["id"]),
                code=z.get("code", "Z-00"),
                risk_level=z.get("risk_level", "MEDIUM"),
                color=tuple(z.get("color", [59, 130, 246])),
                polygon=[(float(p[0]), float(p[1])) for p in z.get("polygon", [])],
                required_ppe=z.get("required_ppe", []),
                optional_ppe=z.get("optional_ppe", []),
                grace_period_seconds=float(z.get("grace_period_seconds", 3.0)),
                alert_severity=z.get("alert_severity", "WARNING"),
                description=z.get("description", "")
            )
            self.zones[zone.id] = zone

        self.ppe_definitions = data.get("ppe_definitions", {})

    def save_zones(self, config_path: Optional[str] = None):
        """Persists current zones to YAML configuration file."""
        path = config_path or self.config_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "zones": [z.to_dict() for z in self.zones.values()],
            "ppe_definitions": self.ppe_definitions
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def get_all_zones(self) -> List[Zone]:
        return list(self.zones.values())

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        return self.zones.get(zone_id)

    def add_or_update_zone(self, zone: Zone, persist: bool = True):
        self.zones[zone.id] = zone
        if persist:
            self.save_zones()

    def delete_zone(self, zone_id: str) -> bool:
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.save_zones()
            return True
        return False

    def get_pixel_polygon(self, zone: Zone, frame_width: int, frame_height: int) -> np.ndarray:
        """Converts normalized polygon coordinates [0.0 - 1.0] to pixel integers."""
        pts = []
        for x, y in zone.polygon:
            # If coordinates are already in pixel space (> 1.0), use directly
            px = int(x * frame_width) if x <= 1.0 else int(x)
            py = int(y * frame_height) if y <= 1.0 else int(y)
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def find_zone_for_person(self, person_det: Detection, frame_width: int, frame_height: int) -> Optional[Zone]:
        """
        Determines which zone a person is standing in.
        Uses the bottom center of the bounding box (feet position on floor) as ground anchor.
        """
        bx, by = person_det.bottom_center
        # Check against each zone
        for zone in self.zones.values():
            if len(zone.polygon) < 3:
                continue
            poly_pts = self.get_pixel_polygon(zone, frame_width, frame_height)
            # cv2.pointPolygonTest returns >= 0 if inside or on edge
            dist = cv2.pointPolygonTest(poly_pts, (float(bx), float(by)), measureDist=False)
            if dist >= 0:
                return zone

        # Fallback: check center of person
        cx, cy = person_det.center
        for zone in self.zones.values():
            if len(zone.polygon) < 3:
                continue
            poly_pts = self.get_pixel_polygon(zone, frame_width, frame_height)
            if cv2.pointPolygonTest(poly_pts, (float(cx), float(cy)), measureDist=False) >= 0:
                return zone

        return None

    def draw_zones_overlay(self, frame: np.ndarray, alpha: float = 0.20) -> np.ndarray:
        """Renders semi-transparent zone polygons and badges on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        for zone in self.zones.values():
            if len(zone.polygon) < 3:
                continue

            pts = self.get_pixel_polygon(zone, w, h)
            # Convert RGB to BGR for OpenCV
            color_bgr = (int(zone.color[2]), int(zone.color[1]), int(zone.color[0]))

            # Fill polygon on overlay
            cv2.fillPoly(overlay, [pts], color_bgr)
            # Draw boundary line
            cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Draw zone badge/label at centroid
            moments = cv2.moments(pts)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = pts[0][0], pts[0][1]

            label = f"[{zone.code}] {zone.name}"
            req_text = "Req: " + (", ".join(zone.required_ppe) if zone.required_ppe else "None")

            # Draw label box
            font = cv2.FONT_HERSHEY_SIMPLEX
            (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
            (rw, rh), _ = cv2.getTextSize(req_text, font, 0.38, 1)
            max_w = max(lw, rw) + 16
            box_h = lh + rh + 18

            bx1 = max(5, cx - max_w // 2)
            by1 = max(5, cy - box_h // 2)
            bx2 = min(w - 5, bx1 + max_w)
            by2 = min(h - 5, by1 + box_h)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 24, 33), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_bgr, 1)

            cv2.putText(frame, label, (bx1 + 8, by1 + lh + 4), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, req_text, (bx1 + 8, by1 + lh + rh + 10), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Blend overlay with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

