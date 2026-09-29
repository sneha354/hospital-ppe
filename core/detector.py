"""
Detection Module for Hospital PPE and Face Mask Monitoring.
Supports SOTA OpenCV YuNet Deep Learning face/mask detection, Heuristic OpenCV,
ONNX YOLO, and Synthetic Simulation backends.
Includes dual-resolution inference scaling for high-FPS live camera streams.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os
import urllib.request
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Standardized PPE Classes
CLASS_PERSON = "person"
CLASS_FACE_MASK = "face_mask"
CLASS_MASK_INCORRECT = "mask_incorrect"
CLASS_NO_MASK = "no_mask"
CLASS_EYE_PROTECTION = "eye_protection"
CLASS_GOWN = "gown"
CLASS_GLOVES = "gloves"
CLASS_HEAD_COVER = "head_cover"
CLASS_SHOE_COVERS = "shoe_covers"

ALL_PPE_CLASSES = [
    CLASS_FACE_MASK,
    CLASS_MASK_INCORRECT,
    CLASS_NO_MASK,
    CLASS_EYE_PROTECTION,
    CLASS_GOWN,
    CLASS_GLOVES,
    CLASS_HEAD_COVER,
    CLASS_SHOE_COVERS,
]

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_YUNET_PATH = "data/models/face_detection_yunet_2023mar.onnx"


@dataclass
class Detection:
    """Represents a single detected object in an image/frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    class_name: str
    confidence: float = 1.0
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Anchor point representing feet on ground."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


class BaseDetector(ABC):
    """Abstract interface for PPE and person detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs detection on an RGB/BGR frame."""
        pass


class YuNetFaceDetector:
    """
    State-of-the-Art Deep Learning Face Detector using OpenCV's built-in cv2.FaceDetectorYN.
    Runs in 2-5ms on CPU, detects faces at extreme angles and distances with 5 facial landmarks.
    """

    def __init__(self, model_path: str = DEFAULT_YUNET_PATH, conf_threshold: float = 0.6):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.detector: Optional[cv2.FaceDetectorYN] = None
        self._input_size = (320, 320)
        self._ensure_model_available()
        self._init_detector()

    def _ensure_model_available(self):
        """Downloads YuNet model file (~336 KB) if not already cached."""
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        try:
            print(f"[YuNetFaceDetector] Downloading lightweight deep learning model from {YUNET_MODEL_URL}...")
            urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
            print(f"[YuNetFaceDetector] Model successfully cached to {self.model_path}")
        except Exception as e:
            print(f"[YuNetFaceDetector] Notice: Could not download YuNet model ({e}). Using Haar fallback.")

    def _init_detector(self):
        if not os.path.exists(self.model_path):
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=self._input_size,
                score_threshold=self.conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
        except Exception as e:
            print(f"[YuNetFaceDetector] Warning: cv2.FaceDetectorYN creation failed: {e}")
            self.detector = None

    def is_ready(self) -> bool:
        return self.detector is not None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float, np.ndarray]]:
        """
        Detects faces in frame.
        Returns: List of (x1, y1, x2, y2, confidence, landmarks_5x2)
        """
        if self.detector is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self._input_size:
            self._input_size = (w, h)
            self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        results = []
        for face in faces:
            fx, fy, fw, fh = face[0:4]
            conf = float(face[14])
            landmarks = face[4:14].reshape((5, 2))  # Right eye, left eye, nose, right mouth, left mouth
            results.append((float(fx), float(fy), float(fx + fw), float(fy + fh), conf, landmarks))

        return results


class HeuristicOpenCVDetector(BaseDetector):
    """
    Robust production detector combining:
    1. OpenCV YuNet Deep Learning Face & Landmark Detection (SOTA high-speed face/mask)
    2. OpenCV HOG + UpperBody person detection
    3. Adaptive HSV color & texture segmentation for gowns, gloves, and shoe covers
    4. Dual-resolution downsampling for high-FPS live camera feeds
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        inference_size: Tuple[int, int] = (640, 360)
    ):
        self.confidence_threshold = confidence_threshold
        self.inference_size = inference_size

        # Deep Learning Face Detector (OpenCV 5+ SOTA)
        self.yunet = YuNetFaceDetector(conf_threshold=confidence_threshold)

        # HOG Person Detector (legacy OpenCV fallback)
        self.hog = None
        if hasattr(cv2, "HOGDescriptor"):
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None

        # Fallback Haar Cascades (legacy OpenCV fallback)
        self.face_cascade = None
        self.upperbody_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_dir = cv2.data.haarcascades
                face_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
                upperbody_path = os.path.join(cascade_dir, "haarcascade_upperbody.xml")
                if os.path.exists(face_path):
                    self.face_cascade = cv2.CascadeClassifier(face_path)
                if os.path.exists(upperbody_path):
                    self.upperbody_cascade = cv2.CascadeClassifier(upperbody_path)
            except Exception:
                pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]

        # Use downscaled frame for high-FPS inference if frame is HD/4K
        use_scaling = (orig_w > self.inference_size[0] or orig_h > self.inference_size[1])
        if use_scaling:
            inf_w, inf_h = self.inference_size
            inf_frame = cv2.resize(frame, (inf_w, inf_h))
            scale_x = orig_w / float(inf_w)
            scale_y = orig_h / float(inf_h)
        else:
            inf_frame = frame
            scale_x, scale_y = 1.0, 1.0

        h, w = inf_frame.shape[:2]
        detections: List[Detection] = []
        gray = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY)

        person_boxes = []
        if self.hog is not None:
            boxes, weights = self.hog.detectMultiScale(
                inf_frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.08
            )
            for (x, y, bw, bh), conf in zip(boxes, weights):
                if conf >= 0.12:
                    x1, y1 = max(0, float(x)), max(0, float(y))
                    x2, y2 = min(w, float(x + bw)), min(h, float(y + bh))
                    person_boxes.append((x1, y1, x2, y2, float(conf)))

        # Upper body cascade for seated/partially visible staff
        if self.upperbody_cascade is not None:
            upper_boxes = self.upperbody_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
            )
            for (ux, uy, uw, uh) in upper_boxes:
                px1 = max(0, float(ux - uw * 0.1))
                py1 = max(0, float(uy))
                px2 = min(w, float(ux + uw * 1.1))
                py2 = min(h, float(uy + uh * 2.2))

                overlap = False
                for bx1, by1, bx2, by2, _ in person_boxes:
                    if (px1 < bx2 and px2 > bx1 and py1 < by2 and py2 > by1):
                        overlap = True
                        break
                if not overlap:
                    person_boxes.append((px1, py1, px2, py2, 0.65))

        # 2. Deep Learning Face Detection via YuNet
        yunet_faces = []
        if self.yunet.is_ready():
            yunet_faces = self.yunet.detect_faces(inf_frame)

        # 2b. Contour & Silhouette Person Segmentation Fallback
        # Essential for hospital surveillance when faces are angled, masked, or in demo streams
        if not person_boxes and not yunet_faces:
            hsv = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            person_mask = ((sat > 18) | (val < 185)).astype(np.uint8) * 255
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 23))
            closed = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh > 75 and bw > 30 and 1.1 <= bh / float(bw) <= 4.2 and bh < h * 0.95:
                    person_boxes.append((float(bx), float(by), float(bx + bw), float(by + bh), 0.88))

        # If person boxes are found, add them and evaluate PPE
        for x1, y1, x2, y2, conf in person_boxes:
            # Scale back to original frame coordinates
            orig_pbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            detections.append(Detection(
                bbox=orig_pbox,
                class_name=CLASS_PERSON,
                confidence=min(1.0, conf)
            ))

            # Analyze PPE within person ROI
            person_roi = inf_frame[int(y1):int(y2), int(x1):int(x2)]
            if person_roi.size > 0:
                ppe_items = self._analyze_person_ppe(person_roi, x1, y1, scale_x, scale_y)
                detections.extend(ppe_items)

        # 3. Associate YuNet detected faces to persons or add standalone persons
        for fx1, fy1, fx2, fy2, fconf, landmarks in yunet_faces:
            # Check if face is already inside a person detection
            inside_person = any(
                d.class_name == CLASS_PERSON and
                d.x1 <= fx1 * scale_x and fx2 * scale_x <= d.x2 and
                d.y1 <= fy1 * scale_y and fy2 * scale_y <= d.y2
                for d in detections
            )

            fw = fx2 - fx1
            fh = fy2 - fy1

            if not inside_person:
                # Synthesize full person box from face
                px1 = max(0, float(fx1 - fw * 0.8))
                py1 = max(0, float(fy1 - fh * 0.3))
                px2 = min(w, float(fx2 + fw * 0.8))
                py2 = min(h, float(fy1 + fh * 5.0))
                orig_pbox = (px1 * scale_x, py1 * scale_y, px2 * scale_x, py2 * scale_y)
                detections.append(Detection(
                    bbox=orig_pbox,
                    class_name=CLASS_PERSON,
                    confidence=0.85
                ))

            # Classify face mask with landmark awareness
            face_roi = inf_frame[int(fy1):int(fy2), int(fx1):int(fx2)]
            if face_roi.size > 0:
                mask_det = self._classify_face_mask(face_roi, fx1 * scale_x, fy1 * scale_y, fw * scale_x, fh * scale_y)
                if mask_det:
                    detections.append(mask_det)

                eye_det = self._classify_eye_protection(face_roi, fx1 * scale_x, fy1 * scale_y, fw * scale_x, fh * scale_y)
                if eye_det:
                    detections.append(eye_det)

        return detections

    def _analyze_person_ppe(
        self,
        person_roi: np.ndarray,
        offset_x: float,
        offset_y: float,
        scale_x: float,
        scale_y: float
    ) -> List[Detection]:
        ph, pw = person_roi.shape[:2]
        ppe_detections: List[Detection] = []

        # --- A. HEAD & FACE REGION (0 - 28% of height) ---
        head_roi = person_roi[0:int(ph * 0.28), :]
        if head_roi.size > 0:
            head_h, head_w = head_roi.shape[:2]
            head_gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

            faces = []
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(head_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

            if len(faces) > 0:
                for (fx, fy, fw, fh) in faces:
                    face_crop = head_roi[fy:fy+fh, fx:fx+fw]
                    abs_fx = (offset_x + fx) * scale_x
                    abs_fy = (offset_y + fy) * scale_y
                    mask_det = self._classify_face_mask(face_crop, abs_fx, abs_fy, fw * scale_x, fh * scale_y)
                    if mask_det:
                        ppe_detections.append(mask_det)
                    eye_det = self._classify_eye_protection(face_crop, abs_fx, abs_fy, fw * scale_x, fh * scale_y)
                    if eye_det:
                        ppe_detections.append(eye_det)
            else:
                # Estimate face in head region
                efx = int(head_w * 0.2)
                efy = int(head_h * 0.15)
                efw = int(head_w * 0.6)
                efh = int(head_h * 0.8)
                face_crop = head_roi[efy:efy+efh, efx:efx+efw]
                if face_crop.size > 0:
                    mask_det = self._classify_face_mask(
                        face_crop,
                        (offset_x + efx) * scale_x,
                        (offset_y + efy) * scale_y,
                        efw * scale_x,
                        efh * scale_y
                    )
                    if mask_det:
                        ppe_detections.append(mask_det)
                    eye_det = self._classify_eye_protection(
                        face_crop,
                        (offset_x + efx) * scale_x,
                        (offset_y + efy) * scale_y,
                        efw * scale_x,
                        efh * scale_y
                    )
                    if eye_det:
                        ppe_detections.append(eye_det)

            # Head cover / bouffant cap check
            cap_roi = head_roi[0:int(head_h * 0.4), :]
            if cap_roi.size > 0:
                cap_det = self._classify_head_cover(
                    cap_roi,
                    offset_x * scale_x,
                    offset_y * scale_y,
                    head_w * scale_x,
                    int(head_h * 0.4) * scale_y
                )
                if cap_det:
                    ppe_detections.append(cap_det)

        # --- B. TORSO REGION: GOWN (22% - 78% of height) ---
        torso_y1 = int(ph * 0.22)
        torso_y2 = int(ph * 0.78)
        torso_roi = person_roi[torso_y1:torso_y2, int(pw * 0.15):int(pw * 0.85)]
        if torso_roi.size > 0:
            gown_det = self._classify_medical_gown(
                torso_roi,
                (offset_x + pw * 0.15) * scale_x,
                (offset_y + torso_y1) * scale_y,
                pw * 0.7 * scale_x,
                (torso_y2 - torso_y1) * scale_y
            )
            if gown_det:
                ppe_detections.append(gown_det)

        # --- C. HANDS REGION: GLOVES (45% - 85% of height) ---
        gloves_dets = self._classify_gloves(person_roi, offset_x, offset_y, scale_x, scale_y)
        ppe_detections.extend(gloves_dets)

        # --- D. FEET REGION: SHOE COVERS (85% - 100% of height) ---
        feet_y1 = int(ph * 0.85)
        feet_roi = person_roi[feet_y1:ph, :]
        if feet_roi.size > 0:
            shoe_det = self._classify_shoe_covers(
                feet_roi,
                offset_x * scale_x,
                (offset_y + feet_y1) * scale_y,
                pw * scale_x,
                (ph - feet_y1) * scale_y
            )
            if shoe_det:
                ppe_detections.append(shoe_det)

        return ppe_detections

    def _classify_face_mask(self, face_crop: np.ndarray, abs_x: float, abs_y: float, w: float, h: float) -> Optional[Detection]:
        if face_crop.size == 0 or w < 15 or h < 15:
            return None

        # Lower half of face (mouth/chin area)
        lower_face = face_crop[int(face_crop.shape[0] * 0.45):, :]
        if lower_face.size == 0:
            return None

        hsv_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        total_pixels = lower_face.shape[0] * lower_face.shape[1]

        # 1. Skin tone
        skin_mask = cv2.inRange(hsv_lower, np.array([0, 25, 60]), np.array([25, 180, 255]))
        skin_ratio = np.count_nonzero(skin_mask) / float(total_pixels)

        # 2. Medical Blue / Cyan Mask
        blue_mask = cv2.inRange(hsv_lower, np.array([80, 50, 50]), np.array([135, 255, 255]))
        blue_ratio = np.count_nonzero(blue_mask) / float(total_pixels)

        # 3. Medical Green Mask
        green_mask = cv2.inRange(hsv_lower, np.array([35, 50, 50]), np.array([80, 255, 255]))
        green_ratio = np.count_nonzero(green_mask) / float(total_pixels)

        medical_mask_ratio = blue_ratio + green_ratio
        bbox = (abs_x, abs_y + h * 0.4, abs_x + w, abs_y + h)

        if medical_mask_ratio > 0.18 and skin_ratio < 0.28:
            return Detection(bbox=bbox, class_name=CLASS_FACE_MASK, confidence=min(0.98, 0.70 + medical_mask_ratio * 0.5))
        elif medical_mask_ratio > 0.12 and skin_ratio > 0.25:
            return Detection(bbox=bbox, class_name=CLASS_MASK_INCORRECT, confidence=0.82)
        elif skin_ratio > 0.20 or medical_mask_ratio < 0.10:
            return Detection(bbox=bbox, class_name=CLASS_NO_MASK, confidence=min(0.96, 0.65 + skin_ratio * 0.5))
        else:
            cname = CLASS_FACE_MASK if medical_mask_ratio > skin_ratio else CLASS_NO_MASK
            return Detection(bbox=bbox, class_name=cname, confidence=0.65)

    def _classify_eye_protection(self, face_crop: np.ndarray, abs_x: float, abs_y: float, w: float, h: float) -> Optional[Detection]:
        eye_band = face_crop[int(face_crop.shape[0] * 0.15):int(face_crop.shape[0] * 0.5), :]
        if eye_band.size == 0 or eye_band.shape[1] < 10:
            return None

        # Sample central 60% of the eye band to avoid surrounding background wall pixels
        w_eye = eye_band.shape[1]
        center_band = eye_band[:, int(w_eye * 0.20):int(w_eye * 0.80)]
        if center_band.size == 0:
            return None

        gray_eyes = cv2.cvtColor(center_band, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.count_nonzero(gray_eyes > 225) / float(gray_eyes.size)
        hsv = cv2.cvtColor(center_band, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 180, 255]))
        skin_ratio = np.count_nonzero(skin) / float(center_band.shape[0] * center_band.shape[1])

        # Eye protection (goggles/visor) features high-reflection bright shield and low skin ratio
        if bright_pixels > 0.08 and skin_ratio < 0.18:
            return Detection(
                bbox=(abs_x, abs_y + h * 0.1, abs_x + w, abs_y + h * 0.55),
                class_name=CLASS_EYE_PROTECTION,
                confidence=0.88
            )
        return None

    def _classify_head_cover(self, cap_roi: np.ndarray, abs_x: float, abs_y: float, w: float, h: float) -> Optional[Detection]:
        hsv = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2HSV)
        total = cap_roi.shape[0] * cap_roi.shape[1]

        # Bouffant surgical cap: teal, medical blue, or scrub green
        blue_teal = cv2.inRange(hsv, np.array([75, 45, 45]), np.array([135, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([75, 255, 255]))

        cap_ratio = (np.count_nonzero(blue_teal) + np.count_nonzero(green)) / float(total)
        if cap_ratio > 0.20:
            return Detection(
                bbox=(abs_x, abs_y, abs_x + w, abs_y + h),
                class_name=CLASS_HEAD_COVER,
                confidence=min(0.94, 0.65 + cap_ratio * 0.5)
            )
        return None

    def _classify_medical_gown(self, torso_roi: np.ndarray, abs_x: float, abs_y: float, w: float, h: float) -> Optional[Detection]:
        hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
        total = torso_roi.shape[0] * torso_roi.shape[1]

        # Clinical scrubs / isolation gowns: surgical blue, isolation yellow, scrub green
        blue_mask = cv2.inRange(hsv, np.array([80, 45, 40]), np.array([135, 255, 240]))
        yellow_mask = cv2.inRange(hsv, np.array([18, 55, 70]), np.array([35, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([36, 45, 40]), np.array([80, 255, 240]))

        gown_pixels = (
            np.count_nonzero(blue_mask) +
            np.count_nonzero(yellow_mask) +
            np.count_nonzero(green_mask)
        )
        gown_ratio = gown_pixels / float(total)

        if gown_ratio > 0.28:
            return Detection(
                bbox=(abs_x, abs_y, abs_x + w, abs_y + h),
                class_name=CLASS_GOWN,
                confidence=min(0.96, 0.60 + gown_ratio * 0.4)
            )
        return None

    def _classify_gloves(
        self,
        person_roi: np.ndarray,
        offset_x: float,
        offset_y: float,
        scale_x: float,
        scale_y: float
    ) -> List[Detection]:
        ph, pw = person_roi.shape[:2]
        dets: List[Detection] = []

        hand_zones = [
            ("left_hand", 0, int(pw * 0.24), int(ph * 0.45), int(ph * 0.78)),
            ("right_hand", int(pw * 0.76), pw, int(ph * 0.45), int(ph * 0.78)),
        ]

        for side, x1, x2, y1, y2 in hand_zones:
            crop = person_roi[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            total = crop.shape[0] * crop.shape[1]

            nitrile_blue = cv2.inRange(hsv, np.array([85, 70, 70]), np.array([135, 255, 255]))
            latex_yellow = cv2.inRange(hsv, np.array([20, 70, 90]), np.array([35, 255, 255]))
            skin_mask = cv2.inRange(hsv, np.array([0, 25, 60]), np.array([25, 180, 255]))

            glove_ratio = (np.count_nonzero(nitrile_blue) + np.count_nonzero(latex_yellow)) / float(total)
            skin_ratio = np.count_nonzero(skin_mask) / float(total)

            if glove_ratio > 0.12 and skin_ratio < 0.25:
                dets.append(Detection(
                    bbox=(
                        (offset_x + x1) * scale_x,
                        (offset_y + y1) * scale_y,
                        (offset_x + x2) * scale_x,
                        (offset_y + y2) * scale_y
                    ),
                    class_name=CLASS_GLOVES,
                    confidence=min(0.92, 0.60 + glove_ratio * 0.6)
                ))
        return dets

    def _classify_shoe_covers(self, feet_roi: np.ndarray, abs_x: float, abs_y: float, w: float, h: float) -> Optional[Detection]:
        hsv = cv2.cvtColor(feet_roi, cv2.COLOR_BGR2HSV)
        total = feet_roi.shape[0] * feet_roi.shape[1]

        blue_cover = cv2.inRange(hsv, np.array([85, 60, 70]), np.array([135, 255, 255]))
        ratio = np.count_nonzero(blue_cover) / float(total)
        if ratio > 0.25:
            return Detection(
                bbox=(abs_x, abs_y, abs_x + w, abs_y + h),
                class_name=CLASS_SHOE_COVERS,
                confidence=min(0.88, 0.5 + ratio * 0.8)
            )
        return None


class SyntheticSimulatorDetector(BaseDetector):
    """Detector companion for the synthetic hospital simulation stream."""

    def __init__(self):
        self._simulated_detections: List[Detection] = []

    def set_current_simulated_detections(self, detections: List[Detection]):
        self._simulated_detections = detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return list(self._simulated_detections)


class ONNXYoloDetector(BaseDetector):
    """Loads YOLOv8/YOLO11 ONNX model for PPE detection using OpenCV DNN."""

    def __init__(self, model_path: str, conf_thresh: float = 0.5, nms_thresh: float = 0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.net = None

        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"[ONNXYoloDetector] Warning: Failed to load ONNX model {model_path}: {e}")
                self.net = None

        self.class_names = [
            CLASS_PERSON,
            CLASS_FACE_MASK,
            CLASS_MASK_INCORRECT,
            CLASS_NO_MASK,
            CLASS_EYE_PROTECTION,
            CLASS_GOWN,
            CLASS_GLOVES,
            CLASS_HEAD_COVER,
            CLASS_SHOE_COVERS
        ]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.net is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        output = np.squeeze(outputs)
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T

        boxes, confidences, class_ids = [], [], []
        x_factor, y_factor = w / 640.0, h / 640.0

        for row in output:
            classes_scores = row[4:]
            max_score = np.max(classes_scores)
            if max_score >= self.conf_thresh:
                class_id = np.argmax(classes_scores)
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = int((cx - 0.5 * bw) * x_factor)
                y1 = int((cy - 0.5 * bh) * y_factor)
                bw_px = int(bw * x_factor)
                bh_px = int(bh * y_factor)

                boxes.append([x1, y1, bw_px, bh_px])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                cid = class_ids[i]
                cname = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                detections.append(Detection(
                    bbox=(float(bx), float(by), float(bx + bw), float(by + bh)),
                    class_name=cname,
                    confidence=confidences[i]
                ))

        return detections


class YOLOv8Detector(BaseDetector):
    """
    Ultralytics YOLOv8 Deep Learning Detector for Hospital Surveillance.
    Detects humans with high precision across real CCTV, webcams, and clinical video footage.
    Extracts anatomical regions for rigorous hospital PPE compliance classification.
    """

    def __init__(
        self,
        model_path: str = "data/models/yolov8n.pt",
        conf_threshold: float = 0.35,
        inference_size: Tuple[int, int] = (640, 360)
    ):
        self.conf_threshold = conf_threshold
        self.inference_size = inference_size
        self.model_path = model_path
        self.model = None
        self._fallback = HeuristicOpenCVDetector(conf_threshold, inference_size)

        if HAS_ULTRALYTICS:
            try:
                load_src = model_path if os.path.exists(model_path) else "data/models/yolov8n.pt"
                if not os.path.exists(load_src):
                    load_src = "yolov8n.pt"
                self.model = YOLO(load_src)
                print(f"[YOLOv8Detector] Loaded YOLOv8 model from {load_src}")
            except Exception as e:
                print(f"[YOLOv8Detector] Notice: Could not initialize YOLOv8 ({e}). Using OpenCV fallback.")
                self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            return self._fallback.detect(frame)

        orig_h, orig_w = frame.shape[:2]
        detections: List[Detection] = []

        try:
            # Person class in COCO dataset is class 0
            results = self.model(frame, verbose=False, conf=self.conf_threshold, classes=[0])
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                    x1 = max(0.0, min(float(orig_w), x1))
                    y1 = max(0.0, min(float(orig_h), y1))
                    x2 = max(0.0, min(float(orig_w), x2))
                    y2 = max(0.0, min(float(orig_h), y2))

                    if (x2 - x1) < 20 or (y2 - y1) < 40:
                        continue

                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        class_name=CLASS_PERSON,
                        confidence=conf
                    ))

                    person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if person_roi.size > 0:
                        ppe_items = self._fallback._analyze_person_ppe(
                            person_roi, x1, y1, 1.0, 1.0
                        )
                        detections.extend(ppe_items)

            # If YOLO returned 0 persons, also check fallback (YuNet / silhouette) to ensure coverage
            if not any(d.class_name == CLASS_PERSON for d in detections):
                return self._fallback.detect(frame)

        except Exception as ex:
            return self._fallback.detect(frame)

        return detections


class DetectorFactory:
    """Factory to instantiate detectors based on system configuration."""

    @staticmethod
    def create_detector(
        backend: str = "auto",
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        inference_size: Tuple[int, int] = (640, 360)
    ) -> BaseDetector:
        backend = backend.lower()
        if backend == "simulation":
            return SyntheticSimulatorDetector()
        elif HAS_ULTRALYTICS and (backend in ["yolo", "yolov8", "auto"]):
            target_model = model_path if (model_path and "yolo" in model_path.lower()) else "data/models/yolov8n.pt"
            return YOLOv8Detector(
                model_path=target_model,
                conf_threshold=conf_threshold,
                inference_size=inference_size
            )
        elif backend == "onnx" or (backend == "auto" and model_path and os.path.exists(model_path) and model_path.lower().endswith(".onnx")):
            return ONNXYoloDetector(model_path, conf_thresh=conf_threshold)
        else:
            return HeuristicOpenCVDetector(
                confidence_threshold=conf_threshold,
                inference_size=inference_size
            )
