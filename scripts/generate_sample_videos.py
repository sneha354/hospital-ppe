"""
Script to generate realistic hospital demonstration video files for all departments.
Generates MP4 video files with real-time frame rates and diverse clinical scenarios.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_or_surgery_video(output_path: str, duration_sec: int = 6, fps: int = 25):
    """Generates Operating Room video with surgical team in sterile barrier gear."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = np.full((h, w, 3), (215, 225, 220), dtype=np.uint8)  # Hospital green/teal tint

        # Floor tiles
        for x in range(0, w, 60):
            cv2.line(frame, (x, 0), (x, h), (195, 205, 200), 1)
        for y in range(0, h, 60):
            cv2.line(frame, (0, y), (w, y), (195, 205, 200), 1)

        # OR Table & Surgical Lamps
        cv2.rectangle(frame, (int(w * 0.35), int(h * 0.4)), (int(w * 0.65), int(h * 0.7)), (180, 195, 190), -1)
        cv2.circle(frame, (int(w * 0.5), int(h * 0.25)), 65, (230, 240, 245), -1)
        cv2.putText(frame, "OR SUITE 01 - STERILE FIELD", (int(w * 0.38), int(h * 0.55)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (140, 150, 145), 1)

        # Surgeon 1 (Lead Surgeon - Fully Compliant: Cap, Goggles, Mask, Sterile Gown, Gloves, Shoe Covers)
        s1_x = int(w * 0.3 + 15 * np.sin(t * 1.2))
        s1_y = int(h * 0.35 + 8 * np.cos(t * 0.8))
        _draw_staff(frame, s1_x, s1_y, cap=True, goggles=True, mask=True, gown=True, gloves=True, shoes=True, scrub_color=(170, 120, 30))

        # Scrub Nurse (Compliant: Cap, Mask, Gown, Gloves, Shoe Covers)
        s2_x = int(w * 0.7 - 20 * np.cos(t * 1.0))
        s2_y = int(h * 0.4 + 10 * np.sin(t * 1.5))
        _draw_staff(frame, s2_x, s2_y, cap=True, goggles=False, mask=True, gown=True, gloves=True, shoes=True, scrub_color=(170, 120, 30))

        # Anesthesiologist (At head of table: Cap, Mask, Scrubs/Gown)
        s3_x = int(w * 0.5)
        s3_y = int(h * 0.28 + 5 * np.sin(t * 0.5))
        _draw_staff(frame, s3_x, s3_y, cap=True, goggles=True, mask=True, gown=True, gloves=True, shoes=False, scrub_color=(160, 130, 40))

        out.write(frame)
    out.release()
    print(f"Created: {output_path}")


def create_icu_isolation_video(output_path: str, duration_sec: int = 6, fps: int = 25):
    """Generates ICU isolation room video with nurse in N95/gown and unmasked visitor entering."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = np.full((h, w, 3), (230, 230, 235), dtype=np.uint8)

        # Room layout
        cv2.line(frame, (int(w * 0.45), 0), (int(w * 0.45), h), (150, 150, 160), 4)
        cv2.putText(frame, "ANTEROOM / AIRLOCK", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 160), 2)
        cv2.putText(frame, "NEGATIVE PRESSURE ICU BED", (int(w * 0.5), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 160), 2)

        # ICU Bed
        cv2.rectangle(frame, (int(w * 0.65), int(h * 0.4)), (int(w * 0.9), int(h * 0.75)), (200, 210, 215), -1)

        # ICU Nurse (Compliant: N95 Mask, Face Shield, Yellow Isolation Gown, Nitrile Gloves)
        n_x = int(w * 0.55 + 15 * np.sin(t * 1.5))
        n_y = int(h * 0.45 + 10 * np.cos(t * 1.2))
        _draw_staff(frame, n_x, n_y, cap=False, goggles=True, mask=True, gown=True, gloves=True, shoes=False, scrub_color=(40, 190, 230))  # Yellow Gown

        # Visitor walking into ICU (Non-compliant: No mask, casual clothes)
        # Walks from anteroom into patient room
        v_x = int(100 + t * 90)
        v_y = int(h * 0.5 + 5 * np.sin(t * 2.0))
        _draw_staff(frame, v_x, v_y, cap=False, goggles=False, mask=False, gown=False, gloves=False, shoes=False, scrub_color=(80, 60, 170))

        out.write(frame)
    out.release()
    print(f"Created: {output_path}")


def create_triage_corridor_video(output_path: str, duration_sec: int = 6, fps: int = 25):
    """Generates Triage corridor video with staff and walking visitors."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = np.full((h, w, 3), (240, 235, 230), dtype=np.uint8)

        # Corridor lines
        cv2.line(frame, (0, int(h * 0.3)), (w, int(h * 0.3)), (180, 180, 180), 2)
        cv2.putText(frame, "EMERGENCY INTAKE & TRIAGE CORRIDOR", (80, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (130, 130, 140), 2)

        # Triage Nurse at desk (Compliant: Mask, White Coat)
        _draw_staff(frame, int(w * 0.2), int(h * 0.38), cap=False, goggles=False, mask=True, gown=True, gloves=False, shoes=False, scrub_color=(200, 200, 200))

        # Person 1 (Visitor with Mask - Compliant)
        p1_x = int(100 + t * 80)
        p1_y = int(h * 0.45)
        _draw_staff(frame, p1_x, p1_y, cap=False, goggles=False, mask=True, gown=False, gloves=False, shoes=False, scrub_color=(120, 80, 60))

        # Person 2 (Visitor without Mask - Non-Compliant)
        p2_x = int(w - 100 - t * 75)
        p2_y = int(h * 0.52)
        _draw_staff(frame, p2_x, p2_y, cap=False, goggles=False, mask=False, gown=False, gloves=False, shoes=False, scrub_color=(60, 110, 150))

        out.write(frame)
    out.release()
    print(f"Created: {output_path}")


def create_cssd_cleanroom_video(output_path: str, duration_sec: int = 6, fps: int = 25):
    """Generates Central Sterile Supply / Cleanroom video."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = np.full((h, w, 3), (245, 245, 240), dtype=np.uint8)

        cv2.putText(frame, "CSSD STERILE COMPOUNDING CLEANROOM", (80, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (140, 150, 160), 2)
        cv2.rectangle(frame, (int(w * 0.2), int(h * 0.45)), (int(w * 0.8), int(h * 0.75)), (210, 220, 225), -1)

        # Tech 1 (Compliant: Cap, Mask, Gown, Gloves, Shoe Covers)
        c1_x = int(w * 0.4 + 20 * np.sin(t * 1.1))
        c1_y = int(h * 0.42)
        _draw_staff(frame, c1_x, c1_y, cap=True, goggles=False, mask=True, gown=True, gloves=True, shoes=True, scrub_color=(180, 140, 50))

        # Tech 2 (Improper Mask: Mask on chin)
        c2_x = int(w * 0.65 - 15 * np.cos(t * 1.3))
        c2_y = int(h * 0.44)
        _draw_staff(frame, c2_x, c2_y, cap=True, goggles=False, mask="incorrect", gown=True, gloves=True, shoes=True, scrub_color=(180, 140, 50))

        out.write(frame)
    out.release()
    print(f"Created: {output_path}")


def _draw_staff(frame, x, y, cap=False, goggles=False, mask=True, gown=True, gloves=False, shoes=False, scrub_color=(150, 120, 40)):
    """Renders a clinical staff member / person with realistic anatomical PPE."""
    aw, ah = 70, 160

    # Head
    hcx, hcy = x + aw // 2, y + int(ah * 0.12)
    hr = int(aw * 0.26)
    cv2.circle(frame, (hcx, hcy), hr, (190, 210, 235), -1)  # Skin tone

    # Torso / Gown
    torso_y1 = y + int(ah * 0.24)
    torso_y2 = y + int(ah * 0.76)
    cv2.rectangle(frame, (x + int(aw * 0.15), torso_y1), (x + int(aw * 0.85), torso_y2), scrub_color, -1)

    # Legs
    cv2.line(frame, (x + int(aw * 0.35), torso_y2), (x + int(aw * 0.35), y + ah), (60, 60, 60), 5)
    cv2.line(frame, (x + int(aw * 0.65), torso_y2), (x + int(aw * 0.65), y + ah), (60, 60, 60), 5)

    # Arms
    cv2.line(frame, (x + int(aw * 0.15), torso_y1 + 8), (x - 2, torso_y1 + int(ah * 0.35)), scrub_color, 5)
    cv2.line(frame, (x + int(aw * 0.85), torso_y1 + 8), (x + aw + 2, torso_y1 + int(ah * 0.35)), scrub_color, 5)

    # Cap
    if cap:
        cv2.ellipse(frame, (hcx, hcy - 8), (hr + 2, hr - 2), 0, 180, 360, (200, 150, 50), -1)

    # Goggles / Face Shield
    if goggles:
        cv2.rectangle(frame, (hcx - hr + 2, hcy - 4), (hcx + hr - 2, hcy + 4), (255, 255, 255), 2)

    # Mask
    if mask is True:
        cv2.rectangle(frame, (hcx - hr + 2, hcy + 1), (hcx + hr - 2, hcy + hr + 2), (230, 180, 80), -1)  # Blue Surgical
    elif mask == "incorrect":
        cv2.rectangle(frame, (hcx - hr + 4, hcy + hr - 2), (hcx + hr - 4, hcy + hr + 8), (230, 180, 80), -1)  # Chin-strap

    # Gloves
    if gloves:
        cv2.circle(frame, (x - 2, torso_y1 + int(ah * 0.35)), 6, (220, 140, 60), -1)
        cv2.circle(frame, (x + aw + 2, torso_y1 + int(ah * 0.35)), 6, (220, 140, 60), -1)

    # Shoe Covers
    if shoes:
        cv2.ellipse(frame, (x + int(aw * 0.35), y + ah), (10, 5), 0, 0, 360, (220, 150, 50), -1)
        cv2.ellipse(frame, (x + int(aw * 0.65), y + ah), (10, 5), 0, 0, 360, (220, 150, 50), -1)


if __name__ == "__main__":
    create_or_surgery_video(os.path.join(OUTPUT_DIR, "or_surgery_demo.mp4"))
    create_icu_isolation_video(os.path.join(OUTPUT_DIR, "icu_isolation_demo.mp4"))
    create_triage_corridor_video(os.path.join(OUTPUT_DIR, "triage_corridor_demo.mp4"))
    create_cssd_cleanroom_video(os.path.join(OUTPUT_DIR, "cssd_cleanroom_demo.mp4"))
    print("All hospital scenario demonstration videos generated successfully!")

