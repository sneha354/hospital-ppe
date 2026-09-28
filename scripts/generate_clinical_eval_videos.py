"""
Clinical Evaluation Demonstration Video Generator.
Produces high-definition (1280x720) demonstration videos for hospital PPE compliance evaluation:
1. Compliant Medical Worker: Person wearing full scrubs, blue surgical mask, nitrile gloves, bouffant cap, eye protection.
2. Non-Compliant Person: Civilian in casual clothing, no scrubs, no mask, bare hands (no gloves), no cap.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_hospital_background(w=1280, h=720, area_type="icu"):
    """Creates a clean, realistic clinical room background."""
    frame = np.full((h, w, 3), (232, 236, 235), dtype=np.uint8)  # Hospital wall tint

    # Floor / Wall division (floor starts at 58% height)
    floor_y = int(h * 0.58)
    # Floor gradient
    for y in range(floor_y, h):
        grad_factor = (y - floor_y) / float(h - floor_y)
        floor_color = (
            int(205 - 20 * grad_factor),
            int(215 - 20 * grad_factor),
            int(210 - 20 * grad_factor)
        )
        cv2.line(frame, (0, y), (w, y), floor_color, 1)

    # Baseboard line
    cv2.line(frame, (0, floor_y), (w, floor_y), (140, 150, 155), 4)

    # Hospital floor tiles (perspective lines)
    for tx in range(0, w + 200, 140):
        cv2.line(frame, (tx, floor_y), (int(tx * 1.3 - 150), h), (185, 195, 190), 1)
    for ty in range(floor_y + 35, h, 45):
        cv2.line(frame, (0, ty), (w, ty), (185, 195, 190), 1)

    # Wall features: Medical equipment counter & monitor
    cv2.rectangle(frame, (int(w * 0.72), int(h * 0.35)), (w - 20, floor_y), (210, 218, 220), -1)
    cv2.rectangle(frame, (int(w * 0.72), int(h * 0.35)), (w - 20, floor_y), (160, 170, 175), 2)
    # Vital signs monitor screen
    cv2.rectangle(frame, (int(w * 0.78), int(h * 0.18)), (int(w * 0.95), int(h * 0.35)), (30, 35, 40), -1)
    cv2.rectangle(frame, (int(w * 0.78), int(h * 0.18)), (int(w * 0.95), int(h * 0.35)), (90, 100, 110), 2)
    # Vital signs green/blue waveform lines
    cv2.putText(frame, "HR 74  SpO2 99%", (int(w * 0.79), int(h * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1)
    cv2.putText(frame, "BP 120/78 (92)", (int(w * 0.79), int(h * 0.26)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 210, 240), 1)

    # Area header banner on wall
    if area_type == "icu":
        title = "ICU SUITE 04 - RESTRICTED AIRBORNE & CONTACT ISOLATION"
        code = "ZONE: ICU-01"
    else:
        title = "GENERAL WARD / OPD TRIAGE INTAKE CORRIDOR"
        code = "ZONE: GW-01"

    cv2.putText(frame, title, (50, 45), cv2.FONT_HERSHEY_DUPLEX, 0.62, (100, 110, 125), 1)
    cv2.putText(frame, code, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 140, 155), 1)

    # Hand hygiene dispenser on wall
    cv2.rectangle(frame, (int(w * 0.12), int(h * 0.30)), (int(w * 0.12) + 30, int(h * 0.30) + 60), (240, 245, 250), -1)
    cv2.rectangle(frame, (int(w * 0.12), int(h * 0.30)), (int(w * 0.12) + 30, int(h * 0.30) + 60), (140, 150, 160), 2)
    cv2.putText(frame, "RUB", (int(w * 0.12) + 3, int(h * 0.30) + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (30, 140, 220), 1)

    return frame


def render_human(
    frame: np.ndarray,
    cx: int,
    cy: int,
    t: float,
    is_compliant: bool = True
):
    """
    Renders a realistic human figure with detailed anatomical structure and PPE layers.
    cx, cy: coordinates for the center base (feet position on floor).
    t: elapsed time in seconds for natural movement.
    """
    # Scale person: height ~ 340px, width ~ 130px
    total_h = 340
    top_y = cy - total_h
    body_center_x = cx + int(6 * np.sin(t * 1.5))
    sway_y = int(3 * np.cos(t * 2.0))
    top_y += sway_y

    head_h = 60
    head_w = 48
    head_center_x = body_center_x
    head_center_y = top_y + head_h // 2

    # Skin tone (light peach/beige BGR: 195, 215, 240)
    skin_color = (195, 215, 240)

    # 1. SHOES / FEET (Floor level)
    foot_y = cy + sway_y
    left_foot_x = body_center_x - 22
    right_foot_x = body_center_x + 22

    if is_compliant:
        # Medical Shoe Covers (Blue BGR: 220, 160, 50)
        shoe_color = (220, 160, 50)
        cv2.ellipse(frame, (left_foot_x, foot_y - 4), (18, 9), 0, 0, 360, shoe_color, -1)
        cv2.ellipse(frame, (right_foot_x, foot_y - 4), (18, 9), 0, 0, 360, shoe_color, -1)
        cv2.ellipse(frame, (left_foot_x, foot_y - 4), (18, 9), 0, 0, 360, (180, 120, 30), 1)
        cv2.ellipse(frame, (right_foot_x, foot_y - 4), (18, 9), 0, 0, 360, (180, 120, 30), 1)
    else:
        # Casual Brown/Black Leather Street Shoes
        shoe_color = (40, 45, 55)
        cv2.ellipse(frame, (left_foot_x, foot_y - 4), (16, 8), 0, 0, 360, shoe_color, -1)
        cv2.ellipse(frame, (right_foot_x, foot_y - 4), (16, 8), 0, 0, 360, shoe_color, -1)

    # 2. LEGS / PANTS
    leg_top_y = top_y + int(total_h * 0.58)
    leg_bot_y = foot_y - 8

    if is_compliant:
        # Medical Scrub Pants (Hospital Surgical Blue BGR: 180, 130, 45)
        pants_color = (180, 130, 45)
        cv2.rectangle(frame, (body_center_x - 30, leg_top_y), (body_center_x - 8, leg_bot_y), pants_color, -1)
        cv2.rectangle(frame, (body_center_x + 8, leg_top_y), (body_center_x + 30, leg_bot_y), pants_color, -1)
    else:
        # Casual Denim Blue Jeans (BGR: 110, 70, 45)
        pants_color = (110, 70, 45)
        cv2.rectangle(frame, (body_center_x - 28, leg_top_y), (body_center_x - 6, leg_bot_y), pants_color, -1)
        cv2.rectangle(frame, (body_center_x + 6, leg_top_y), (body_center_x + 28, leg_bot_y), pants_color, -1)

    # 3. TORSO (Scrubs/Gown vs Casual Street Jacket)
    torso_top_y = top_y + head_h
    torso_bot_y = leg_top_y + 8
    torso_w = 76

    if is_compliant:
        # Medical Isolation Gown / Scrubs (Hospital Surgical Cyan-Blue: BGR 185, 135, 40)
        gown_color = (185, 135, 40)
        cv2.rectangle(frame, (body_center_x - torso_w // 2, torso_top_y), (body_center_x + torso_w // 2, torso_bot_y), gown_color, -1)
        # Gown contour & tie fold
        cv2.rectangle(frame, (body_center_x - torso_w // 2, torso_top_y), (body_center_x + torso_w // 2, torso_bot_y), (145, 105, 30), 2)
        cv2.line(frame, (body_center_x, torso_top_y + 10), (body_center_x, torso_bot_y - 10), (160, 115, 35), 2)
        # Stethoscope around neck
        cv2.ellipse(frame, (body_center_x, torso_top_y + 22), (20, 18), 0, 0, 180, (40, 40, 40), 3)
    else:
        # Casual Red/Burgundy Street Jacket (BGR: 50, 45, 185)
        jacket_color = (50, 45, 185)
        cv2.rectangle(frame, (body_center_x - torso_w // 2, torso_top_y), (body_center_x + torso_w // 2, torso_bot_y), jacket_color, -1)
        # Open zipper with gray t-shirt underneath
        cv2.rectangle(frame, (body_center_x - 12, torso_top_y + 5), (body_center_x + 12, torso_bot_y), (180, 180, 185), -1)
        cv2.line(frame, (body_center_x - 12, torso_top_y + 5), (body_center_x - 12, torso_bot_y), (35, 30, 140), 2)
        cv2.line(frame, (body_center_x + 12, torso_top_y + 5), (body_center_x + 12, torso_bot_y), (35, 30, 140), 2)

    # 4. ARMS & HANDS
    arm_l_x = body_center_x - torso_w // 2 - 14
    arm_r_x = body_center_x + torso_w // 2 + 14
    elbow_y = torso_top_y + int((torso_bot_y - torso_top_y) * 0.55)
    hand_y = torso_bot_y - 5 + int(8 * np.sin(t * 2.5))

    if is_compliant:
        # Scrub sleeves
        cv2.line(frame, (body_center_x - torso_w // 2 + 4, torso_top_y + 6), (arm_l_x, elbow_y), gown_color, 14)
        cv2.line(frame, (arm_l_x, elbow_y), (arm_l_x + 4, hand_y), gown_color, 12)
        cv2.line(frame, (body_center_x + torso_w // 2 - 4, torso_top_y + 6), (arm_r_x, elbow_y), gown_color, 14)
        cv2.line(frame, (arm_r_x, elbow_y), (arm_r_x - 4, hand_y), gown_color, 12)

        # Nitrile Medical Gloves (Purple-Blue BGR: 230, 145, 60)
        glove_color = (230, 145, 60)
        cv2.circle(frame, (arm_l_x + 4, hand_y + 6), 11, glove_color, -1)
        cv2.circle(frame, (arm_r_x - 4, hand_y + 6), 11, glove_color, -1)
        cv2.circle(frame, (arm_l_x + 4, hand_y + 6), 11, (190, 110, 40), 1)
        cv2.circle(frame, (arm_r_x - 4, hand_y + 6), 11, (190, 110, 40), 1)
    else:
        # Casual Jacket Sleeves
        cv2.line(frame, (body_center_x - torso_w // 2 + 4, torso_top_y + 6), (arm_l_x, elbow_y), jacket_color, 14)
        cv2.line(frame, (arm_l_x, elbow_y), (arm_l_x + 4, hand_y), jacket_color, 12)
        cv2.line(frame, (body_center_x + torso_w // 2 - 4, torso_top_y + 6), (arm_r_x, elbow_y), jacket_color, 14)
        cv2.line(frame, (arm_r_x, elbow_y), (arm_r_x - 4, hand_y), jacket_color, 12)

        # Bare Hands (Skin tone, NO GLOVES)
        cv2.circle(frame, (arm_l_x + 4, hand_y + 6), 10, skin_color, -1)
        cv2.circle(frame, (arm_r_x - 4, hand_y + 6), 10, skin_color, -1)
        cv2.circle(frame, (arm_l_x + 4, hand_y + 6), 10, (160, 180, 205), 1)
        cv2.circle(frame, (arm_r_x - 4, hand_y + 6), 10, (160, 180, 205), 1)

    # 5. HEAD & FACE
    # Head base ellipse (Skin tone)
    cv2.ellipse(frame, (head_center_x, head_center_y), (head_w // 2, head_h // 2), 0, 0, 360, skin_color, -1)
    cv2.ellipse(frame, (head_center_x, head_center_y), (head_w // 2, head_h // 2), 0, 0, 360, (160, 180, 205), 1)

    # Neck
    cv2.rectangle(frame, (head_center_x - 12, head_center_y + head_h // 2 - 6), (head_center_x + 12, torso_top_y + 4), skin_color, -1)

    if is_compliant:
        # Bouffant Surgical Cap (Teal / Blue: BGR 210, 160, 50)
        cap_color = (210, 160, 50)
        cv2.ellipse(frame, (head_center_x, head_center_y - 12), (head_w // 2 + 5, head_h // 2 - 6), 0, 170, 370, cap_color, -1)
        cv2.ellipse(frame, (head_center_x, head_center_y - 12), (head_w // 2 + 5, head_h // 2 - 6), 0, 170, 370, (170, 120, 30), 2)
        # Cap elastic band across forehead
        cv2.line(frame, (head_center_x - head_w // 2 - 2, head_center_y - 10), (head_center_x + head_w // 2 + 2, head_center_y - 10), (170, 120, 30), 2)

        # Eye Protection / Goggles
        goggle_y = head_center_y - 4
        # Frame
        cv2.rectangle(frame, (head_center_x - 20, goggle_y - 6), (head_center_x + 20, goggle_y + 6), (255, 255, 255), -1)
        cv2.rectangle(frame, (head_center_x - 20, goggle_y - 6), (head_center_x + 20, goggle_y + 6), (120, 130, 140), 2)
        # Clear shield glare line
        cv2.line(frame, (head_center_x - 14, goggle_y + 4), (head_center_x - 4, goggle_y - 4), (220, 245, 255), 2)
        cv2.line(frame, (head_center_x + 4, goggle_y + 4), (head_center_x + 14, goggle_y - 4), (220, 245, 255), 2)

        # Medical Face Mask (Surgical Cyan-Blue: BGR 235, 185, 90)
        mask_top_y = head_center_y + 2
        mask_bot_y = head_center_y + head_h // 2 + 3
        mask_w = head_w - 2
        mask_color = (235, 185, 90)
        cv2.rectangle(frame, (head_center_x - mask_w // 2, mask_top_y), (head_center_x + mask_w // 2, mask_bot_y), mask_color, -1)
        cv2.rectangle(frame, (head_center_x - mask_w // 2, mask_top_y), (head_center_x + mask_w // 2, mask_bot_y), (180, 135, 60), 1)
        # Pleats / folds on mask
        cv2.line(frame, (head_center_x - mask_w // 2 + 4, mask_top_y + 8), (head_center_x + mask_w // 2 - 4, mask_top_y + 8), (210, 160, 75), 1)
        cv2.line(frame, (head_center_x - mask_w // 2 + 4, mask_top_y + 16), (head_center_x + mask_w // 2 - 4, mask_top_y + 16), (210, 160, 75), 1)
        # Ear loops
        cv2.line(frame, (head_center_x - mask_w // 2, mask_top_y + 6), (head_center_x - head_w // 2, head_center_y), (220, 220, 230), 1)
        cv2.line(frame, (head_center_x + mask_w // 2, mask_top_y + 6), (head_center_x + head_w // 2, head_center_y), (220, 220, 230), 1)

    else:
        # Civilian Hair (Dark Brown / Black: BGR 35, 30, 45)
        hair_color = (35, 30, 45)
        cv2.ellipse(frame, (head_center_x, head_center_y - 12), (head_w // 2 + 2, head_h // 2 - 6), 0, 180, 360, hair_color, -1)

        # Eyes & Eyebrows (NO EYE PROTECTION)
        eye_y = head_center_y - 5
        cv2.circle(frame, (head_center_x - 10, eye_y), 3, (30, 30, 30), -1)
        cv2.circle(frame, (head_center_x + 10, eye_y), 3, (30, 30, 30), -1)
        cv2.line(frame, (head_center_x - 14, eye_y - 5), (head_center_x - 6, eye_y - 5), (35, 30, 45), 2)
        cv2.line(frame, (head_center_x + 6, eye_y - 5), (head_center_x + 14, eye_y - 5), (35, 30, 45), 2)

        # Nose & Lips (NO MASK - BARE FACE)
        nose_y = head_center_y + 5
        cv2.line(frame, (head_center_x, nose_y - 3), (head_center_x - 2, nose_y + 4), (160, 175, 200), 2)
        cv2.line(frame, (head_center_x - 2, nose_y + 4), (head_center_x + 3, nose_y + 4), (160, 175, 200), 2)

        mouth_y = head_center_y + 18
        cv2.line(frame, (head_center_x - 8, mouth_y), (head_center_x + 8, mouth_y), (120, 120, 180), 2)


def generate_compliant_video(output_path: str, duration_sec: int = 10, fps: int = 25):
    """Generates demonstration video of a clinical healthcare worker in full PPE."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = draw_hospital_background(w, h, area_type="icu")

        # Person standing in center of ICU/Operating Room, attending to patient
        cx = int(w * 0.45 + 18 * np.sin(t * 0.8))
        cy = int(h * 0.78)

        render_human(frame, cx, cy, t, is_compliant=True)

        # Subtle camera motion / timestamp
        cv2.putText(
            frame,
            f"REC [LIVE] | CLINICAL EVALUATION FEED | {t:.1f}s",
            (w - 420, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 60),
            1
        )
        out.write(frame)

    out.release()
    print(f"Generated compliant video: {output_path} ({total_frames} frames)")


def generate_non_compliant_video(output_path: str, duration_sec: int = 10, fps: int = 25):
    """Generates demonstration video of an unmasked person in casual clothes."""
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / float(fps)
        frame = draw_hospital_background(w, h, area_type="gw")

        # Visitor walking across ward from left to right
        walk_progress = (t / float(duration_sec))
        cx = int(w * 0.22 + walk_progress * (w * 0.56))
        cy = int(h * 0.78)

        render_human(frame, cx, cy, t, is_compliant=False)

        cv2.putText(
            frame,
            f"REC [LIVE] | CLINICAL EVALUATION FEED | {t:.1f}s",
            (w - 420, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 60),
            1
        )
        out.write(frame)

    out.release()
    print(f"Generated non-compliant video: {output_path} ({total_frames} frames)")


if __name__ == "__main__":
    generate_compliant_video(os.path.join(OUTPUT_DIR, "clinical_person_compliant_scrubs_ppe.mp4"))
    generate_non_compliant_video(os.path.join(OUTPUT_DIR, "clinical_person_non_compliant_casual.mp4"))
    print("Clinical evaluation demonstration videos generated successfully.")
