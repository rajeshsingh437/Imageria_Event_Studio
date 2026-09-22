"""
IMAGERIA EVENT STUDIO — STEP 3: STRICT BOUNDARY PORTRAIT CROPPING
File: step3_anatomy_crop_test_gemini.py
Role: Crops standard 7:10 (2160x3000 px) stage portraits strictly within real image bounds.
      Eliminates edge reflections, eliminates phantom limbs/elbows, levels horizon gently,
      and preserves full natural framing.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# MediaPipe Robust Imports
try:
    from mediapipe.python.solutions import face_detection as mp_face
    from mediapipe.python.solutions import pose as mp_pose
except (ImportError, AttributeError):
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    mp_pose = mp.solutions.pose

# ================= USER CONFIGURATION =================
DEFAULT_INPUT_DIR = r"E:\HTS\Step2_Sorted_Event_Cohorts\01_Performers_and_Students"
TARGET_WIDTH = 2160
TARGET_HEIGHT = 3000
ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT   # 0.72 (7:10)
DEFAULT_FOLDER_LIMIT = 3                       # Process first 3 performer folders for quick QA
# ======================================================


def calculate_dynamic_crop(img_bgr: np.ndarray, face_detector, pose_detector) -> tuple[np.ndarray | None, str]:
    """
    Crops strictly within the real image boundaries:
    - Never uses reflective padding (prevents mirror/ghost artifacts at the bottom).
    - Uses full vertical height of original image down to the floor/cut-off.
    - Preserves eye line and balances center based on shoulders.
    """
    img_h, img_w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Face Detection
    face_results = face_detector.process(rgb)
    if not face_results.detections:
        return None, "No face detected"

    det = max(
        face_results.detections,
        key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height
    )
    kp_coords = [(k.x * img_w, k.y * img_h) for k in det.location_data.relative_keypoints]
    (r_eye_x, r_eye_y), (l_eye_x, l_eye_y) = kp_coords[:2]

    bbox = det.location_data.relative_bounding_box
    w_face = bbox.width * img_w
    h_face = bbox.height * img_h

    # 2. Subtle Horizon Leveling (within bounds)
    dx = l_eye_x - r_eye_x
    dy = l_eye_y - r_eye_y
    tilt_deg = 0.0
    working_img = img_bgr

    if dx != 0:
        tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        # Only rotate if noticeable tilt, and keep within valid pixels
        if 0.5 < abs(tilt_deg) <= 8.0:
            rot_m = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), tilt_deg, 1.0)
            working_img = cv2.warpAffine(
                img_bgr, rot_m, (img_w, img_h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
            )
            # Recompute eyes after slight rotation
            r_eye_pt = rot_m.dot(np.array([r_eye_x, r_eye_y, 1.0]))
            l_eye_pt = rot_m.dot(np.array([l_eye_x, l_eye_y, 1.0]))
            r_eye_x, r_eye_y = r_eye_pt[0], r_eye_pt[1]
            l_eye_x, l_eye_y = l_eye_pt[0], l_eye_pt[1]

    x_center = (r_eye_x + l_eye_x) / 2.0
    y_eye = (r_eye_y + l_eye_y) / 2.0

    # 3. Center on Shoulders / Body
    straight_rgb = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
    pose_results = pose_detector.process(straight_rgb)
    if pose_results.pose_landmarks:
        plms = pose_results.pose_landmarks.landmark
        l_sh = plms[11]
        r_sh = plms[12]
        if l_sh.visibility > 0.2 and r_sh.visibility > 0.2:
            body_center_x = ((l_sh.x + r_sh.x) / 2.0) * img_w
            x_center = (0.75 * body_center_x) + (0.25 * x_center)

    # 4. Strict Coordinate Calculation (No artificial expansion)
    # y2 is locked to the real bottom of the captured photo
    y2 = img_h

    # Headroom: Keep a clean 1.3x face height above eye level (or clamp to top 0)
    desired_y1 = y_eye - (1.3 * h_face)
    y1 = int(max(0, desired_y1))

    h_crop = y2 - y1

    # Solve width from 7:10 target aspect ratio
    w_crop = int(round(ASPECT_RATIO * h_crop))

    # If w_crop exceeds image width, fit height to available width
    if w_crop > img_w:
        w_crop = img_w
        h_crop = int(round(w_crop / ASPECT_RATIO))
        # Keep y2 at bottom and re-calculate y1
        y1 = max(0, y2 - h_crop)
        h_crop = y2 - y1

    # Horizontal center placement
    x1 = int(round(x_center - (w_crop / 2.0)))
    x2 = x1 + w_crop

    # Clamp horizontal coordinates strictly inside real image
    if x1 < 0:
        x2 = min(img_w, x2 - x1)
        x1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 = max(0, x1 - shift)
        x2 = img_w

    # Ensure integer dimensions strictly match aspect ratio
    actual_w = x2 - x1
    actual_h = int(round(actual_w / ASPECT_RATIO))
    y1 = max(0, y2 - actual_h)

    # 5. Extract strictly real pixels
    crop_region = working_img[y1:y2, x1:x2]
    if crop_region.shape[0] < 50 or crop_region.shape[1] < 50:
        return None, "Crop region invalid"

    # Resample cleanly to 2160x3000 px print standard
    final_portrait = cv2.resize(
        crop_region, (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LANCZOS4
    )

    info = f"Tilt: {tilt_deg:+.1f} deg | Crop: {x2-x1}x{y2-y1} -> 2160x3000"
    return final_portrait, info


def run_step3_cropping(input_base_dir: str = DEFAULT_INPUT_DIR, folder_limit: int | None = DEFAULT_FOLDER_LIMIT):
    if not os.path.exists(input_base_dir):
        print(f"[ERROR] Input directory not found: {input_base_dir}")
        return

    print("=" * 80)
    print(" IMAGERIA EVENT STUDIO — STEP 3: STRICT BOUNDARY PORTRAIT CROPPING")
    print(f" Cohorts Folder : {input_base_dir}")
    print(f" Target Standard: {TARGET_WIDTH} x {TARGET_HEIGHT} px @ 300 DPI (7:10 Portrait)")
    print(f" Test Scope     : {'First ' + str(folder_limit) + ' student folders' if folder_limit else 'All Student Folders'}")
    print("=" * 80)

    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.30)
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.20)

    student_dirs = []
    for entry in sorted(os.listdir(input_base_dir)):
        p = os.path.join(input_base_dir, entry)
        if os.path.isdir(p):
            student_dirs.append(p)

    total_folders = len(student_dirs)
    print(f"[*] Found {total_folders} student/performer folders.")

    if folder_limit and folder_limit > 0:
        process_dirs = student_dirs[:folder_limit]
        print(f"[*] Quick-Test Mode: Processing first {len(process_dirs)} folders.\n")
    else:
        process_dirs = student_dirs
        print(f"[*] Full Batch Mode: Processing all {len(process_dirs)} folders.\n")

    total_cropped = 0

    for f_idx, s_dir in enumerate(process_dirs):
        s_name = os.path.basename(s_dir)
        print(f"\n--- [{f_idx+1}/{len(process_dirs)}] Processing Cohort: {s_name} ---")

        files = os.listdir(s_dir)
        target_files = []
        for f in files:
            if f.endswith("-crop.jpg") or f.startswith("GROUP_"):
                continue
            if f.endswith("-ed.jpg"):
                target_files.append(f)
            elif not any(f.startswith(Path(tf).stem.replace("-ed", "")) for tf in target_files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    target_files.append(f)

        for tf in target_files:
            src_path = os.path.join(s_dir, tf)
            stem = Path(tf).stem.replace("-ed", "").replace("_ORIGINAL", "")
            out_crop_name = f"{stem}-crop.jpg"
            out_crop_path = os.path.join(s_dir, out_crop_name)

            im = cv2.imread(src_path)
            if im is None:
                continue

            portrait, info = calculate_dynamic_crop(im, face_detector, pose_detector)
            if portrait is not None:
                cv2.imwrite(out_crop_path, portrait, [cv2.IMWRITE_JPEG_QUALITY, 98])
                total_cropped += 1
                print(f"  [+] Cropped: {tf} -> {out_crop_name} ({info})")
            else:
                print(f"  [-] Skipped: {tf} ({info})")

    face_detector.close()
    pose_detector.close()

    print("\n" + "=" * 80)
    print(f" STEP 3 COMPLETE: {total_cropped} clean portraits generated.")
    print("=" * 80)


if __name__ == "__main__":
    target_dir = DEFAULT_INPUT_DIR
    limit = DEFAULT_FOLDER_LIMIT

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_dir = sys.argv[1]
    if "--all" in sys.argv:
        limit = None
    elif "--limit" in sys.argv:
        try:
            lim_idx = sys.argv.index("--limit") + 1
            limit = int(sys.argv[lim_idx])
        except (IndexError, ValueError):
            limit = DEFAULT_FOLDER_LIMIT

    run_step3_cropping(target_dir, folder_limit=limit)