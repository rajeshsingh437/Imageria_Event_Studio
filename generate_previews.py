import os
import cv2
import numpy as np

try:
    from mediapipe.python.solutions import face_detection as mp_face
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.face_detection as mp_face
    except (ImportError, AttributeError):
        import mediapipe as mp
        mp_face = mp.solutions.face_detection

def find_photo(base_dir, target_num):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if target_num in f and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, f)
    return None

def crop_convocation_portrait(img_path, out_path):
    if not img_path or not os.path.exists(img_path):
        print(f"[ERROR] Could not find source photo: {img_path}")
        return False

    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return False

    img_h, img_w = bgr_img.shape[:2]
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.35)
    results = face_detector.process(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

    if not results.detections:
        print(f"[WARNING] No face detected in {img_path}")
        return False

    det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
    kp_coords = [(k.x * img_w, k.y * img_h) for k in det.location_data.relative_keypoints]
    (r_eye_x, r_eye_y), (l_eye_x, l_eye_y) = kp_coords[:2]

    bbox = det.location_data.relative_bounding_box
    w_face = bbox.width * img_w
    h_face = bbox.height * img_h

    # 1. Pre-pad canvas to protect head ceiling during rotation
    pre_pad_y = int(1.2 * h_face)
    pre_pad_x = int(1.0 * w_face)
    padded_bgr = cv2.copyMakeBorder(
        bgr_img, pre_pad_y, pre_pad_y, pre_pad_x, pre_pad_x,
        borderType=cv2.BORDER_REFLECT_101
    )

    pad_h, pad_w = padded_bgr.shape[:2]

    # 2. Level shoulders and eyes on horizontal grid line
    dx = l_eye_x - r_eye_x
    dy = l_eye_y - r_eye_y
    if dx != 0:
        tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        if abs(tilt_deg) > 0.3 and abs(tilt_deg) <= 15.0:
            rot_m = cv2.getRotationMatrix2D((pad_w // 2, pad_h // 2), tilt_deg, 1.0)
            padded_bgr = cv2.warpAffine(padded_bgr, rot_m, (pad_w, pad_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

    # 3. Re-detect on straightened canvas for exact upright coordinates
    results_straight = face_detector.process(cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB))
    if results_straight.detections:
        det_s = max(results_straight.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
        kp_s = [(k.x * pad_w, k.y * pad_h) for k in det_s.location_data.relative_keypoints]
        (r_eye_x, r_eye_y), (l_eye_x, l_eye_y) = kp_s[:2]
        bbox_s = det_s.location_data.relative_bounding_box
        w_face = bbox_s.width * pad_w
        h_face = bbox_s.height * pad_h
    else:
        r_eye_x += pre_pad_x
        r_eye_y += pre_pad_y
        l_eye_x += pre_pad_x
        l_eye_y += pre_pad_y

    x_center = (r_eye_x + l_eye_x) / 2.0
    y_eye = (r_eye_y + l_eye_y) / 2.0

    # 4. Locked Eye-Line Framing Math
    # Top: 1.25x face height above eye line (pushed 5-7cm higher, neat headroom)
    y1 = int(max(0, y_eye - (1.25 * h_face)))

    # Bottom: 5.35x face height below eye line (room below hands and folder, floor excluded)
    y2 = int(min(pad_h, y_eye + (5.35 * h_face)))

    h_crop = y2 - y1

    # Width: 0.72x height (covers full shoulders + arms + 2-3 cm side margins)
    w_crop = int(0.72 * h_crop)

    x1 = int(max(0, x_center - (w_crop / 2.0)))
    x2 = int(min(pad_w, x1 + w_crop))

    crop = padded_bgr[y1:y2, x1:x2]

    # Standardized High-Res 300+ DPI Print Output (2160 x 3000 px)
    final_crop = cv2.resize(crop, (2160, 3000), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(out_path, final_crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"[PREVIEW READY] Successfully Generated -> {out_path} ({final_crop.shape}x{final_crop.shape[0]} px)")
    return True

base_folder = "./Cminds"
targets = [
    ("3114", "preview_1_RK_3114.jpg"),
    ("3133", "preview_2_RK_3133.jpg"),
    ("3136", "preview_3_RK_3136.jpg"),
    ("3173", "preview_4_RK_3173.jpg")
]

generated_files = []
for num, out_name in targets:
    src_path = find_photo(base_folder, num)
    if src_path:
        success = crop_convocation_portrait(src_path, out_name)
        if success:
            generated_files.append(out_name)
    else:
        print(f"[ERROR] Could not find photo with number {num} in {base_folder}")

print(f"\n[DONE] Generated {len(generated_files)} preview images.")
