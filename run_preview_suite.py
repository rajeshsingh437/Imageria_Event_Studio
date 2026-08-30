import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from mediapipe.python.solutions import face_detection as mp_face
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.face_detection as mp_face
    except (ImportError, AttributeError):
        import mediapipe as mp
        mp_face = mp.solutions.face_detection

def apply_master_retouch(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)

    base_l = cv2.bilateralFilter(l_float, d=5, sigmaColor=18.0, sigmaSpace=18.0)
    detail_l = l_float - base_l

    soft_base = (0.75 * base_l) + (0.25 * l_float)
    retouched_l = np.clip(soft_base + detail_l, 0, 255).astype(np.uint8)
    retouched_bgr = cv2.cvtColor(cv2.merge([retouched_l, a, b]), cv2.COLOR_LAB2BGR)

    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    skin_mask = cv2.GaussianBlur(skin_mask, (25, 25), 0).astype(np.float32) / 255.0
    skin_mask = skin_mask[..., np.newaxis]

    final = (skin_mask * retouched_bgr.astype(np.float32)) + ((1.0 - skin_mask) * crop.astype(np.float32))
    return np.clip(final, 0, 255).astype(np.uint8)

def solve_convocation_crop(bgr_img):
    img_h, img_w = bgr_img.shape[:2]
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.35)
    results = face_detector.process(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None, None

    det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
    kp_coords = [(k.x * img_w, k.y * img_h) for k in det.location_data.relative_keypoints]
    (r_eye_x, r_eye_y), (l_eye_x, l_eye_y) = kp_coords[:2]

    bbox = det.location_data.relative_bounding_box
    w_face = bbox.width * img_w
    h_face = bbox.height * img_h

    # 1. Pre-pad canvas with seamless reflection before rotation
    pre_pad_y = int(1.2 * h_face)
    pre_pad_x = int(1.0 * w_face)
    padded_bgr = cv2.copyMakeBorder(
        bgr_img, pre_pad_y, pre_pad_y, pre_pad_x, pre_pad_x,
        borderType=cv2.BORDER_REFLECT_101
    )
    pad_h, pad_w = padded_bgr.shape[:2]

    # 2. Level horizon so shoulders & eyes sit horizontally
    dx = l_eye_x - r_eye_x
    dy = l_eye_y - r_eye_y
    if dx != 0:
        tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        if abs(tilt_deg) > 0.3 and abs(tilt_deg) <= 15.0:
            rot_m = cv2.getRotationMatrix2D((pad_w // 2, pad_h // 2), tilt_deg, 1.0)
            padded_bgr = cv2.warpAffine(padded_bgr, rot_m, (pad_w, pad_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

    # 3. Re-detect on straightened canvas
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

    # 4. Convocation Full-Body Dynamic Anchor Solver
    y1 = int(max(0, y_eye - (1.25 * h_face)))
    y2 = int(min(pad_h, y_eye + (5.35 * h_face)))
    h_crop = y2 - y1
    w_crop = int(0.72 * h_crop)

    x1 = int(max(0, x_center - (w_crop / 2.0)))
    x2 = int(min(pad_w, x1 + w_crop))

    crop_raw = padded_bgr[y1:y2, x1:x2]
    if crop_raw is None or crop_raw.size == 0:
        return None, None

    # Apply Stage 4 Retouching
    crop_retouched = apply_master_retouch(crop_raw)

    # Standardize to 2160x3000 at high quality
    final_portrait = cv2.resize(crop_retouched, (2160, 3000), interpolation=cv2.INTER_LANCZOS4)
    raw_resized = cv2.resize(crop_raw, (2160, 3000), interpolation=cv2.INTER_LANCZOS4)

    return final_portrait, raw_resized

def generate_sample_pdf(hero_img_path, out_pdf_path):
    canvas_w, canvas_h = 2160, 3000
    pages = []

    # Page 1: Cover
    p1 = Image.new('RGB', (canvas_w, canvas_h), color=(11, 29, 58))
    draw1 = ImageDraw.Draw(p1)
    draw1.rectangle([60, 60, canvas_w - 60, canvas_h - 60], outline=(212, 175, 55), width=8)
    draw1.rectangle([90, 90, canvas_w - 90, canvas_h - 90], outline=(212, 175, 55), width=3)
    draw1.ellipse([(canvas_w // 2 - 250, 1150), (canvas_w // 2 + 250, 1650)], outline=(212, 175, 55), width=6)
    pages.append(p1)

    # Page 2: Solo Hero
    p2 = Image.new('RGB', (canvas_w, canvas_h), color=(248, 249, 250))
    draw2 = ImageDraw.Draw(p2)
    if os.path.exists(hero_img_path):
        hero_bgr = cv2.imread(hero_img_path)
        if hero_bgr is not None:
            hero_rgb = cv2.cvtColor(hero_bgr, cv2.COLOR_BGR2RGB)
            hero_pil = Image.fromarray(hero_rgb)
            target_w = 1800
            target_h = int(target_w * (3000 / 2160))
            if target_h > 2400:
                target_h = 2400
                target_w = int(target_h * (2160 / 3000))
            hero_resized = hero_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x_pos = (canvas_w - target_w) // 2
            y_pos = 320
            p2.paste(hero_resized, (x_pos, y_pos))
            draw2.rectangle([x_pos - 6, y_pos - 6, x_pos + target_w + 6, y_pos + target_h + 6], outline=(212, 175, 55), width=6)
    pages.append(p2)

    # Page 3 & 4
    pages.append(Image.new('RGB', (canvas_w, canvas_h), color=(248, 249, 250)))
    pages.append(Image.new('RGB', (canvas_w, canvas_h), color=(11, 29, 58)))

    pages[0].save(out_pdf_path, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])
    print(f"[PDF CREATED] -> {out_pdf_path}")

def find_file(fname):
    for root, _, files in os.walk(r"D:\Imageria\Cminds"):
        for f in files:
            if f.lower() == fname.lower():
                return os.path.join(root, f)
    return None

preview_map = {
    "_RK_3114.JPG": "preview_3114.jpg",
    "_RK_3133.JPG": "preview_3133.jpg",
    "_RK_3136.JPG": "preview_3136.jpg",
    "_RK_3139.JPG": "preview_3139.jpg"
}

first_hero = None
for src_name, out_name in preview_map.items():
    src_path = find_file(src_name)
    if src_path:
        img = cv2.imread(src_path)
        if img is not None:
            retouched, raw_cropped = solve_convocation_crop(img)
            if retouched is not None:
                cv2.imwrite(out_name, retouched, [cv2.IMWRITE_JPEG_QUALITY, 98])
                proof = np.hstack([raw_cropped, retouched])
                proof_name = f"PROOF_{out_name}"
                cv2.imwrite(proof_name, proof, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"[PREVIEW GENERATED] {out_name} (2160x3000 px) & {proof_name}")
                if first_hero is None:
                    first_hero = out_name

if first_hero:
    generate_sample_pdf(first_hero, "Student_001_Sample_Memoir.pdf")

print("\n[ALL PREVIEWS & PROOFS GENERATED SUCCESSFULLY!]")
