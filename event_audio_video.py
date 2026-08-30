import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
from datetime import datetime

# --- Mute AI Backend Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
try:
    import absl.logging
    absl.logging.set_verbosity('error')
except ImportError:
    pass

import mediapipe as mp

# --- Bulletproof MediaPipe Import ---
try:
    import mediapipe.python.solutions as mp_solutions
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except (ImportError, ModuleNotFoundError, AttributeError):
    mp_solutions = mp.solutions
    mp_face_mesh = mp.solutions.face_mesh
# ------------------------------------

# ================= USER CONFIGURATION =================
INPUT_PATH = r"./raw_photos"
OUTPUT_PATH = r"" 
WATERMARK_TEXT = "" 
TIME_GAP_SECONDS = 25 
BLUR_THRESHOLD = 80.0 
EAR_THRESHOLD = 0.20
# ======================================================

IMAGE_EXTS = ('.cr2', '.nef', '.arw', '.dng', '.cr3', '.jpg', '.jpeg')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mts')

def get_capture_time(file_path):
    return datetime.fromtimestamp(os.path.getmtime(file_path))


class AdaptiveRetouchAndQualityEngine:
    def __init__(self, ear_threshold=EAR_THRESHOLD, blur_threshold=BLUR_THRESHOLD, watermark_text=WATERMARK_TEXT):
        self.ear_threshold = ear_threshold
        self.blur_threshold = blur_threshold
        self.watermark_text = watermark_text
        self.mp_face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def luminance_denoise(self, img):
        """Fast edge-preserving chroma & luminance cleanup."""
        return cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=15)

    def adaptive_white_balance(self, img):
        """Downsampled proxy calculation for massive speedup on 24MP+ images."""
        h, w = img.shape[:2]
        scale = 1600.0 / max(h, w) if max(h, w) > 1600 else 1.0
        
        if scale < 1.0:
            small_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small_img = img

        lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)
        base_l = cv2.bilateralFilter(l_float, d=5, sigmaColor=18.0, sigmaSpace=18.0)
        detail_l = l_float - base_l
        soft_base = (0.75 * base_l) + (0.25 * l_float)
        retouched_l = np.clip(soft_base + detail_l, 0, 255).astype(np.uint8)
        retouched_bgr_small = cv2.cvtColor(cv2.merge([retouched_l, a, b]), cv2.COLOR_LAB2BGR)
        
        ycrcb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0).astype(np.float32) / 255.0
        skin_mask = skin_mask[..., np.newaxis]
        
        retouched_small = (skin_mask * retouched_bgr_small.astype(np.float32)) + ((1.0 - skin_mask) * small_img.astype(np.float32))
        retouched_small = np.clip(retouched_small, 0, 255).astype(np.uint8)

        if scale < 1.0:
            return cv2.resize(retouched_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return retouched_small

    def apply_vignette(self, img, cx, cy):
        h, w = img.shape[:2]
        max_dist = np.sqrt(w**2 + h**2) / 1.2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = 1.0 - np.clip((dist_from_center / max_dist)**2.5, 0, 0.45)
        return cv2.convertScaleAbs(img * mask[:, :, np.newaxis])

    def sharpen_features(self, img, face_landmarks, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)
        
        def draw_poly(indices):
            pts = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in indices])
            cv2.fillPoly(mask, [pts], 255)
            
        draw_poly([33, 160, 158, 133, 153, 144])  # Left eye
        draw_poly([362, 385, 387, 263, 373, 380])  # Right eye
        draw_poly([70, 63, 105, 66, 107])  # Left brow
        draw_poly([336, 296, 334, 293, 300])  # Right brow
        draw_poly([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95])  # Lips
        
        mask_blur = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
        mask_blur = np.expand_dims(mask_blur, axis=2)
        
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        return (img * (1 - mask_blur) + sharpened * mask_blur).astype(np.uint8)

    def calculate_ear(self, landmarks, eye_indices, w, h):
        pts = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in eye_indices]
        v1, v2, h1 = np.linalg.norm(pts[1] - pts[5]), np.linalg.norm(pts[2] - pts[4]), np.linalg.norm(pts[0] - pts[3])
        return 0.30 if h1 == 0 else (v1 + v2) / (2.0 * h1)

    def evaluate_quality_gate(self, img, face_landmarks):
        h, w = img.shape[:2]
        landmarks = face_landmarks.landmark
        left_ear = self.calculate_ear(landmarks, [33, 160, 158, 133, 144, 153], w, h)
        right_ear = self.calculate_ear(landmarks, [362, 385, 387, 263, 373, 380], w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.ear_threshold:
            return False, f"Blink Detected (EAR = {avg_ear:.2f})"
            
        fx, fy = int(landmarks[1].x * w), int(landmarks[10].y * h)
        fw, fh = int(abs(landmarks[454].x - landmarks[234].x) * w), int(abs(landmarks[152].y - landmarks[10].y) * h)
        x1, y1 = max(0, fx - fw // 2), max(0, fy)
        x2, y2 = min(w, x1 + fw), min(h, y1 + fh)
        
        face_roi = img[y1:y2, x1:x2]
        if face_roi.size > 0:
            laplacian_var = cv2.Laplacian(cv2.GaussianBlur(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), (3, 3), 0), cv2.CV_64F).var()
            if laplacian_var < self.blur_threshold:
                return False, f"Blur Detected (Var = {laplacian_var:.1f})"
        return True, "PASSED"

    def consolidate_group_bounding_box(self, img, all_faces):
        h, w = img.shape[:2]
        min_x, min_y, max_x, max_y = w, h, 0, 0
        for face in all_faces:
            lm = face.landmark
            min_x, min_y = min(min_x, int(lm[234].x * w)), min(min_y, int(lm[10].y * h))
            max_x, max_y = max(max_x, int(lm[454].x * w)), max(max_y, int(lm[152].y * h))
        
        pad_x, pad_top = int((max_x - min_x) * 0.30), int((max_y - min_y) * 0.42)
        crop_x1, crop_y1 = max(0, min_x - pad_x), max(0, min_y - pad_top)
        crop_x2, crop_y2 = min(w, max_x + pad_x), min(h, max_y + int((max_y - min_y) * 0.20))
        return img[crop_y1:crop_y2, crop_x1:crop_x2]

    def add_watermark(self, img, text):
        if not text: return img
        h, w = img.shape[:2]
        font_scale = max(0.5, w / 1500)
        thickness = max(1, int(font_scale * 2))
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        x, y = w - text_w - 20, h - 20
        cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img

    def process_image_pipeline(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            enhanced = self.luminance_denoise(self.adaptive_white_balance(img))
            return self.add_watermark(enhanced, self.watermark_text), "[PASSED] No Face - General Enhancement"
            
        faces = results.multi_face_landmarks
        for idx, face in enumerate(faces):
            passed, reason = self.evaluate_quality_gate(img, face)
            if not passed: 
                return None, f"[REJECTED] Face #{idx+1}: {reason}"
                
        img = self.luminance_denoise(self.adaptive_white_balance(img))
        
        if len(faces) == 1:
            cx, cy = int(faces[0].landmark[1].x * w), int(faces[0].landmark[1].y * h)
            img = self.sharpen_features(img, faces[0], w, h)
            img = self.apply_vignette(img, cx, cy)
            img = self.add_watermark(img, self.watermark_text)
            return img, "[PASSED] Single Face Portrait (Sharpened & Vignetted)"
            
        cropped_group = self.consolidate_group_bounding_box(img, faces)
        cropped_group = self.add_watermark(cropped_group, self.watermark_text)
        return cropped_group, f"[PASSED] Group Photo ({len(faces)} Faces)"


# ================= AUDIO & VIDEO PIPELINE =================
def process_audio_pipeline(video_path, output_audio_path):
    print("  └─ 🎵 Extracting and cleaning audio track...")
    cmd_extract = ["ffmpeg", "-y", "-i", video_path, "-f", "s16le", "-ac", "1", "-ar", "44100", "pipe:1"]
    process = subprocess.Popen(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_audio, _ = process.communicate()
    
    if not raw_audio: 
        raise Exception("No audio stream found.")
        
    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    sr = 44100
    
    n = len(audio)
    fft_data = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mask = np.ones_like(freqs)
    mask[freqs < 80] = 0.0
    mask[freqs > 12000] = 0.0
    audio = np.fft.irfft(fft_data * mask, n)
    
    max_peak = np.max(np.abs(audio))
    if max_peak != 0:
        audio = audio * min((10 ** (-1.0 / 20.0)) / max_peak, 10.0)

    threshold = 10 ** (-1.5 / 20.0)
    over_thresh = np.abs(audio) > threshold
    if np.any(over_thresh):
        compressed = threshold + (1 - threshold) * np.tanh((np.abs(audio) - threshold) / (1 - threshold))
        audio[over_thresh] = np.sign(audio)[over_thresh] * compressed[over_thresh]
        
    int_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    cmd_save = ["ffmpeg", "-y", "-f", "s16le", "-ac", "1", "-ar", str(sr), "-i", "pipe:0", output_audio_path]
    process_save = subprocess.Popen(cmd_save, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    process_save.communicate(input=int_audio.tobytes())

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0

def create_flipbook_gif(video_path, output_gif_path):
    try:
        from PIL import Image
    except ImportError:
        print("  └─ ⚠️ Pillow not installed. Skipping GIF generation.")
        return

    print("  └─ 🎞️ Extracting sharpest flipbook frames...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * 0.4)) 
    
    frames, variances = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if count % frame_interval == 0:
            # Downsample for ultra-fast Laplacian variance scoring
            small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    
    if not frames: return
    
    top_indices = np.argsort(variances)[-min(8, len(frames)):]
    top_indices.sort() 
    
    gif_frames = []
    for idx in top_indices:
        img = Image.fromarray(frames[idx])
        img.thumbnail((480, 480), Image.Resampling.LANCZOS)
        gif_frames.append(img)
        
    gif_frames[0].save(output_gif_path, save_all=True, append_images=gif_frames[1:], duration=350, loop=0)

def process_video_by_mode(input_video, audio_path, master_out, wa_out, mode="1"):
    """
    mode 1: WhatsApp/Web Only (Fast 720p)
    mode 2: Enhanced Master (Native) + WhatsApp
    mode 3: 4K Master (Upscaled/Master 3840x2160) + WhatsApp
    """
    duration = get_video_duration(input_video)
    fade_v = f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, duration-0.5):.2f}:d=0.5"
    fade_a = f"afade=t=in:st=0:d=0.5,afade=t=out:st={max(0, duration-0.5):.2f}:d=0.5"
    
    # Common WhatsApp export command
    print("  └─ 🚀 Rendering Mobile/WhatsApp Video...")
    vf_wa = f"scale=720:-2,{fade_v},eq=brightness=0.02:contrast=1.05:saturation=1.05"
    cmd_wa = ["ffmpeg", "-y", "-fflags", "+genpts", "-i", input_video]
    if audio_path: 
        cmd_wa.extend(["-i", audio_path])
    cmd_wa.extend(["-vf", vf_wa, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"])
    
    if audio_path:
        cmd_wa.extend(["-c:a", "aac", "-b:a", "96k", "-af", fade_a, "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    else:
        cmd_wa.extend(["-c:a", "aac", "-b:a", "96k"])
    cmd_wa.append(wa_out)
    subprocess.run(cmd_wa, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Master Render for Modes 2 and 3
    if mode in ("2", "3"):
        print(f"  └─ 🌟 Rendering {'4K' if mode == '3' else 'Enhanced Standard'} Master Video...")
        vf_master = f"{fade_v},eq=brightness=0.02:contrast=1.05:saturation=1.08"
        if mode == "3":
            vf_master = "scale=3840:2160:flags=lanczos," + vf_master
            
        cmd_master = ["ffmpeg", "-y", "-fflags", "+genpts", "-i", input_video]
        if audio_path: 
            cmd_master.extend(["-i", audio_path])
        cmd_master.extend(["-vf", vf_master, "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p"])
        
        if audio_path:
            cmd_master.extend(["-c:a", "aac", "-b:a", "192k", "-af", fade_a, "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        else:
            cmd_master.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd_master.append(master_out)
        subprocess.run(cmd_master, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ================= MAIN EXECUTION BLOCK =================
if __name__ == "__main__":
    print("="*60)
    print("    EVENT MEDIA AUTO-SORTER & ENHANCEMENT ENGINE   ")
    print("="*60)

    raw_in = INPUT_PATH.strip('"').strip("'")
    if not os.path.exists(raw_in):
        print(f"❌ Input folder missing: {raw_in}")
        sys.exit(1)

    out_folder = OUTPUT_PATH.strip('"').strip("'")
    if not out_folder:
        out_folder = os.path.join(raw_in, "_Sorted_Output")
    os.makedirs(out_folder, exist_ok=True)

    print("\nSelect Output Profile:")
    print("  [1] WhatsApp / Web Only (Fastest Export)")
    print("  [2] Enhanced Master + WhatsApp / Web (Balanced Production)")
    print("  [3] 4K Master + WhatsApp / Web (Ultra High-Res)")
    choice = input("Enter choice (1, 2, or 3, Default: 1): ").strip()
    if choice not in ("1", "2", "3"):
        choice = "1"

    all_files = [f for f in os.listdir(raw_in) if f.lower().endswith(IMAGE_EXTS + VIDEO_EXTS)]
    if not all_files:
        print(f"[WARNING] No valid media files found in '{raw_in}'.")
        sys.exit(0)

    file_time_pairs = [(file, os.path.join(raw_in, file), get_capture_time(os.path.join(raw_in, file))) for file in all_files]
    file_time_pairs.sort(key=lambda x: x[2])

    print(f"\n[FOUND] {len(file_time_pairs)} items. Beginning batch processing with Mode [{choice}]...\n")
    
    retouch_engine = AdaptiveRetouchAndQualityEngine()
    student_count, last_time = 1, None
    
    for index, (file, full_path, current_time) in enumerate(file_time_pairs, start=1):
        if last_time is not None and (current_time - last_time).total_seconds() > TIME_GAP_SECONDS:
            student_count += 1
        last_time = current_time

        # Folder Architecture
        student_dir = os.path.join(out_folder, f"Student_{student_count:03d}")
        web_dir = os.path.join(student_dir, "web_whatsapp_flipbook")
        gif_dir = os.path.join(student_dir, "Digital Flipbook")
        
        for d in [student_dir, web_dir, gif_dir]: 
            os.makedirs(d, exist_ok=True)
        
        ext, base_name = os.path.splitext(file)[1].lower(), os.path.splitext(file)[0]
        
        if ext in VIDEO_EXTS:
            print(f"[{index}/{len(file_time_pairs)}] 🎬 Processing Video ({file}) -> Student_{student_count:03d}/")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio = tmp.name
                
            has_enhanced_audio = False
            try:
                process_audio_pipeline(full_path, temp_audio)
                has_enhanced_audio = True
            except Exception:
                print(f"  └─ ⚠️ Audio enhancement skipped (using original).")
            
            master_prefix = "4K_MASTER_" if choice == "3" else "ENHANCED_"
            master_path = os.path.join(student_dir, f"{master_prefix}{base_name}.mp4")
            wa_path = os.path.join(web_dir, f"MOBILE_{base_name}.mp4")
            gif_path = os.path.join(gif_dir, f"FLIPBOOK_{base_name}.gif")
            
            create_flipbook_gif(full_path, gif_path)
            process_video_by_mode(full_path, temp_audio if has_enhanced_audio else None, master_path, wa_path, mode=choice)
            
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    pass
            print(f"  └─ ✅ Video processing complete.")
                
        elif ext in IMAGE_EXTS:
            print(f"[{index}/{len(file_time_pairs)}] 📸 Processing Photo ({file}) -> Student_{student_count:03d}/")
            img = cv2.imread(full_path)
            if img is not None:
                processed_img, status = retouch_engine.process_image_pipeline(img)
                if processed_img is not None:
                    # Save enhanced original photo
                    out_img_path = os.path.join(student_dir, f"{base_name}.jpg")
                    cv2.imwrite(out_img_path, processed_img)
                    
                    # Save web-sized compressed copy
                    web_h, web_w = processed_img.shape[:2]
                    target_w = min(1920, web_w)
                    web_img = cv2.resize(processed_img, (target_w, int(target_w * (web_h / web_w))), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(web_dir, f"{base_name}_web.jpg"), web_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"  └─ ✅ {status}")
                else:
                    blur_dir = os.path.join(out_folder, "_Blurry_or_Discarded")
                    os.makedirs(blur_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(blur_dir, f"{base_name}.jpg"), img)
                    print(f"  └─ ⚠️ {status} (Moved to _Blurry)")

    print(f"\n[FINISHED] Execution complete. All files organized and processed into: {out_folder}")