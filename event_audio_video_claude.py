import os
import sys
import time
import subprocess
import tempfile
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np

# --- Mute AI Backend Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
try:
    import absl.logging
    absl.logging.set_verbosity('error')
except ImportError:
    pass

import mediapipe as mp_lib

# --- Bulletproof MediaPipe Import ---
try:
    import mediapipe.python.solutions as mp_solutions
    from mediapipe.python.solutions import face_mesh as mp_face_mesh  # noqa: F401
except (ImportError, ModuleNotFoundError, AttributeError):
    mp_solutions = mp_lib.solutions

# ================= USER CONFIGURATION =================
INPUT_PATH = r"D:\raw_photos"
OUTPUT_PATH = r""
WATERMARK_TEXT = ""
TIME_GAP_SECONDS = 25
BLUR_THRESHOLD = 80.0
EAR_THRESHOLD = 0.20

# Set to os.cpu_count() - 1 for max throughput, or a fixed number if the
# machine also needs to stay responsive for other work.
WORKER_PROCESSES = max(1, (os.cpu_count() or 2) - 1)
FFMPEG_PROGRESS_INTERVAL_SEC = 2.0  # how often to refresh the live % line
# ======================================================

IMAGE_EXTS = ('.cr2', '.nef', '.arw', '.dng', '.cr3', '.jpg', '.jpeg')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mts')


def log(msg, end="\n"):
    """Timestamped, flushed print so output never sits buffered/invisible."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", end=end, flush=True)


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
        """Edge-preserving chroma & luminance cleanup.

        fastNlMeansDenoisingColored is the single slowest step in the whole
        pipeline on large (24MP+) images -- it scales with pixel count, so a
        full-resolution 6000x4000 photo can take many seconds on its own.
        We denoise a downsampled proxy (same trick already used for white
        balance below) and upscale back; the noise pattern is low-frequency
        enough that this costs almost nothing in visible quality but is 4-10x
        faster on large images.
        """
        h, w = img.shape[:2]
        scale = 1200.0 / max(h, w) if max(h, w) > 1200 else 1.0
        if scale < 1.0:
            small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            denoised_small = cv2.fastNlMeansDenoisingColored(small, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=15)
            return cv2.resize(denoised_small, (w, h), interpolation=cv2.INTER_LINEAR)
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
        if not text:
            return img
        h, w = img.shape[:2]
        font_scale = max(0.5, w / 1500)
        thickness = max(1, int(font_scale * 2))
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        x, y = w - text_w - 20, h - 20
        cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
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
                return None, f"[REJECTED] Face #{idx + 1}: {reason}"

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
def process_audio_pipeline(video_path, output_audio_path, timeout=180):
    """Extract & clean the audio track using ffmpeg's own native filters.

    The previous version pulled raw PCM into Python, ran a rectangular
    (brick-wall) mask in the frequency domain, then a hard tanh compressor.
    A rectangular frequency-domain mask causes ringing/clicking artifacts in
    the time domain (Gibbs phenomenon) -- that ringing is very likely what
    you're hearing as "poor quality" audio. It also round-trips the entire
    audio through Python twice (extract -> numpy -> re-encode), which is
    slow for long recordings.

    This version does everything inside a single ffmpeg process using
    proper filters: a real highpass/lowpass (no ringing), afftdn for
    spectral noise reduction, a gentle compressor, and loudnorm for
    broadcast-standard, consistent loudness. One process, no Python-side
    audio math, noticeably faster and cleaner-sounding.
    """
    af_chain = (
        "highpass=f=80,"
        "lowpass=f=12000,"
        "afftdn=nf=-25,"
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-af", af_chain,
        "-ar", "44100", "-ac", "1",
        output_audio_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg audio extraction timed out after {timeout}s")

    if result.returncode != 0 or not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
        err = result.stderr.decode(errors="ignore").strip().splitlines()
        raise RuntimeError(err[-1] if err else "unknown ffmpeg error (no audio stream?)")


def get_video_duration(video_path, timeout=15):
    """Duration via ffprobe instead of cv2.VideoCapture.

    cv2's FRAME_COUNT/FPS metadata is frequently wrong or zero for
    transport-stream containers like .MTS anyway, and querying it opens
    the same cv2 decoder that can hang on Windows (see note on
    create_flipbook_gif below). ffprobe reads the container header
    directly, is far more reliable for these files, and has its own
    timeout so it can't hang the script either.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        return float(result.stdout.decode().strip())
    except Exception:
        return 0.0


def _decode_flipbook_frames(video_path, frame_interval, result):
    """Runs in a background thread so it can be abandoned on timeout
    instead of freezing the whole script (see create_flipbook_gif).
    cv2.CAP_FFMPEG is requested explicitly -- on Windows, cv2's default
    backend is MSMF, which is known to hang indefinitely (0% CPU, no
    error, just blocked forever) on some .MTS/AVCHD streams. Forcing
    the FFMPEG backend avoids MSMF entirely.
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        result['error'] = 'Could not open video (unsupported/corrupt container?)'
        return
    frames, variances = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    result['frames'] = frames
    result['variances'] = variances
    result['count'] = count


def create_flipbook_gif(video_path, output_gif_path, decode_timeout=45):
    """Builds the flipbook GIF by decoding the video frame-by-frame.

    Decoding now runs in a background thread with a hard timeout. If the
    underlying OpenCV backend hangs (the MSMF issue described above),
    this no longer freezes the whole run forever -- after decode_timeout
    seconds it logs a warning, skips the GIF for this one file, and lets
    the rest of the pipeline (audio, video render) continue normally.
    """
    try:
        from PIL import Image
    except ImportError:
        log("  └─ ⚠️ Pillow not installed. Skipping GIF generation.")
        return

    log("  └─ 🎞️ Decoding frames for flipbook GIF...")
    t0 = time.time()

    # frame_interval needs fps, which also risks a hang via cv2 -- ask
    # ffprobe instead, same reasoning as get_video_duration().
    duration = get_video_duration(video_path)
    fps = 30.0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15
        )
        num, den = probe.stdout.decode().strip().split('/')
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    except Exception:
        pass
    frame_interval = max(1, int(fps * 0.4))

    result = {}
    t = threading.Thread(target=_decode_flipbook_frames, args=(video_path, frame_interval, result), daemon=True)
    t.start()
    t.join(timeout=decode_timeout)

    if t.is_alive():
        log(f"  └─ ⚠️ Frame decoder hung on this file after {decode_timeout}s (likely a Windows/MSMF codec "
            f"issue with this container) — skipping flipbook GIF, continuing with the rest of the pipeline.")
        return

    if 'error' in result:
        log(f"  └─ ⚠️ {result['error']} — skipping flipbook GIF.")
        return

    frames, variances, count = result.get('frames', []), result.get('variances', []), result.get('count', 0)

    if not frames:
        log(f"  └─ ⚠️ No frames decoded from {video_path} — skipping GIF.")
        return

    log(f"  └─ 🎞️ Decoded {count} frames in {time.time() - t0:.1f}s, building GIF...")

    top_indices = np.argsort(variances)[-min(8, len(frames)):]
    top_indices.sort()

    gif_frames = []
    for idx in top_indices:
        img = Image.fromarray(frames[idx])
        img.thumbnail((480, 480), Image.Resampling.LANCZOS)
        gif_frames.append(img)

    gif_frames[0].save(output_gif_path, save_all=True, append_images=gif_frames[1:], duration=350, loop=0)


def _run_ffmpeg_with_progress(cmd, duration, label):
    """Run ffmpeg while streaming a live % line instead of swallowing all
    output.

    This also fixes a real deadlock: the previous version redirected both
    stdout AND stderr to pipes, but only drained stdout in a loop and only
    read stderr *after* that loop finished. ffmpeg -- especially on .MTS
    sources, which very commonly trigger repeated "non-monotonic dts" /
    timestamp warnings -- can write far more to stderr than the OS pipe
    buffer holds (tens of KB). Once that buffer fills, ffmpeg blocks trying
    to write the next line, our loop blocks waiting for stdout that will
    never come because ffmpeg is stalled -- both processes then sit at 0%
    CPU forever. This is almost certainly what you just hit right after
    the GIF finished building. Reproduced it directly and confirmed the
    fix: stderr is now drained concurrently on a background thread so it
    can never back up and stall the process.
    """
    progress_cmd = cmd + ["-progress", "pipe:1", "-nostats"]
    proc = subprocess.Popen(progress_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stderr_lines = []

    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    last_print = 0.0
    for line in proc.stdout:
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.strip().split("=")[1])
                pct = min(100.0, (ms / 1_000_000.0) / max(duration, 0.01) * 100.0)
                now = time.time()
                if now - last_print >= FFMPEG_PROGRESS_INTERVAL_SEC:
                    log(f"  └─ {label}: {pct:5.1f}%", end="\r")
                    last_print = now
            except ValueError:
                pass

    proc.wait()
    stderr_thread.join(timeout=5)
    print()  # move past the \r progress line
    if proc.returncode != 0:
        tail = stderr_lines[-1].strip() if stderr_lines else "unknown ffmpeg error"
        log(f"  └─ ❌ {label} failed: {tail}")
    return proc.returncode


def process_video_by_mode(input_video, audio_path, master_out, wa_out, mode="1"):
    """
    mode 1: WhatsApp/Web Only (Fast 720p)
    mode 2: Enhanced Master (Native) + WhatsApp
    mode 3: 4K Master (Upscaled/Master 3840x2160) + WhatsApp
    """
    duration = get_video_duration(input_video)
    fade_v = f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, duration - 0.5):.2f}:d=0.5"
    fade_a = f"afade=t=in:st=0:d=0.5,afade=t=out:st={max(0, duration - 0.5):.2f}:d=0.5"

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
    _run_ffmpeg_with_progress(cmd_wa, duration, "Mobile/WhatsApp render")

    if mode in ("2", "3"):
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
        label = "4K Master render" if mode == "3" else "Enhanced Master render"
        _run_ffmpeg_with_progress(cmd_master, duration, label)


# ================= PARALLEL PHOTO WORKER =================
_worker_engine = None  # one FaceMesh engine per worker process, created once


def _init_worker():
    global _worker_engine
    _worker_engine = AdaptiveRetouchAndQualityEngine()


def _process_one_photo(args):
    """Runs in a worker process. Does the full read -> pipeline -> save for
    a single photo, so heavy per-photo CPU work (denoise, face mesh,
    sharpening) is spread across cores instead of one file at a time.
    """
    full_path, student_dir, web_dir, base_name, out_folder = args
    img = cv2.imread(full_path)
    if img is None:
        return base_name, "[SKIPPED] Could not read file"

    processed_img, status = _worker_engine.process_image_pipeline(img)
    if processed_img is not None:
        out_img_path = os.path.join(student_dir, f"{base_name}.jpg")
        cv2.imwrite(out_img_path, processed_img)

        web_h, web_w = processed_img.shape[:2]
        target_w = min(1920, web_w)
        web_img = cv2.resize(processed_img, (target_w, int(target_w * (web_h / web_w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(web_dir, f"{base_name}_web.jpg"), web_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        blur_dir = os.path.join(out_folder, "_Blurry_or_Discarded")
        os.makedirs(blur_dir, exist_ok=True)
        cv2.imwrite(os.path.join(blur_dir, f"{base_name}.jpg"), img)

    return base_name, status


# ================= MAIN EXECUTION BLOCK =================
if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print("    EVENT MEDIA AUTO-SORTER & ENHANCEMENT ENGINE   ")
    print("=" * 60)

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
        log(f"[WARNING] No valid media files found in '{raw_in}'.")
        sys.exit(0)

    file_time_pairs = [(file, os.path.join(raw_in, file), get_capture_time(os.path.join(raw_in, file))) for file in all_files]
    file_time_pairs.sort(key=lambda x: x[2])

    log(f"[FOUND] {len(file_time_pairs)} items. Beginning batch processing with Mode [{choice}] "
        f"using {WORKER_PROCESSES} worker process(es) for photos...")

    # ---- Pass 1 (cheap, sequential): assign student folders in timestamp order ----
    photo_jobs = []
    video_jobs = []
    student_count, last_time = 1, None

    for file, full_path, current_time in file_time_pairs:
        if last_time is not None and (current_time - last_time).total_seconds() > TIME_GAP_SECONDS:
            student_count += 1
        last_time = current_time

        student_dir = os.path.join(out_folder, f"Student_{student_count:03d}")
        web_dir = os.path.join(student_dir, "web_whatsapp_flipbook")
        gif_dir = os.path.join(student_dir, "Digital Flipbook")
        for d in (student_dir, web_dir, gif_dir):
            os.makedirs(d, exist_ok=True)

        ext, base_name = os.path.splitext(file)[1].lower(), os.path.splitext(file)[0]
        if ext in VIDEO_EXTS:
            video_jobs.append((file, full_path, student_dir, web_dir, gif_dir, base_name, student_count))
        elif ext in IMAGE_EXTS:
            photo_jobs.append((full_path, student_dir, web_dir, base_name, out_folder))

    # ---- Pass 2: photos in parallel across CPU cores ----
    if photo_jobs:
        log(f"[PHOTOS] Processing {len(photo_jobs)} photo(s) across {WORKER_PROCESSES} process(es)...")
        done = 0
        with ProcessPoolExecutor(max_workers=WORKER_PROCESSES, initializer=_init_worker) as pool:
            futures = {pool.submit(_process_one_photo, job): job for job in photo_jobs}
            for future in as_completed(futures):
                base_name, status = future.result()
                done += 1
                log(f"  └─ [{done}/{len(photo_jobs)}] {base_name}: {status}")

    # ---- Pass 3: videos sequentially (ffmpeg already multithreads internally) ----
    for i, (file, full_path, student_dir, web_dir, gif_dir, base_name, s_count) in enumerate(video_jobs, start=1):
        log(f"[VIDEO {i}/{len(video_jobs)}] 🎬 {file} -> Student_{s_count:03d}/")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio = tmp.name

        log("  └─ 🎵 Extracting & cleaning audio...")
        t_audio = time.time()
        has_enhanced_audio = False
        try:
            process_audio_pipeline(full_path, temp_audio)
            has_enhanced_audio = True
            log(f"  └─ 🎵 Audio cleaned in {time.time() - t_audio:.1f}s")
        except Exception as e:
            log(f"  └─ ⚠️ Audio enhancement skipped ({e}); using original audio.")

        master_prefix = "4K_MASTER_" if choice == "3" else "ENHANCED_"
        master_path = os.path.join(student_dir, f"{master_prefix}{base_name}.mp4")
        wa_path = os.path.join(web_dir, f"MOBILE_{base_name}.mp4")
        gif_path = os.path.join(gif_dir, f"FLIPBOOK_{base_name}.gif")

        t0 = time.time()
        create_flipbook_gif(full_path, gif_path)
        process_video_by_mode(full_path, temp_audio if has_enhanced_audio else None, master_path, wa_path, mode=choice)
        log(f"  └─ ✅ Done in {time.time() - t0:.1f}s")

        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass

    log(f"[FINISHED] {len(photo_jobs)} photo(s), {len(video_jobs)} video(s) in "
        f"{time.time() - start_time:.1f}s -> {out_folder}")