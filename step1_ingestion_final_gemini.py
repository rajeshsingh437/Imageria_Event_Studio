"""
IMAGERIA EVENT STUDIO — STEP 1: RAW INGESTION & DYNAMIC ENHANCEMENT (LOCKED PRODUCTION)
File: step1_ingestion_final_gemini.py
Status: LOCKED & PRODUCTION READY
Role: Validates file integrity, quarantines corrupted files, and applies
      spotlight-aware dynamic exposure & highlight protection per frame.

Pair-Programming Production Rules:
- Original Unedited photo saved as: [name]_ORIGINAL.[ext]
- Dynamically Enhanced photo saved as: [name]-ed.jpg
- Default runs on the FULL batch unless --limit N is explicitly passed.
"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
from pathlib import Path

# ================= PRODUCTION CONFIGURATION =================
DEFAULT_INPUT_DIR = r"E:\HTS"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.cr2', '.cr3', '.nef', '.arw', '.dng')
# ============================================================


def validate_and_read_image(fpath: str) -> tuple[bool, np.ndarray | None, str]:
    """
    Stream-inspects image file integrity.
    Returns: (is_valid, bgr_image_array, error_message)
    """
    if not os.path.exists(fpath):
        return False, None, "File does not exist"
    
    file_size = os.path.getsize(fpath)
    if file_size == 0:
        return False, None, "Zero-byte file (corrupted write)"

    try:
        im = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if im is None or im.size == 0:
            return False, None, "OpenCV failed to decode image buffer"
        return True, im, "OK"
    except Exception as e:
        return False, None, f"Decode exception: {str(e)}"


def dynamic_exposure_normalizer(im_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Spotlight-Aware Dynamic Exposure Normalizer:
    - Detects theatrical spotlights (bright subject against dark stage/curtain).
    - If subject is already well-lit, PRESERVES natural stage contrast and does NOT overexpose.
    - Applies soft-knee highlight protection so white shirts/gowns never blow out.
    - Only lifts deep shadows if the subject itself is truly underexposed.
    """
    lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    mean_l = float(np.mean(l_channel))
    p5 = float(np.percentile(l_channel, 5))      # Deepest shadow floor
    p50 = float(np.percentile(l_channel, 50))    # Median brightness
    p90 = float(np.percentile(l_channel, 90))    # Subject highlight level
    p98 = float(np.percentile(l_channel, 98))    # Extreme highlight peaks
    
    treatment_applied = []
    
    # 1. SPOTLIGHT DETECTION
    is_spotlight = (p90 >= 155.0) and (p50 < 115.0)
    
    if is_spotlight:
        treatment_applied.append("Spotlight Scene Detected (Preserved Authentic Stage Look)")
        
        # Soft-knee highlight roll-off: Prevent bright spotlights from clipping white garments/skin
        if p98 > 230.0:
            high_mask = l_channel > 215
            l_channel[high_mask] = 215 + ((l_channel[high_mask] - 215) * 0.55).astype("uint8")
            treatment_applied.append("Highlight Rolloff (Anti-Blowout)")
            
        # Very gentle local contrast
        clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
    elif p90 < 125.0 and mean_l < 75.0:
        # TRUE UNDEREXPOSURE: Both subject and background are dark
        gamma = max(1.10, min(1.40, 95.0 / (mean_l + 1e-6)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_channel = cv2.LUT(l_channel, lut)
        treatment_applied.append(f"Subtle Shadow Recovery (gamma={gamma:.2f})")
        
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        treatment_applied.append("Gentle Micro-Contrast")
    else:
        # BALANCED SCENE
        treatment_applied.append("Natural Exposure Verified (No Global Lift Needed)")
        if p98 > 240.0:
            high_mask = l_channel > 225
            l_channel[high_mask] = 225 + ((l_channel[high_mask] - 225) * 0.60).astype("uint8")
            treatment_applied.append("Gentle Highlight Cushion")
            
    lab[:, :, 0] = l_channel
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    stats = {
        "mean_l": round(mean_l, 1),
        "p5_shadow": round(p5, 1),
        "p90_subject": round(p90, 1),
        "p98_peak": round(p98, 1),
        "is_spotlight": is_spotlight,
        "treatments": treatment_applied
    }
    return enhanced_bgr, stats


def dynamic_color_cast_corrector(im_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Stage Atmosphere Preservation:
    Theatrical backdrops and stage lighting are preserved authentic.
    """
    return im_bgr, {"stage_atmosphere_preserved": True}


def run_step1_ingestion(input_dir: str, sample_limit: int | None = None, output_dir: str | None = None):
    if not output_dir:
        output_dir = os.path.join(input_dir, "Step1_Ingestion_Enhanced")
    quarantine_dir = os.path.join(output_dir, "_Corrupted_or_Unreadable")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(" IMAGERIA EVENT STUDIO — STEP 1: RAW INGESTION & DYNAMIC ENHANCEMENT (LOCKED)")
    print(f" Source Folder     : {input_dir}")
    print(f" Output Destination: {output_dir}")
    print(f" Execution Scope   : {'First ' + str(sample_limit) + ' images' if sample_limit else 'Full Event Batch (All Images)'}")
    print("=" * 80)

    all_files = []
    for root, _, files in os.walk(input_dir):
        if "Stage" in root or "Step1" in root or "Output" in root or "Universal" in root:
            continue
        for f in sorted(files):
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                all_files.append(os.path.join(root, f))

    total_discovered = len(all_files)
    print(f"\n[*] Discovered {total_discovered} total event images.")

    if sample_limit and sample_limit > 0:
        process_files = all_files[:sample_limit]
        print(f"[*] Processing limited batch of {len(process_files)} images.\n")
    else:
        process_files = all_files
        print(f"[*] Processing full production batch of {len(process_files)} images.\n")

    manifest = {
        "total_discovered": total_discovered,
        "processed_count": len(process_files),
        "valid_count": 0,
        "corrupt_count": 0,
        "files": []
    }

    for idx, fpath in enumerate(process_files):
        fname = os.path.basename(fpath)
        stem = Path(fname).stem
        ext = Path(fname).suffix

        # 1. Integrity Check
        is_valid, im_bgr, err_msg = validate_and_read_image(fpath)
        if not is_valid:
            manifest["corrupt_count"] += 1
            os.makedirs(quarantine_dir, exist_ok=True)
            shutil.copy2(fpath, os.path.join(quarantine_dir, fname))
            sys.stdout.write(f"\r [!] [{idx+1:03d}/{len(process_files)}] QUARANTINED: {fname} ({err_msg})\n")
            sys.stdout.flush()
            continue

        manifest["valid_count"] += 1

        # 2. Dynamic Exposure Normalization
        exp_balanced, exp_stats = dynamic_exposure_normalizer(im_bgr)

        # 3. Dynamic Color Cast Preservation
        final_enhanced, color_stats = dynamic_color_cast_corrector(exp_balanced)

        # 4. Save Both Original and Edited Side-by-Side
        orig_dest = os.path.join(output_dir, f"{stem}_ORIGINAL{ext}")
        if not os.path.exists(orig_dest):
            shutil.copy2(fpath, orig_dest)

        enhanced_dest = os.path.join(output_dir, f"{stem}-ed.jpg")
        cv2.imwrite(enhanced_dest, final_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 96])

        file_record = {
            "name": fname,
            "exposure_stats": exp_stats,
            "color_stats": color_stats
        }
        manifest["files"].append(file_record)

        percent = ((idx + 1) / len(process_files)) * 100.0
        mode_str = "Spotlight" if exp_stats.get('is_spotlight') else "Normal"
        sys.stdout.write(f"\r -> [{idx+1:03d}/{len(process_files)}] ({percent:5.1f}%) [{mode_str}] {fname} | Subject: {exp_stats['p90_subject']} | Peak: {exp_stats['p98_peak']}")
        sys.stdout.flush()

    manifest_path = os.path.join(output_dir, "ingestion_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print("\n\n" + "=" * 80)
    print(" STEP 1: INGESTION & DYNAMIC ENHANCEMENT COMPLETE")
    print(f" Total Processed  : {manifest['processed_count']}")
    print(f" Valid & Enhanced : {manifest['valid_count']}")
    print(f" Corrupted Files  : {manifest['corrupt_count']}")
    print(f" Output Directory : {output_dir}")
    print(f" Manifest Report  : {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    target_dir = DEFAULT_INPUT_DIR
    limit = None

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_dir = sys.argv[1]
    if "--limit" in sys.argv:
        try:
            lim_idx = sys.argv.index("--limit") + 1
            limit = int(sys.argv[lim_idx])
        except (IndexError, ValueError):
            limit = None

    run_step1_ingestion(target_dir, sample_limit=limit)
